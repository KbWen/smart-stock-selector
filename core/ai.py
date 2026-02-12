import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import joblib
import os

# ===== SNIPER STRATEGY PARAMETERS =====
PRED_DAYS = 20       # Look-ahead window (max 20 trading days)
TARGET_GAIN = 0.15   # +15% profit target
STOP_LOSS = 0.05     # -5% stop loss
# Win = price hits +15% BEFORE it hits -5% within 20 days
# This is a 3:1 Risk/Reward ratio

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model_sniper.pkl")

# ===== FEATURE ENGINEERING =====
FEATURE_COLS = [
    'rsi', 'macd', 'macd_hist',
    'sma_diff', 'price_vs_sma20', 'price_vs_sma60',
    'return_1d', 'return_5d', 'return_10d',
    'vol_ratio',           # Volume / MA(Volume, 20)
    'atr_norm',            # ATR / Close (normalized volatility)
    'bb_width', 'bb_percent',  # Bollinger Bands
    'k', 'd', 'kd_diff',  # Stochastic Oscillator
]

def prepare_features(df):
    """
    Creates tabular features + Sniper target for each row.
    Target: 1 = Price hits +15% before hitting -5% within 20 days.
    """
    if df.empty or len(df) < 80:
        return pd.DataFrame(), pd.Series()
    
    df = df.copy()
    
    # --- Check required base indicators ---
    required = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_60', 'k', 'd', 'bb_width', 'bb_percent']
    for col in required:
        if col not in df.columns:
            return pd.DataFrame(), pd.Series()
    
    # --- Derived Features ---
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['sma_diff'] = (df['sma_20'] - df['sma_60']) / df['sma_60'].replace(0, np.nan)
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20'].replace(0, np.nan)
    df['price_vs_sma60'] = (df['close'] - df['sma_60']) / df['sma_60'].replace(0, np.nan)
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    
    vol_ma = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / vol_ma.replace(0, np.nan)
    
    # ATR (normalized)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    df['atr_norm'] = atr / df['close'].replace(0, np.nan)
    
    df['kd_diff'] = df['k'] - df['d']
    
    # --- SNIPER TARGET ---
    # For each row i, look at rows i+1 to i+PRED_DAYS
    # Win = High reaches close * (1 + TARGET_GAIN) BEFORE Low reaches close * (1 - STOP_LOSS)
    targets = []
    for i in range(len(df)):
        entry_price = df.iloc[i]['close']
        target_price = entry_price * (1 + TARGET_GAIN)
        stop_price = entry_price * (1 - STOP_LOSS)
        
        result = 0  # Default: loss/timeout
        
        # Look ahead
        future = df.iloc[i+1 : i+1+PRED_DAYS]
        for _, day in future.iterrows():
            if day['high'] >= target_price:
                result = 1  # WIN - hit profit target first
                break
            if day['low'] <= stop_price:
                result = 0  # LOSS - hit stop loss first
                break
        
        targets.append(result)
    
    df['target'] = targets
    
    # Drop NaN rows
    df_clean = df.dropna(subset=FEATURE_COLS + ['target'])
    
    # Remove the last PRED_DAYS rows (they don't have full future data)
    if len(df_clean) > PRED_DAYS:
        df_clean = df_clean.iloc[:-PRED_DAYS]
    
    if df_clean.empty:
        return pd.DataFrame(), pd.Series()
    
    return df_clean[FEATURE_COLS], df_clean['target']


def train_and_save(all_dfs):
    """
    Trains a GradientBoosting model on all stock data.
    Uses TimeSeriesSplit for validation.
    """
    print("=" * 60)
    print("SNIPER AI - Training (Target: +15% before -5% in 20 days)")
    print("=" * 60)
    
    X_list = []
    y_list = []
    
    for df in all_dfs:
        X, y = prepare_features(df)
        if not X.empty and len(X) > 20:
            X_list.append(X)
            y_list.append(y)
    
    if not X_list:
        print("No valid training data found.")
        return
    
    X_all = pd.concat(X_list).reset_index(drop=True)
    y_all = pd.concat(y_list).reset_index(drop=True)
    
    # Replace inf with NaN, then fill NaN with 0
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"Total samples: {len(X_all)}")
    print(f"Win rate in data: {y_all.mean():.2%}")
    
    # --- Time Series Split for validation ---
    tscv = TimeSeriesSplit(n_splits=3)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]
        
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        clf.fit(X_tr, y_tr)
        
        y_pred = clf.predict(X_val)
        print(f"\n--- Fold {fold+1} ---")
        print(classification_report(y_val, y_pred, target_names=['Loss/Timeout', 'Win (+15%)'], zero_division=0))
    
    # --- Final model: train on ALL data ---
    print("\nTraining final model on all data...")
    final_clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    final_clf.fit(X_all, y_all)
    
    # Feature importance
    importance = pd.Series(final_clf.feature_importances_, index=FEATURE_COLS)
    print("\nFeature Importance (Top 10):")
    print(importance.sort_values(ascending=False).head(10).to_string())
    
    joblib.dump(final_clf, MODEL_PATH)
    print(f"\nSniper model saved to {MODEL_PATH}")
    print("=" * 60)


def predict_prob(df):
    """
    Predicts WIN probability for the LATEST data point.
    Returns: float (0-1) or None if model doesn't exist.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    
    try:
        clf = joblib.load(MODEL_PATH)
    except Exception:
        return None
    
    # Need enough data
    if df.empty or len(df) < 60:
        return None
    
    df = df.copy()
    
    # Check required indicators
    required = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_60', 'k', 'd', 'bb_width', 'bb_percent']
    for col in required:
        if col not in df.columns:
            return None
    
    # Compute derived features for the latest row
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    prev5 = df.iloc[-6] if len(df) > 5 else latest
    prev10 = df.iloc[-11] if len(df) > 10 else latest
    
    vol_ma = df['volume'].rolling(20).mean().iloc[-1]
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]
    
    close = latest['close'] if latest['close'] != 0 else 1
    sma20 = latest['sma_20'] if latest['sma_20'] != 0 else 1
    sma60 = latest['sma_60'] if latest['sma_60'] != 0 else 1
    
    features = {
        'rsi': latest['rsi'],
        'macd': latest['macd'],
        'macd_hist': latest['macd'] - latest['macd_signal'],
        'sma_diff': (latest['sma_20'] - latest['sma_60']) / sma60,
        'price_vs_sma20': (latest['close'] - latest['sma_20']) / sma20,
        'price_vs_sma60': (latest['close'] - latest['sma_60']) / sma60,
        'return_1d': (latest['close'] - prev['close']) / prev['close'] if prev['close'] != 0 else 0,
        'return_5d': (latest['close'] - prev5['close']) / prev5['close'] if prev5['close'] != 0 else 0,
        'return_10d': (latest['close'] - prev10['close']) / prev10['close'] if prev10['close'] != 0 else 0,
        'vol_ratio': latest['volume'] / vol_ma if vol_ma != 0 else 1,
        'atr_norm': atr / close,
        'bb_width': latest['bb_width'],
        'bb_percent': latest['bb_percent'],
        'k': latest['k'],
        'd': latest['d'],
        'kd_diff': latest['k'] - latest['d'],
    }
    
    X_single = pd.DataFrame([features])
    X_single = X_single.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    try:
        prob = clf.predict_proba(X_single)[0]
        # Class 1 = Win
        win_prob = float(prob[1]) if len(prob) > 1 else 0.0
        return win_prob
    except Exception:
        return None
