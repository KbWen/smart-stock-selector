import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import os
import shutil
import json
import glob
from datetime import datetime
from core.ai.common import FEATURE_COLS, MODEL_PATH, PRED_DAYS, TARGET_GAIN, STOP_LOSS

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
            
    # --- Add Rise Scores (Vectorized) ---
    from core.analysis import add_rise_scores_to_df
    df = add_rise_scores_to_df(df)
    
    # --- Derived Features (Normalized) ---
    close = df['close'].replace(0, np.nan)
    df['macd_rel'] = df['macd'] / close
    df['macd_hist_rel'] = (df['macd'] - df['macd_signal']) / close
    
    df['sma_diff'] = (df['sma_20'] - df['sma_60']) / df['sma_60'].replace(0, np.nan)
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20'].replace(0, np.nan)
    df['price_vs_sma60'] = (df['close'] - df['sma_60']) / df['sma_60'].replace(0, np.nan)
    
    # Slopes (5-day relative change)
    df['sma20_slope'] = df['sma_20'].pct_change(5)
    df['sma60_slope'] = df['sma_60'].pct_change(5)
    
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
    df['atr_norm'] = atr / close
    
    df['kd_diff'] = df['k'] - df['d']
    
    # --- SNIPER TARGET (MULTI-LABEL) ---
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    targets = [0] * n
    
    for i in range(n - PRED_DAYS):
        entry_price = closes[i]
        target_strong = entry_price * (1 + TARGET_GAIN)
        target_buy = entry_price * (1 + 0.10) # 10% threshold
        stop_price = entry_price * (1 - STOP_LOSS)
        
        result = 0
        for j in range(i + 1, i + 1 + PRED_DAYS):
            if lows[j] <= stop_price:
                result = 0
                break
            if highs[j] >= target_strong:
                result = 2
                break
            if highs[j] >= target_buy:
                result = max(result, 1)
        targets[i] = result
        
    df['target'] = targets
    df_clean = df.dropna(subset=FEATURE_COLS + ['target'])
    if len(df_clean) > PRED_DAYS:
        df_clean = df_clean.iloc[:-PRED_DAYS]
    
    if df_clean.empty:
        return pd.DataFrame(), pd.Series()
    return df_clean[FEATURE_COLS], df_clean['target']

def train_and_save(all_dfs):
    print("=" * 60)
    print("SNIPER AI - Training (Refactored Modular Structure)")
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
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    win_rate_2 = (y_all == 2).astype(float).mean()
    win_rate_1 = (y_all == 1).astype(float).mean()
    win_rate_0 = (y_all == 0).astype(float).mean()
    
    print(f"Total samples: {len(X_all)}")
    print(f"Class Dist: StrongBuy(2): {win_rate_2:.1%}, Buy(1): {win_rate_1:.1%}, Hold(0): {win_rate_0:.1%}")
    
    class_weights = {
        0: 1.0, 
        1: 1.0 / (win_rate_1 if win_rate_1 > 0 else 1),
        2: 2.0 / (win_rate_2 if win_rate_2 > 0 else 1)
    }
    total_w = sum(class_weights.values())
    class_weights = {k: v/total_w * 3 for k, v in class_weights.items()}
    final_weights = y_all.map(class_weights)
    
    print("\nTraining Ensemble (GB + RF + MLP)...")
    clf_gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    clf_gb.fit(X_all, y_all, sample_weight=final_weights)

    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    clf_rf.fit(X_all, y_all)

    clf_mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=1000, early_stopping=True, random_state=42))
    clf_mlp.fit(X_all, y_all)
    
    ensemble_model = {'gb': clf_gb, 'rf': clf_rf, 'mlp': clf_mlp}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    version_tag = f"v4.{timestamp}"
    model_metadata = {'version': version_tag, 'trained_at': datetime.now().isoformat(), 'ensemble': ensemble_model, 'features': FEATURE_COLS}
    
    base_dir = os.path.dirname(MODEL_PATH)
    name_part, ext_part = os.path.splitext(os.path.basename(MODEL_PATH))
    versioned_path = os.path.join(base_dir, f"{name_part}_{timestamp}{ext_part}")
    joblib.dump(model_metadata, versioned_path)
    shutil.copy(versioned_path, MODEL_PATH)

    # History Log
    history_path = os.path.join(base_dir, "models_history.json")
    history_entry = {"timestamp": timestamp, "version": version_tag, "samples": len(X_all), "win_rates": {"hold": round(win_rate_0, 3), "buy": round(win_rate_1, 3), "strong": round(win_rate_2, 3)}}
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f: history = json.load(f)
        except: pass
    history.append(history_entry)
    with open(history_path, 'w') as f: json.dump(history[-50:], f, indent=2)
    
    # Rotation
    pattern = os.path.join(base_dir, f"{name_part}_*{ext_part}")
    files = sorted(glob.glob(pattern))
    if len(files) > 3:
        for f in files[:-3]:
            try: os.remove(f)
            except: pass
    print(f"Model trained and saved as {version_tag}")
