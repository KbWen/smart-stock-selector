import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import joblib
import os
import shutil
import json
import glob
from datetime import datetime
from core import config

# ===== SNIPER STRATEGY PARAMETERS =====
PRED_DAYS = config.PRED_DAYS       # Look-ahead window (max 20 trading days)
TARGET_GAIN = config.TARGET_GAIN   # +15% profit target
STOP_LOSS = config.STOP_LOSS     # -5% stop loss
# Win = price hits +15% BEFORE it hits -5% within 20 days
# This is a 3:1 Risk/Reward ratio

MODEL_PATH = config.MODEL_PATH

# Global variable to track the currently loaded model version
CURRENT_MODEL_VERSION = "unknown"

# ===== FEATURE ENGINEERING =====
FEATURE_COLS = [
    'rsi', 'macd_rel', 'macd_hist_rel',
    'sma_diff', 'price_vs_sma20', 'price_vs_sma60',
    'sma20_slope', 'sma60_slope',
    'return_1d', 'return_5d', 'return_10d',
    'vol_ratio',
    'atr_norm',
    'bb_width', 'bb_percent',
    'k', 'd', 'kd_diff',
    'total_score', 'trend_score', 'momentum_score', 'volatility_score'
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
    # 2: Strong Buy (+15% before -5%)
    # 1: Buy (+10% before -5%)
    # 0: Hold (Default)
    # Optimization: Use numpy arrays for much faster iteration
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    targets = [0] * n
    
    for i in range(n - PRED_DAYS):
        entry_price = closes[i]
        target_strong = entry_price * 1.15
        target_buy = entry_price * 1.10
        stop_price = entry_price * 0.95
        
        result = 0
        # Look ahead PRED_DAYS
        for j in range(i + 1, i + 1 + PRED_DAYS):
            # Check Stop Loss FIRST
            if lows[j] <= stop_price:
                result = 0
                break
            # Check Strong Buy
            if highs[j] >= target_strong:
                result = 2
                break
            # Check normal Buy
            if highs[j] >= target_buy:
                result = max(result, 1) # Keep looking for 2 unless we hit stop
        
        targets[i] = result
        
    df['target'] = targets
    
    # Drop NaN rows
    df_clean = df.dropna(subset=FEATURE_COLS + ['target'])
    
    if len(df_clean) > PRED_DAYS:
        df_clean = df_clean.iloc[:-PRED_DAYS]
    
    if df_clean.empty:
        return pd.DataFrame(), pd.Series()
    
    return df_clean[FEATURE_COLS], df_clean['target']


def get_model_version():
    """Returns the current model version string."""
    return CURRENT_MODEL_VERSION

def list_available_models():
    """Returns a list of all trained model versions found in the history log."""
    history_path = os.path.join(os.path.dirname(MODEL_PATH), "models_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

from datetime import datetime
def train_and_save(all_dfs):
    print("=" * 60)
    print("SNIPER AI - Training (Optimized V2: Normalized + Slopes)")
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
    
    # --- Time Series Split ---
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Imbalance handling: higher weight for win classes
    # Use class_weight='balanced' or manual weights
    # For simplicity, we'll let the models handle it or use a dict
    class_weights = {
        0: 1.0, 
        1: 1.0 / (win_rate_1 if win_rate_1 > 0 else 1),
        2: 2.0 / (win_rate_2 if win_rate_2 > 0 else 1) # Double weight for strong buy
    }
    # Normalize weights
    total_w = sum(class_weights.values())
    class_weights = {k: v/total_w * 3 for k, v in class_weights.items()}
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]
        
        # Apply weights to training slice
        weights_tr = y_tr.map(class_weights)
        
        clf = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.8,
            random_state=42
        )
        clf.fit(X_tr, y_tr, sample_weight=weights_tr)
        
        y_pred = clf.predict(X_val)
        print(f"\n--- Fold {fold+1} ---")
        print(classification_report(y_val, y_pred, target_names=['Hold', 'Buy (+10%)', 'StrongBuy (+15%)'], zero_division=0))
    
    # --- ENSEMBLE TRAINING (Manual) ---
    # We train models separately because they handle weights differently.
    
    # Calculate weights for final model
    final_weights = y_all.map(class_weights)
    
    # 1. GradientBoosting (Uses sample_weight)
    print("\nTraining GradientBoosting...")
    clf_gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    clf_gb.fit(X_all, y_all, sample_weight=final_weights)

    # 2. RandomForest (Uses class_weight='balanced')
    print("Training RandomForest...")
    from sklearn.ensemble import RandomForestClassifier
    clf_rf = RandomForestClassifier(
        n_estimators=200,    # Tuned: Increased from 100
        max_depth=10,        # Tuned: Increased from 7
        min_samples_split=5, # Tuned: Decreased from 10
        random_state=42,
        class_weight='balanced'
    )
    clf_rf.fit(X_all, y_all)

    # 3. MLP Neural Network (Diff logic, no weights)
    print("Training MLP (Deep Learning)...")
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    
    clf_mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64), # Tuned: Deeper/Wider
            activation='relu',
            solver='adam',
            alpha=0.0001, # Tuned: Less regularization
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
    )
    clf_mlp.fit(X_all, y_all)
    
    # ... (save logic same) ...
    ensemble_model = {
        'gb': clf_gb,
        'rf': clf_rf,
        'mlp': clf_mlp
    }
    
    # Feature importance (from GB)
    print("\nFeature Importance (from GradientBoosting component):")
    try:
        importance = pd.Series(clf_gb.feature_importances_, index=FEATURE_COLS)
        print(importance.sort_values(ascending=False).head(10).to_string())
    except:
        pass
    
    # --- SAVE WITH METADATA & VERSIONING ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    version_tag = f"v4.{timestamp}"
    
    model_metadata = {
        'version': version_tag,
        'trained_at': datetime.now().isoformat(),
        'ensemble': ensemble_model,
        'features': FEATURE_COLS
    }
    
    # 1. Save specific version
    base_dir = os.path.dirname(MODEL_PATH)
    filename = os.path.basename(MODEL_PATH)
    name_part, ext_part = os.path.splitext(filename)
    
    versioned_filename = f"{name_part}_{timestamp}{ext_part}"
    versioned_path = os.path.join(base_dir, versioned_filename)
    
    joblib.dump(model_metadata, versioned_path)
    print(f"\nSaved version: {versioned_path}")
    
    # 2. Update "Latest" (Symlink/Copy behavior)
    shutil.copy(versioned_path, MODEL_PATH)
    print(f"Updated latest model: {MODEL_PATH}")
    
    # 3. Update History Log
    history_path = os.path.join(base_dir, "models_history.json")
    history_entry = {
        "timestamp": timestamp,
        "version": version_tag,
        "samples": len(X_all),
        "win_rates": {
            "hold": round(win_rate_0, 3),
            "buy": round(win_rate_1, 3),
            "strong": round(win_rate_2, 3)
        }
    }
    
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except:
            history = []
            
    history.append(history_entry)
    # Keep last 50 logs
    if len(history) > 50:
        history = history[-50:]
        
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    # 4. Rotation (Keep last 3 files)
    # Pattern: model_sniper_*.pkl
    pattern = os.path.join(base_dir, f"{name_part}_*{ext_part}")
    files = sorted(glob.glob(pattern))
    
    # Keep last 3
    if len(files) > 3:
        for f in files[:-3]:
            try:
                os.remove(f)
                print(f"Rotated (deleted) old model: {f}")
            except Exception as e:
                print(f"Error rotating {f}: {e}")
    
    # Update global version
    global CURRENT_MODEL_VERSION
    CURRENT_MODEL_VERSION = version_tag
    print("=" * 60)


def predict_prob(df):
    if not os.path.exists(MODEL_PATH):
        return None
    
    global CURRENT_MODEL_VERSION
    try:
        data = joblib.load(MODEL_PATH)
        if isinstance(data, dict) and 'version' in data:
            CURRENT_MODEL_VERSION = data['version']
            model_data = data['ensemble']
        else:
            CURRENT_MODEL_VERSION = "legacy"
            model_data = data
    except Exception:
        return None
    
    if df.empty or len(df) < 60:
        return None
    
    df = df.copy()
    required = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_60', 'k', 'd', 'bb_width', 'bb_percent']
    for col in required:
        if col not in df.columns:
            return None
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    prev5 = df.iloc[-6] if len(df) > 5 else latest
    prev10 = df.iloc[-11] if len(df) > 10 else latest
    
    vol_ma = df['volume'].rolling(20).mean().iloc[-1]
    
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
        'macd_rel': latest['macd'] / close,
        'macd_hist_rel': (latest['macd'] - latest['macd_signal']) / close,
        'sma_diff': (latest['sma_20'] - latest['sma_60']) / sma60,
        'price_vs_sma20': (latest['close'] - latest['sma_20']) / sma20,
        'price_vs_sma60': (latest['close'] - latest['sma_60']) / sma60,
        'sma20_slope': (latest['sma_20'] - prev5['sma_20']) / (prev5['sma_20'] if prev5['sma_20'] != 0 else 1),
        'sma60_slope': (latest['sma_60'] - prev5['sma_60']) / (prev5['sma_60'] if prev5['sma_60'] != 0 else 1),
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
    
    # --- Add Rise Scores for Prediction ---
    from core.analysis import calculate_rise_score
    scores = calculate_rise_score(df)
    features.update({
        'total_score': scores['total_score'],
        'trend_score': scores['trend_score'],
        'momentum_score': scores['momentum_score'],
        'volatility_score': scores['volatility_score']
    })
    
    X_single = pd.DataFrame([features])
    X_single = X_single.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    try:
        if isinstance(model_data, dict):
            # Ensemble (Manual Voting)
            # Result is now multi-class [0, 1, 2]
            probs = {}
            total_prob = 0
            count = 0
            for name, clf in model_data.items():
                p_array = clf.predict_proba(X_single)[0]
                # p_array order: [Prob(0), Prob(1), Prob(2)] if all classes exist in training
                # To be safe, check length.
                if len(p_array) == 3:
                    # Combined probability of being a "Buy" or "Strong Buy"
                    win_p = float(p_array[1] + p_array[2])
                elif len(p_array) == 2:
                    win_p = float(p_array[1])
                else:
                    win_p = 0.0
                
                probs[name] = win_p
                total_prob += win_p
                count += 1
            
            final_win_prob = total_prob / count if count > 0 else 0
            
            return {
                "prob": final_win_prob,
                "details": probs
            }
        else:
            # Legacy single model
            prob_vec = model_data.predict_proba(X_single)[0]
            if len(prob_vec) >= 2:
                win_prob = float(np.sum(prob_vec[1:]))
            else:
                win_prob = 0.0
            return {
                "prob": win_prob,
                "details": {"legacy": win_prob}
            }
            
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None

