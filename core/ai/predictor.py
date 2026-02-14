import numpy as np
import pandas as pd
import joblib
import os
import json
from typing import Optional
from core.ai.common import FEATURE_COLS, MODEL_PATH

# Global variable to track the currently loaded model version
CURRENT_MODEL_VERSION = "unknown"
_model_cache = {}

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

def predict_prob(df, version: Optional[str] = None):
    """
    Predicts buy probability. Supports specific version loading with caching.
    """
    global CURRENT_MODEL_VERSION
    
    # 1. Determine Path
    target_path = MODEL_PATH
    if version and version != "latest":
        # Extract timestamp from version tag (e.g. v4.20260213_2240 -> 20260213_2240)
        ts = version.split('.')[-1]
        base_dir = os.path.dirname(MODEL_PATH)
        name_part = os.path.splitext(os.path.basename(MODEL_PATH))[0]
        # Match pattern like model_sniper_20260213_2240.pkl
        target_path = os.path.join(base_dir, f"{name_part}_{ts}.pkl")
        if not os.path.exists(target_path):
            target_path = MODEL_PATH # Fallback

    # 2. Load Model (with Cache)
    if target_path in _model_cache:
        model_data_all = _model_cache[target_path]
    else:
        if not os.path.exists(target_path):
            return None
        try:
            model_data_all = joblib.load(target_path)
            _model_cache[target_path] = model_data_all
        except Exception:
            return None

    # 3. Extract Model components
    if isinstance(model_data_all, dict) and 'ensemble' in model_data_all:
        CURRENT_MODEL_VERSION = model_data_all.get('version', 'unknown')
        model_data = model_data_all['ensemble']
    else:
        CURRENT_MODEL_VERSION = "legacy"
        model_data = model_data_all
    
    if df.empty or len(df) < 60:
        return None
    
    # --- Feature Extraction ---
    try:
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
            'macd_rel': latest['macd'] / close,
            'macd_hist_rel': (latest['macd'] - latest['macd_signal']) / close,
            'sma_diff': (latest['sma_20'] - latest['sma_60']) / sma60,
            'price_vs_sma20': (latest['close'] - latest['sma_20']) / sma20,
            'price_vs_sma60': (latest['close'] - latest['sma_60']) / sma60,
            'sma20_slope': (latest['sma_20'] - prev5['sma_20']) / (prev5['sma_20'] if prev5['sma_20'] != 0 else 1),
            'sma60_slope': (latest['sma_60'] - prev5['sma_60']) / (prev5['sma_60'] if prev5['sma_60'] != 0 else 1),
            'return_1d': (latest['close'] - prev['close']) / (prev['close'] if prev['close'] != 0 else 1),
            'return_5d': (latest['close'] - prev5['close']) / (prev5['close'] if prev5['close'] != 0 else 1),
            'return_10d': (latest['close'] - prev10['close']) / (prev10['close'] if prev10['close'] != 0 else 1),
            'vol_ratio': latest['volume'] / vol_ma if vol_ma != 0 else 1,
            'atr_norm': atr / close,
            'bb_width': latest['bb_width'],
            'bb_percent': latest['bb_percent'],
            'k': latest['k'],
            'd': latest['d'],
            'kd_diff': latest['k'] - latest['d'],
        }
        
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
        
        if isinstance(model_data, dict):
            # Ensemble Voting
            probs = {}
            total_prob = 0
            count = 0
            for name, clf in model_data.items():
                p_array = clf.predict_proba(X_single)[0]
                win_p = float(np.sum(p_array[1:])) if len(p_array) > 1 else 0.0
                probs[name] = win_p
                total_prob += win_p
                count += 1
            return {"prob": total_prob / count if count > 0 else 0, "details": probs}
        else:
            # Single model
            p_vec = model_data.predict_proba(X_single)[0]
            win_p = float(np.sum(p_vec[1:])) if len(p_vec) > 1 else 0.0
            return {"prob": win_p, "details": {"legacy": win_p}}
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None
