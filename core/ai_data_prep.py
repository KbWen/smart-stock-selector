import pandas as pd
import numpy as np

def generate_sniper_labels(df: pd.DataFrame, window: int = 20, target: float = 0.15, stop: float = -0.05) -> pd.Series:
    """
    Generates 'Win' labels for the Sniper strategy.
    Win = 1 if price reaches +15% before reaching -5% within the 'window' (20 days).
    """
    if len(df) <= window:
        return pd.Series(0, index=df.index)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(df)
    labels = np.zeros(n)

    # We iterate through each day to see its future outcome
    for i in range(n - window):
        entry_price = close[i]
        future_highs = high[i+1 : i+1+window]
        future_lows = low[i+1 : i+1+window]
        
        target_price = entry_price * (1 + target)
        stop_price = entry_price * (1 + stop)
        
        hit_target = -1
        hit_stop = -1
        
        # Find the first day target or stop is hit
        for d in range(window):
            if future_highs[d] >= target_price:
                hit_target = d
            if future_lows[d] <= stop_price:
                hit_stop = d
            
            # If both are hit on the same day, we are conservative and check which happened 
            # In a daily chart we can't be sure, but usually we'd assume stop hit first if 
            # we want to be safe. 
            
            if hit_target != -1 or hit_stop != -1:
                # If target hit and (stop not hit yet OR target hit before stop)
                if hit_target != -1 and (hit_stop == -1 or hit_target < hit_stop):
                    labels[i] = 1
                # Otherwise it's a loss or neutral
                break
                
    return pd.Series(labels, index=df.index)

def prepare_v4_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline to prepare features and labels for V4.1 training.
    """
    from core.indicators_v2 import compute_v4_indicators
    from core.rise_score_v2 import calculate_rise_score_v2
    
    # 1. Compute Indicators & Scores
    df = compute_v4_indicators(df)
    df = calculate_rise_score_v2(df)
    
    # 2. Generate Labels
    df['label_win'] = generate_sniper_labels(df)
    
    # 3. Clean up (Remove rows where we can't know the label yet)
    # The last 20 rows won't have complete labels
    df = df.iloc[:-20]
    
    # 4. Feature Selection (only normalized/clean features)
    feature_cols = [
        'dist_sma20', 'dist_sma60', 'ma_spread', 'sma20_slope', 'sma60_slope',
        'trend_alignment', 'pos_52w', 'norm_rsi', 'norm_macd_hist', 'kd_cross_flag',
        'rel_vol', 'atr_percent', 'is_squeeze', 'bb_percent',
        'trend_score_v2', 'momentum_score_v2', 'volatility_score_v2', 'total_score_v2'
    ]
    
    # Drop rows with NaNs (due to rolling windows)
    df = df.dropna(subset=feature_cols + ['label_win'])
    
    return df[feature_cols + ['label_win']]
