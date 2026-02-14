import pandas as pd
import numpy as np

def calculate_rise_score_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements Rise Score 2.0 (V4.1 Sniper logic).
    Adds 'trend_score', 'momentum_score', 'volatility_score', and 'total_score' to DF.
    Each branch is capped at its theoretical max weight.
    """
    if df.empty or 'trend_alignment' not in df.columns:
        # Fallback if factors not pre-computed
        return df

    df = df.copy()
    
    # Weights (0-100 scale)
    W_TREND = 40
    W_MOMENTUM = 30
    W_VOLATILITY = 30

    # 1. Trend Score (Max 40)
    # Alignment: 20 pts
    # SMA Slopes: 10 pts each
    trend_score = df['trend_alignment'] * 20
    trend_score += np.where(df['sma20_slope'] > 0, 10, 0)
    trend_score += np.where(df['sma60_slope'] > 0, 10, 0)
    df['trend_score_v2'] = trend_score.clip(0, W_TREND)

    # 2. Momentum Score (Max 30)
    # RSI Setup (40-70 range is ideal for SNIPER): 15 pts
    # MACD Hist Positive: 5 pts
    # KD Golden Cross: 10 pts
    momentum_score = np.where((df['rsi'] >= 40) & (df['rsi'] <= 70), 10, 0)
    momentum_score += np.where((df['rsi'] > 70) & (df['rsi'] <= 80), 5, 0) # Overheat penalty starts > 80
    momentum_score += np.where(df['norm_macd_hist'] > 0, 10, 0)
    momentum_score += df['kd_cross_flag'] * 10
    df['momentum_score_v2'] = momentum_score.clip(0, W_MOMENTUM)

    # 3. Volatility / Setup Score (Max 30)
    # Bollinger Squeeze: 15 pts
    # Relative Volume > 1.5: 10 pts
    # BB Percent Support (Low yet rising): 5 pts
    vol_score = df['is_squeeze'] * 15
    vol_score += np.where(df['rel_vol'] > 1.5, 10, 0)
    vol_score += np.where((df['bb_percent'] >= 0) & (df['bb_percent'] <= 0.3), 5, 0)
    df['volatility_score_v2'] = vol_score.clip(0, W_VOLATILITY)

    # Total Score
    df['total_score_v2'] = df['trend_score_v2'] + df['momentum_score_v2'] + df['volatility_score_v2']
    
    return df
