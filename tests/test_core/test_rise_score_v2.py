import pytest
import pandas as pd
import numpy as np
from core.rise_score_v2 import calculate_rise_score_v2

def test_calculate_rise_score_v2_perfect_score():
    """Verify that a stock matching all 'Sniper' criteria gets a high score."""
    data = {
        'trend_alignment': [1],
        'sma20_slope': [0.01],
        'sma60_slope': [0.005],
        'rsi': [50],
        'norm_macd_hist': [0.01],
        'kd_cross_flag': [1],
        'is_squeeze': [1],
        'rel_vol': [2.0],
        'bb_percent': [0.15]
    }
    df = pd.DataFrame(data)
    df = calculate_rise_score_v2(df)
    
    # Expected: 
    # Trend: 20 (align) + 10 (slope20) + 10 (slope60) = 40
    # Momentum: 10 (rsi) + 10 (macd) + 10 (kd) = 30
    # Volatility: 15 (squeeze) + 10 (rel_vol) + 5 (bb_percent) = 30
    # Total: 100
    assert df.loc[0, 'total_score_v2'] == 100

def test_calculate_rise_score_v2_bad_stock():
    """Verify that a poorly performing stock gets a low score."""
    data = {
        'trend_alignment': [0],
        'sma20_slope': [-0.01],
        'sma60_slope': [-0.005],
        'rsi': [20],
        'norm_macd_hist': [-0.01],
        'kd_cross_flag': [0],
        'is_squeeze': [0],
        'rel_vol': [0.5],
        'bb_percent': [0.9]
    }
    df = pd.DataFrame(data)
    df = calculate_rise_score_v2(df)
    
    assert df.loc[0, 'total_score_v2'] == 0

def test_score_v2_empty_df():
    """Ensure it handles empty DataFrames gracefully."""
    df = pd.DataFrame()
    result = calculate_rise_score_v2(df)
    assert result.empty
