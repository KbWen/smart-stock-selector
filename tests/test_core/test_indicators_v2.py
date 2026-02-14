import pytest
import pandas as pd
import numpy as np
from core.indicators_v2 import compute_v4_indicators

def test_compute_v4_indicators_basic(sample_stock_data):
    """Test that all expected V2 factors are present in the output."""
    df = compute_v4_indicators(sample_stock_data)
    
    # Check factor presence
    expected_columns = [
        'dist_sma20', 'dist_sma60', 'ma_spread', 
        'sma20_slope', 'sma60_slope', 'trend_alignment',
        'norm_rsi', 'norm_macd_hist', 'kd_cross_flag',
        'rel_vol', 'atr_percent', 'is_squeeze'
    ]
    for col in expected_columns:
        assert col in df.columns, f"Column {col} missing from indicators"

def test_trend_alignment_logic():
    """Verify alignment flag logic (Price > SMA20 > SMA60)."""
    data = {
        'close': [100, 150, 80],
        'sma_20': [90, 140, 100],
        'sma_60': [80, 130, 120],
        'volume': [1000, 1000, 1000],
        'high': [101, 151, 101],
        'low': [99, 149, 79]
    }
    df = pd.DataFrame(data)
    # Mocking compute_v4_indicators call to just test trend factors if needed, 
    # but let's run the whole thing for a small sample.
    from core.indicators_v2 import calculate_trend_factors
    df = calculate_trend_factors(df)
    
    # Index 0: 100 > 90 > 80 -> True (1)
    # Index 1: 150 > 140 > 130 -> True (1)
    # Index 2: 80 < 100 < 120 -> False (0)
    assert df.loc[0, 'trend_alignment'] == 1
    assert df.loc[1, 'trend_alignment'] == 1
    assert df.loc[2, 'trend_alignment'] == 0

def test_bb_squeeze_logic():
    """Verify BB Squeeze detection."""
    # Data with contracting volatility
    n = 100
    closes = [100] * n
    # Simulate width decreasing
    widths = [1.0 - (i * 0.01) for i in range(n)]
    
    df = pd.DataFrame({
        'close': closes,
        'bb_width': widths,
        'volume': [1000] * n
    })
    from core.indicators_v2 import calculate_volatility_factors
    # Need vol_ma20 for calculate_volatility_factors
    df['vol_ma20'] = 1000.0
    df['atr'] = 1.0
    df = calculate_volatility_factors(df)
    
    # The last few points should be in the lowest 20th percentile
    assert df.iloc[-1]['is_squeeze'] == 1
