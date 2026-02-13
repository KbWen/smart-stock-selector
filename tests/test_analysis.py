import pytest
import pandas as pd
from core.analysis import calculate_rise_score, generate_analysis_report
from core.features import compute_all_indicators

def test_compute_indicators(sample_stock_data):
    """Verify that all required indicators are computed."""
    df = compute_all_indicators(sample_stock_data)
    
    required_cols = [
        'rsi', 'macd', 'macd_signal', 'sma_5', 'sma_20', 'sma_60',
        'ema_20', 'ema_50', 'k', 'd', 'bb_upper', 'bb_lower', 
        'bb_width', 'bb_percent', 'atr', 'vol_ma20'
    ]
    for col in required_cols:
        assert col in df.columns
        # Last 60+ rows should have values for most indicators
        assert not df[col].tail(10).isna().any()

def test_rise_score_calculation(sample_stock_data):
    """Verify the Rise Score logic with synthetic data."""
    df = compute_all_indicators(sample_stock_data)
    score_dict = calculate_rise_score(df)
    
    assert 'total_score' in score_dict
    assert 'trend_score' in score_dict
    assert 'momentum_score' in score_dict
    assert 'volatility_score' in score_dict
    
    # Scores should be within defined range
    assert 0 <= score_dict['trend_score'] <= 40
    assert 0 <= score_dict['momentum_score'] <= 30
    assert 0 <= score_dict['volatility_score'] <= 30
    assert score_dict['total_score'] == score_dict['trend_score'] + score_dict['momentum_score'] + score_dict['volatility_score']

def test_generate_analysis_report(sample_stock_data):
    """Verify text report generation."""
    df = compute_all_indicators(sample_stock_data)
    score_dict = calculate_rise_score(df)
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    report = generate_analysis_report(
        last_row, prev_row, 
        score_dict['trend_score'], score_dict['momentum_score'], score_dict['volatility_score']
    )
    
    assert 'trend' in report
    assert 'momentum' in report
    assert 'setup' in report
    assert isinstance(report['trend'], str)
    assert len(report['trend']) > 0

def test_empty_dataframe_handling():
    """Ensure analysis functions handle empty DataFrames gracefully."""
    empty_df = pd.DataFrame()
    score = calculate_rise_score(empty_df)
    assert score['total_score'] == 0
    assert score['last_price'] == 0
