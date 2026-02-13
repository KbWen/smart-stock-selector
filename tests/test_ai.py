import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from core.ai import predict_prob, prepare_features
from core.features import compute_all_indicators

def test_prepare_features(sample_stock_data):
    """Verify that features are correctly prepared for the AI model."""
    df = compute_all_indicators(sample_stock_data)
    X, y = prepare_features(df)
    
    # Feature columns should match config
    from core.ai import FEATURE_COLS
    for col in FEATURE_COLS:
        assert col in X.columns
    
    assert len(X) == len(y)
    assert not X.isna().any().any()

@patch('joblib.load')
@patch('os.path.exists')
def test_predict_prob_mocked(mock_exists, mock_load, sample_stock_data):
    """Test prediction logic with a mocked model."""
    mock_exists.return_value = True
    
    # Mock model breakdown
    mock_model_gb = MagicMock()
    mock_model_gb.predict_proba.return_value = [[0.8, 0.1, 0.1]] # 20% win
    
    mock_model_rf = MagicMock()
    mock_model_rf.predict_proba.return_value = [[0.7, 0.2, 0.1]] # 30% win
    
    mock_model_mlp = MagicMock()
    mock_model_mlp.predict_proba.return_value = [[0.6, 0.3, 0.1]] # 40% win
    
    mock_load.return_value = {
        'version': 'v4.0-test',
        'ensemble': {
            'gb': mock_model_gb,
            'rf': mock_model_rf,
            'mlp': mock_model_mlp
        }
    }
    
    df = compute_all_indicators(sample_stock_data)
    result = predict_prob(df)
    
    assert result is not None
    assert 'prob' in result
    assert 'details' in result
    # Ave: (0.2 + 0.3 + 0.4) / 3 = 0.3
    assert pytest.approx(result['prob'], 0.01) == 0.3
