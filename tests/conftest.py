import pytest
import pandas as pd
import numpy as np
import sqlite3
import os
from unittest.mock import MagicMock

@pytest.fixture
def sample_stock_data():
    """Generates 100 days of synthetic stock data with a slight upward trend."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
    data = {
        'date': dates,
        'open': np.linspace(100, 110, 100) + np.random.normal(0, 1, 100),
        'high': np.linspace(102, 112, 100) + np.random.normal(0, 1, 100),
        'low': np.linspace(98, 108, 100) + np.random.normal(0, 1, 100),
        'close': np.linspace(101, 111, 100) + np.random.normal(0, 1, 100),
        'volume': np.random.randint(1000, 5000, 100)
    }
    df = pd.DataFrame(data)
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    return df

@pytest.fixture
def mock_db(tmp_path):
    """Creates a temporary in-memory SQLite database for testing."""
    db_file = tmp_path / "test_stocks.db"
    conn = sqlite3.connect(db_file)
    # Create required tables
    conn.execute("CREATE TABLE stock_history (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)")
    conn.execute("CREATE TABLE stock_scores (ticker TEXT, date TEXT, score_json TEXT, ai_prob REAL, model_version TEXT)")
    conn.commit()
    yield conn
    conn.close()
