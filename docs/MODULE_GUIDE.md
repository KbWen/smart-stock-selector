# Module Guide

This document provides a detailed breakdown of the key Python modules in the system.

## Core Modules (`core/`)

### `core/data.py`

**Purpose**: Handles data acquisition and database interactions.

* `fetch_stock_data(ticker, days)`: Downloads OHLCV data from yfinance or twstock.
* `load_from_db(ticker)`: Retrieves DataFrame from SQLite.
* `save_score_to_db(...)`: Persists V2 scores and AI probabilities.

### `core/analysis.py`

**Purpose**: The legacy technical analysis engine.

* `compute_all_indicators(df)`: Calculates SMA, RSI, MACD, BB (V1 logic).
* `calculate_rise_score(df)`: Computes the V1 Rise Score (0-100).

### `core/indicators_v2.py` (New in V4.1)

**Purpose**: The optimized V2 indicator library.

* `compute_v4_indicators(df)`: High-performance calculation of Trend, Momentum, and Volatility factors.

### `core/ai/` (Package)

**Purpose**: The AI brain, modularized for specialized training and prediction.

* `core/ai/common.py`: Universal features and threshold constants.
* `core/ai/trainer.py`: Model training logic (GB + RF + MLP Ensemble).
* `core/ai/predictor.py`: Probabilistic inference with model version caching.

### `core/logger.py`

**Purpose**: Centralized observability.

* `setup_logger(name)`: Configures rotating file logs (10MB) and console output.
* `AlertHandler`: Triggers `send_alert()` notifications on `ERROR` or `CRITICAL` events.

## Backend Modules (`backend/`)

### `backend/main.py`

**Purpose**: The FastAPI application entry point.

* `GET /api/v4/sniper/candidates`: Returns top-ranked stocks using persistent scores.
* `GET /api/backtest`: Runs the simulation engine.

### `backend/backtest.py`

**Purpose**: Historical simulation logic.

* `run_time_machine(days_ago, version)`: Reconstructs the market state from N days ago and evaluates strategy performance.

### `backend/recalculate.py`

**Purpose**: Batch processing script.

* Iterates through all 1000+ stocks in the DB.
* Computes V2 indicators and scores.
* Updates the `stock_scores` table.
