# Smart Stock Selector Configuration Guide

This document outlines the available configuration variables and how to tune the system's performance and scoring behavior.

## Core Settings

Configuration is managed in `core/config.py` and can be overridden via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `data/stocks.db` | Path to the SQLite database file. |
| `CONCURRENCY_WORKERS` | `4` | Number of parallel threads for data synchronization. |
| `STOCK_SYNC_DAYS` | `365` | Number of days of historical data to fetch for each stock. |
| `CACHE_EXPIRY` | `86400` | Expiry time for stock list and indicator cache (seconds). |

## Rise Score Weights

The Rise Score (Technical Launch Score) is calculated based on three components. You can adjust their relative importance by setting these environment variables:

- `WEIGHT_TREND` (Default: `0.40`) - Component: SMA 20/60 alignment and slope.
- `WEIGHT_MOMENTUM` (Default: `0.30`) - Component: RSI, MACD, and KD Golden Cross.
- `WEIGHT_VOLATILITY` (Default: `0.30`) - Component: Volume Surge and Bollinger Band Squeeze.

> [!NOTE]
> The total weight should usually sum to 1.0 for standard scoring (0-100 range).

## File-Based Caching

The system uses several JSON files to speed up startup and improve the user experience:

- `STOCK_LIST_CACHE`: `data/stock_list_cache.json` - Cached Taiwan stock codes/names.
- `MARKET_HISTORY_CACHE`: `data/market_history.json` - Historical market breadth data for the chart.
- `MODEL_HISTORY_CACHE`: `data/model_history.json` - AI model metadata and version history.

## Database Optimization

The database includes an index on `(ticker, date)` in the `stock_history` table to ensure fast lookups during indicator calculation and scanning.

```sql
CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_history(ticker, date);
```

## AI Model Versions

The system supports multiple AI training versions. Each version's results are stored separately in the `stock_scores` table, allowing for side-by-side comparison in the UI.

- Table: `stock_scores`
- Primary Key: `(ticker, model_version)`
