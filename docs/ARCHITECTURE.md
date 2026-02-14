# System Architecture - Sniper V4.1

## Overview

Smart Stock Selector (Sniper V4.1) is a specialized decision support system for the Taiwan Stock Market (TWSE). It combines technical analysis, machine learning (Ensemble V4), and a modern React-based frontend to identify high-probability "Sniper" setups.

## High-Level Architecture

```mermaid
graph TD
    User[User / Trader] --> Frontend[Frontend (React/Vite)]
    Frontend --> API[Backend API (FastAPI)]
    
    subgraph "Backend Core (Python)"
        API --> Engine[Factor Engine (core/analysis.py)]
        API --> AI[AI Engine (core/ai.py)]
        API --> DB[(SQLite DB)]
        
        Engine --> DB
        AI --> DB
        
        Sync[Daily Sync (scripts)] --> Data[Data Layer (core/data.py)]
        Data --> DB
    end
    
    subgraph "External"
        Data --> TWSE[TWSE / TPEX]
        Data --> Yahoo[Yahoo Finance API]
    end
```

## Directory Structure & Responsibilities

### `backend/` - The API Layer

* **Role**: Serves data to the frontend, handles long-running tasks (backtests), and manages process lifecycles.
* **Key Files**:
  * `main.py`: FastAPI entry point. Defines routes (`/api/v4/...`).
  * `backtest.py`: The "Time Machine" simulation logic.
  * `recalculate.py`: Batch processing script to update scores and indicators.

### `core/` - The Intelligence Layer

* **Role**: Contains all business logic, math, and AI models. This layer is framework-agnostic (could be used by CLI or Web).
* **Key Files**:
  * `data.py`: Data fetching (yfinance/twstock) and database I/O.
  * `analysis.py`: Technical analysis (SMA, RSI, MACD, Bollinger Bands) and Rise Score logic.
  * `ai.py`: Feature engineering, model training (Ensemble), and prediction.
  * `indicators_v2.py`: The V4.1 optimized indicator library.
  * `rise_score_v2.py`: The V4.1 scoring rules engine.

### `frontend/v4/` - The Presentation Layer

* **Role**: A modern, responsive SPA (Single Page Application) built with React, TypeScript, and Tailwind CSS.
* **Key Design**:
  * **Glassmorphism**: Dark mode with translucent panels.
  * **Component-Based**: Reusable widgets like `SniperCard`, `StockList`.
  * **State Management**: React Hooks (`useState`, `useEffect`) for localized state.

## Data Flow

1. **Ingestion (Daily)**:
    * `daily_run.bat` triggers `backend/main.py --sync`.
    * Market data (OHLCV) is fetched for 1000+ tickers and stored in `storage.db`.
2. **Processing**:
    * `recalculate.py` runs immediately after sync.
    * It computes V2 Indicators and V2 Rise Scores for all stocks.
    * Results are **persisted** to the `stock_scores` table to ensure <100ms API response.
3. **Consumption**:
    * User opens Dashboard.
    * `GET /api/v4/sniper/candidates` queries `stock_scores` (filtered by top rank).
    * Frontend renders the list.
4. **AI Prediction**:
    * The system loads the latest `model_sniper.pkl`.
    * It generates a probability ($P_{win}$) for each candidate based on technical features.

## Design Patterns

* **Repository Pattern**: `core/data.py` abstracts all DB interactions.
* **Strategy Pattern**: `core/ai.py` supports multiple model versions and legacy/ensemble switching.
* **Lazy Loading**: The frontend initializes non-critical data (like market charts) asynchronously to speed up First Contentful Paint (FCP).
