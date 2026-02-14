# Operational Workflows

This guide outlines the standard procedures for running and maintaining the Smart Stock Selector system.

## 1. Daily Routine (Automated)

The most common operation is the daily data sync, AI training, and score recalculation.

**Command:**

```cmd
daily_run.bat
```

**What it does:**

1. **Sync**: Fetches latest OHLCV data for 1000+ TWSE stocks (`backend/main.py --sync`).
2. **Train**: Retrains the AI model if data has changed (`core/ai.py`).
3. **Recalc**: Computes V2 Indicators and Rise Scores for all stocks (`backend/recalculate.py`).

## 2. Manual Operations

### A. Sync Data Only

To fetch the latest prices without triggering a full recalc (useful for intraday checks):

```bash
python backend/main.py --sync
```

### B. Force Recalculation

If you changed the scoring logic in `core/rise_score_v2.py` and want to update all existing stocks:

```bash
python backend/recalculate.py
```

### C. Run Backtest Simulation

To run the "Time Machine" from the command line (for debugging):

```python
# Create a temporary script or use python shell
from backend.backtest import run_time_machine
print(run_time_machine(days_ago=30, version="latest"))
```

## 3. Frontend Development

To start the React development server:

```bash
cd frontend/v4
npm run dev
```

The frontend will be available at `http://localhost:5173`.
Ensure the backend is running on `http://localhost:8000` for API calls to work.

## 4. Production Deployment

To serve the app in production mode:

1. **Build Frontend**:

    ```bash
    cd frontend/v4
    npm run build
    ```

    This creates static assets in `frontend/v4/dist`.

2. **Start Backend**:

    ```bash
    python backend/main.py
    ```

    The FastAPI server is configured to serve the static files from `frontend/v4/dist` at the root URL `/`.
