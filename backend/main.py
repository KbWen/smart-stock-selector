from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import threading
import time
from typing import Optional, List

# Add parent directory to path - MUST BE FIRST for core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config

# Setup Logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from core.data import (
    fetch_stock_data, get_all_tw_stocks, save_score_to_db, 
    get_top_scores_from_db, get_db_connection,
    save_indicators_to_db, load_indicators_from_db, load_from_db,
    get_stock_name
)
from core.analysis import (
    calculate_rise_score, generate_analysis_report
)
from core.features import compute_all_indicators
from core.ai import predict_prob, get_model_version
from core.alerts import check_smart_conditions
from core.market import get_market_status
from backend.backtest import run_time_machine

from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request
from core.utils import safe_float, parse_date
from core.logger import setup_logger

logger = setup_logger("backend")

app = FastAPI(title="Smart Stock Selector")

# Global Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Error Catch: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": f"Server Side Error: {str(exc)}", "path": request.url.path}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if not os.path.exists(frontend_path):
    os.makedirs(frontend_path)

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))

app.mount("/static", StaticFiles(directory=frontend_path, html=True), name="static")

# -- Global Cache Removed: Uses get_all_tw_stocks() with @lru_cache now --


# -- Global State for Sync Progress --
sync_status = {
    "is_syncing": False,
    "total": 0,
    "current": 0,
    "current_ticker": "",
    "last_updated": None
}

def run_sync_task():
    global sync_status
    if sync_status["is_syncing"]:
        logger.warning("Sync task triggered but already running.")
        return

    sync_status["is_syncing"] = True
    logger.info("Starting background sync task...")
    
    try:
        all_stocks = get_all_tw_stocks()
        sync_status["total"] = len(all_stocks)
        sync_status["current"] = 0
        
        sync_lock = threading.Lock()

        def process_stock(stock):
            ticker = stock["code"]
            name = stock["name"]
            
            with sync_lock:
                sync_status["current_ticker"] = f"{ticker} {name}"
                
            try:
                # Smart Sync: Check version AND timestamp
                cached = load_indicators_from_db(ticker)
                current_model_version = get_model_version()
                
                should_skip = False
                if cached:
                    # 1. Check Model Version (MUST match)
                    if cached.get('model_version') == current_model_version:
                        # 2. Check Timestamp (SQLite might return string)
                        updated_at = cached['updated_at']
                        if isinstance(updated_at, str):
                            try:
                                updated_at = datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S.%f')
                            except ValueError:
                                updated_at = datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S')

                        # Skip if updated within last 6 hours
                        if datetime.now() - updated_at < timedelta(hours=6):
                            should_skip = True

                if should_skip:
                    with sync_lock:
                        sync_status["current"] += 1
                    return
                
                # Fetch and Process
                df = fetch_stock_data(ticker, days=200, force_download=True)
                if not df.empty and len(df) >= 60:
                    df = compute_all_indicators(df)
                    score = calculate_rise_score(df)
                    
                    ai_result = predict_prob(df)
                    ai_prob = 0.0
                    if isinstance(ai_result, dict):
                        ai_prob = ai_result.get('prob', 0.0)
                        score['ai_details'] = ai_result.get('details', {})
                    else:
                        ai_prob = ai_result
                        
                    save_score_to_db(ticker, score, ai_prob, model_version=current_model_version)
                    save_indicators_to_db(ticker, df, model_version=current_model_version)
            except Exception:
                logger.exception(f"Sync error for {ticker}")
                
            with sync_lock:
                sync_status["current"] += 1

        # Dynamic workers based on load and config
        num_workers = min(config.CONCURRENCY_WORKERS, len(all_stocks))
        logger.info(f"Syncing {len(all_stocks)} stocks with {num_workers} workers.")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(process_stock, all_stocks)
            
        logger.info("Sync task completed successfully.")
            
    except Exception:
        logger.exception("Fatal error in run_sync_task")
    finally:
        sync_status["is_syncing"] = False
        sync_status["last_updated"] = "Just now"

@app.post("/api/sync")
def trigger_sync(background_tasks: BackgroundTasks):
    if sync_status["is_syncing"]:
        return {"message": "Sync already in progress"}
    
    background_tasks.add_task(run_sync_task)
    return {"message": "Sync started in background"}

@app.get("/api/sync/status")
def get_sync_status():
    return sync_status

@app.get("/api/stocks")
def search_stocks(q: Optional[str] = None):
    all_stocks = get_all_tw_stocks()
    if not q:
        return all_stocks[:50]
    
    q = q.lower()
    filtered = [
        s for s in all_stocks 
        if q in s["code"].lower() or q in s["name"].lower()
    ]
    return filtered[:20]

@app.get("/api/models")
def get_model_history_list():
    """Returns list of available trained model versions."""
    from core.ai import list_available_models
    return list_available_models()

@app.get("/api/search")
def search_stocks_global_api(q: str = Query(..., min_length=1)):
    """Universal search for any stock ticker or name."""
    from core.data import search_stocks_global
    return search_stocks_global(q)

@app.get("/api/init")
def get_init_data():
    """Consolidated endpoint for homepage initialization to reduce round-trips."""
    t0 = time.time()
    
    # 1. Market Status
    from core.market import get_market_status
    market = get_market_status()
    
    # 2. Top Picks (Latest Technical)
    picks = get_top_picks(sort="score")
    
    # 3. Models
    from core.ai import list_available_models
    models = list_available_models()
    
    # 4. Sync Status
    curr_sync = sync_status
    
    total_time = time.time() - t0
    logger.info(f"Consolidated Init took {total_time:.4f}s")
    
    return {
        "market": market,
        "top_picks": picks,
        "models": models,
        "sync": curr_sync,
        "perf_ms": int(total_time * 1000)
    }

@app.get("/api/top_picks")
def get_top_picks(sort: str = "score", version: Optional[str] = None):
    picks = get_top_scores_from_db(limit=50, sort_by=sort, version=version)
    
    if picks:
        # Optimized with name_map cache internally
        result = []
        for p in picks:
            last_price = p.get('last_price', 0) or 0
            ai_prob = p.get('ai_probability') or 0
            
            result.append({
                "ticker": p['ticker'],
                "name": get_stock_name(p['ticker']),
                "ai_probability": ai_prob,
                "model_version": p.get('model_version', 'legacy'),
                "last_sync": p.get('last_sync'),
                "ai_target_price": round(last_price * 1.15, 2) if last_price else 0,
                "ai_stop_price": round(last_price * 0.95, 2) if last_price else 0,
                "score": {
                    "total_score": p['total_score'],
                    "trend_score": p['trend_score'],
                    "momentum_score": p['momentum_score'],
                    "volatility_score": p['volatility_score'],
                    "last_price": last_price,
                    "change_percent": p.get('change_percent', 0) or 0
                }
            })
        return result
    else:
        return []

@app.get("/api/stock/{ticker}")
def get_stock_detail(ticker: str):
    df = fetch_stock_data(ticker)
    if df.empty:
        raise HTTPException(status_code=404, detail="Stock not found")
    
    # Compute ALL indicators
    df = compute_all_indicators(df)
    
    score = calculate_rise_score(df)
    
    # --- Text Analysis ---
    prev_row = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    analysis_report = generate_analysis_report(
        df.iloc[-1], prev_row, 
        score['trend_score'], score['momentum_score'], score['volatility_score']
    )
    score['analysis'] = analysis_report
    
    ai_result = predict_prob(df)
    
    ai_prob = 0.0
    ai_details = {}
    
    if isinstance(ai_result, dict):
        ai_prob = ai_result.get('prob', 0.0)
        ai_details = ai_result.get('details', {})
    elif isinstance(ai_result, float):
        ai_prob = ai_result
        
    score['ai_details'] = ai_details
    
    last_price = score['last_price'] or 0
    
    history = df.tail(30)[['date', 'close', 'volume']].to_dict('records')
    for h in history:
        if hasattr(h['date'], 'strftime'):
            h['date'] = h['date'].strftime('%Y-%m-%d')
    
    return {
        "ticker": ticker,
        "score": score,
        "ai_probability": ai_prob,
        "ai_target_price": round(last_price * 1.15, 2) if last_price else 0,
        "ai_stop_price": round(last_price * 0.95, 2) if last_price else 0,
        "history": history
    }

@app.get("/api/backtest")
def run_backtest_simulation(days: int = 30, version: Optional[str] = None):
    """
    Run 'Time Machine' backtest. Supports specific model version.
    """
    try:
        # Pass version to run_time_machine
        result = run_time_machine(days_ago=days, limit=100, version=version)
        return result
    except Exception as e:
        logger.error(f"Backtest API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market_status")
def market_status():
    """
    Returns market-wide risk metrics (Bull/Bear Ratio, Temp, Sentiment) + History.
    """
    from core.market import save_market_history, get_market_history
    status = get_market_status()
    status["model_version"] = get_model_version()
    
    # Save snapshot to history
    if status.get("bull_ratio") is not None:
        save_market_history(status)
        
    status["history"] = get_market_history()
    return status

@app.post("/api/smart_scan")
def smart_scan(criteria: List[str] = []):
    """
    Scans Top 100 stocks for composite conditions using CACHED indicators where possible.
    """
    candidates = get_top_scores_from_db(limit=100, sort_by="score")
    all_stocks = get_all_tw_stocks()
    name_map = {s['code']: s['name'] for s in all_stocks}
    
    results = []
    
    for c in candidates:
        ticker = c['ticker']
        try:
            # 1. Try to load from CACHE first (Super Fast)
            cached_indicators = load_indicators_from_db(ticker)
            
            if cached_indicators:
                # check_smart_conditions expects a DataFrame or enough info to calculate
                # Let's adjust check_smart_conditions to handle cached data if possible,
                # or just fetch if cache is missing.
                
                # OPTIMIZATION: Use load_from_db instead of fetch_stock_data to avoid network calls.
                # If data is stale, it's better to scan stale data than hang.
                df = load_from_db(ticker) 
                
                if df.empty or len(df) < 60: continue
                
                # Slicing: Only compute indicators for recent history (last 300 days is enough)
                # calculating indicators on 20 years of data is slow and unnecessary.
                if len(df) > 300:
                    df = df.tail(300).copy()
                    
                # We skip compute_all_indicators if we have cache? 
                # Actually check_smart_conditions needs indicators in columns.
                # load_from_db returns OHLCV only. We must compute indicators.
                df = compute_all_indicators(df) 
                
                # Ensure 'c' has required keys before accessing
                if not isinstance(c, dict): continue
                
                ai_prob = c.get('ai_probability', 0)
                if check_smart_conditions(df, ai_prob, criteria):
                    results.append({
                        "ticker": ticker,
                        "name": name_map.get(ticker, ticker),
                        "ai_probability": ai_prob,
                        "model_version": c.get('model_version', 'legacy'),
                        "last_sync": c.get('last_sync'),
                        "score": c, # Return full object for frontend compatibility
                        "price": c.get('last_price', 0),
                        "ai_target_price": round(c.get('last_price', 0) * 1.15, 2),
                        "ai_stop_price": round(c.get('last_price', 0) * 0.95, 2),
                        "matches": criteria
                    })
        except:
            continue
            
    return results

@app.get("/api/health")
def health_check():
    """System Health Check Endpoint"""
    health_status = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "db": "disconnected",
        "model_version": get_model_version(),
        "concurrency_workers": config.CONCURRENCY_WORKERS
    }
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        health_status["db"] = "connected"
    except Exception as e:
        health_status["status"] = "error"
        health_status["db"] = str(e)
        
    return health_status

# ==========================================
# V4.1 SNIPER API ENDPOINTS
# ==========================================

from core.rise_score_v2 import calculate_rise_score_v2
from core.indicators_v2 import compute_v4_indicators
from core.ai import predict_prob

@app.get("/api/v4/sniper/candidates")
def get_v4_candidates(limit: int = 50):
    """
    Returns Top Sniper Candidates using Rise Score 2.0 & AI Lite.
    Optimized for React Frontend (StockList.tsx).
    """
    from core.data import get_top_scores_from_db, get_stock_name
    
    try:
        # Optimized path: Fetch top 50 directly from DB if they have V4 versions
        # We fetch a bit more to handle potential filtering
        raw_candidates = get_top_scores_from_db(limit=limit * 2, sort_by='score')
        
        results = []
        for c in raw_candidates:
            ticker = c['ticker']
            
            # If the score in DB is already from a V4 version, we trust it for the list view
            # This is significantly faster than re-calculating everything
            if c.get('model_version', '').startswith('v4'):
                results.append({
                    "ticker": ticker,
                    "name": get_stock_name(ticker),
                    "price": safe_float(c.get('last_price', 0)),
                    "change_percent": safe_float(c.get('change_percent', 0)),
                    "rise_score": round(safe_float(c['total_score']), 1),
                    "ai_prob": round(safe_float(c.get('ai_probability', 0)) * 100, 1),
                    "trend": round(safe_float(c['trend_score']), 1),
                    "momentum": round(safe_float(c['momentum_score']), 1),
                    "volatility": round(safe_float(c['volatility_score']), 1),
                    "rsi_14": 0, # Placeholders for list view if not stored in scores table
                    "macd_diff": 0,
                    "volume_ratio": 0,
                    "signals": []
                })
                if len(results) >= limit: break
                continue

            # Fallback for old versions: On-the-fly (Slow)
            try:
                df = load_from_db(ticker)
                if df.empty or len(df) < 60: continue
                df = compute_v4_indicators(df)
                df = calculate_rise_score_v2(df)
                latest = df.iloc[-1]
                ai_result = predict_prob(df) 
                ai_prob = ai_result.get('prob', 0) if isinstance(ai_result, dict) else ai_result
                    
                results.append({
                    "ticker": ticker, "name": get_stock_name(ticker),
                    "price": safe_float(latest['close']),
                    "change_percent": safe_float((latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100 if len(df) > 1 else 0),
                    "rise_score": round(safe_float(latest['total_score_v2']), 1),
                    "ai_prob": round(safe_float(ai_prob) * 100, 1),
                    "trend": round(safe_float(latest['trend_score_v2']), 1),
                    "momentum": round(safe_float(latest['momentum_score_v2']), 1),
                    "volatility": round(safe_float(latest['volatility_score_v2']), 1),
                    "rsi_14": round(safe_float(latest.get('rsi', 50)), 1),
                    "macd_diff": round(safe_float(latest.get('macd_hist', 0)), 2),
                    "volume_ratio": round(safe_float(latest.get('rel_vol', 1.0)), 2),
                    "signals": []
                })
            except Exception: continue
            if len(results) >= limit: break
                
        return results
    except Exception as e:
        logger.error(f"API ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v4/stock/{ticker}")
def get_v4_stock_detail(ticker: str):
    """
    Returns detailed analysis for SniperCard.tsx.
    """
    df = fetch_stock_data(ticker)
    if df.empty:
        raise HTTPException(status_code=404, detail="Stock not found")
        
    # V4 Pipeline
    df = compute_v4_indicators(df)
    df = calculate_rise_score_v2(df)
    
    latest = df.iloc[-1]
    
    # AI
    ai_result = predict_prob(df)
    ai_prob = 0.0
    if isinstance(ai_result, dict):
        ai_prob = ai_result.get('prob', 0)
    elif isinstance(ai_result, float):
        ai_prob = ai_result

    # Generate "AI Analyst" Text
    analyst_text = []
    if latest['trend_alignment'] == 1:
        analyst_text.append("✅ **Strong Uptrend**: Price is consistently above SMA20 & SMA60.")
    elif latest['sma20_slope'] > 0:
        analyst_text.append("🌤️ **Recovering**: Price is building momentum above SMA20.")
        
    if 40 <= latest['rsi'] <= 70:
        analyst_text.append("⚡ **Momentum**: RSI is in the bullish zone (40-70).")
    elif latest['rsi'] > 80:
        analyst_text.append("⚠️ **Overheated**: RSI indicates overbought territory.")
        
    if latest['is_squeeze']:
        analyst_text.append("💥 **Squeeze Alert**: Low volatility detected, expecting a major move.")
    elif latest['rel_vol'] > 1.5:
        analyst_text.append("📢 **Volume Spike**: Heavy trading activity detected.")

    return {
        "ticker": ticker,
        "name": get_stock_name(ticker),
        "price": latest['close'],
        "rise_score_breakdown": {
            "total": round(latest['total_score_v2'], 1),
            "trend": round(latest['trend_score_v2'], 1),
            "momentum": round(latest['momentum_score_v2'], 1),
            "volatility": round(latest['volatility_score_v2'], 1)
        },
        "ai_probability": round(ai_prob * 100, 1),
        "analyst_summary": " ".join(analyst_text) if analyst_text else "Market is neutral. Watch for setup signals.",
        "signals": {
            "squeeze": bool(latest['is_squeeze']),
            "golden_cross": bool(latest['kd_cross_flag']),
            "volume_spike": bool(latest['rel_vol'] > 1.5)
        }
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", help="Run sync and exit")
    args = parser.parse_args()
    
    if args.sync:
        print("Starting Sync-Only Mode...")
        run_sync_task()
        print("Sync Complete.")
    else:
        try:
            get_all_tw_stocks() # Pre-cache stocks
            logger.info("Stock list cached successfully.")
        except Exception as e:
            logger.error(f"Failed to cache stock list: {e}")

        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
