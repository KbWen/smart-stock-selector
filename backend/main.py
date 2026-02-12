from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import asyncio
import time
from typing import Optional, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import fetch_stock_data, get_all_tw_stocks, save_score_to_db, get_top_scores_from_db, get_db_connection
from core.analysis import (
    calculate_rise_score, calculate_rsi, calculate_macd, 
    calculate_smas, calculate_kd, calculate_bollinger, calculate_atr,
    generate_analysis_report
)
from core.ai import predict_prob

from fastapi.responses import FileResponse

app = FastAPI(title="Smart Stock Selector")

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

# -- Global Cache for Stock List --
STOCK_LIST_CACHE = []
LAST_CACHE_UPDATE = 0
CACHE_DURATION = 3600 # 1 hour

def get_cached_stocks():
    global STOCK_LIST_CACHE, LAST_CACHE_UPDATE
    if not STOCK_LIST_CACHE or (time.time() - LAST_CACHE_UPDATE > CACHE_DURATION):
        try:
            STOCK_LIST_CACHE = get_all_tw_stocks()
            LAST_CACHE_UPDATE = time.time()
        except Exception as e:
            if not STOCK_LIST_CACHE: return []
    return STOCK_LIST_CACHE

def compute_all_indicators(df):
    """Compute ALL indicators needed for both Score and AI."""
    df['rsi'] = calculate_rsi(df)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    df = calculate_smas(df)
    df = calculate_kd(df)
    df = calculate_bollinger(df)
    df['atr'] = calculate_atr(df)
    return df

# -- Global State for Sync Progress --
sync_status = {
    "is_syncing": False,
    "total": 0,
    "current": 0,
    "last_updated": None
}

def run_sync_task():
    global sync_status
    sync_status["is_syncing"] = True
    all_stocks = get_cached_stocks()
    sync_status["total"] = len(all_stocks)
    sync_status["current"] = 0
    
    for stock in all_stocks:
        ticker = stock["code"]
        try:
            df = fetch_stock_data(ticker, days=200, force_download=True)
            if not df.empty and len(df) >= 60:
                df = compute_all_indicators(df)
                score = calculate_rise_score(df)
                ai_prob = predict_prob(df)
                save_score_to_db(ticker, score, ai_prob)
        except Exception:
            pass
        sync_status["current"] += 1
        
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
    all_stocks = get_cached_stocks()
    if not q:
        return all_stocks[:50]
    
    q = q.lower()
    filtered = [
        s for s in all_stocks 
        if q in s["code"].lower() or q in s["name"].lower()
    ]
    return filtered[:20]

@app.get("/api/top_picks")
def get_top_picks(sort: str = "score"):
    picks = get_top_scores_from_db(limit=50, sort_by=sort)
    
    if picks:
        all_stocks_list = get_cached_stocks()
        name_map = {s['code']: s['name'] for s in all_stocks_list}
        
        result = []
        for p in picks:
            last_price = p.get('last_price', 0) or 0
            ai_prob = p.get('ai_probability') or 0
            
            result.append({
                "ticker": p['ticker'],
                "name": name_map.get(p['ticker'], p['ticker']),
                "ai_probability": ai_prob,
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
    
    # Save latest calculation to DB
    save_score_to_db(ticker, score, ai_prob)
    
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
            get_cached_stocks()
        except:
            pass

        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
