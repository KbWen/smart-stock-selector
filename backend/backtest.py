import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import get_all_tw_stocks, fetch_stock_data
from core.analysis import calculate_rise_score
from core.features import compute_all_indicators
from core.ai import predict_prob

from core import config

MODEL_PATH = config.MODEL_PATH
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ WARNING: AI model not found at {MODEL_PATH}")

from typing import Optional

def run_time_machine(days_ago=30, limit=20, version: Optional[str] = None):
    """
    Simulates Top Picks from 'days_ago' and calculates their actual return until now.
    Supports specific model version analysis.
    """
    print(f"⏳ Time Machine Started (Version: {version or 'latest'}): Traveling back {days_ago} days...")
    
    # OPTIMIZATION: Instead of scanning ALL stocks (which triggers massive API calls),
    # we only scan stocks that we track (in DB).
    # We increase the limit significantly (e.g. 300) to find the "True Top 20" from 30 days ago.
    from core.data import get_top_scores_from_db
    
    # Check if version is V4
    is_v4 = (version or "").startswith("v4") or (not version) # Default to v4 if none
    
    # Get candidates based on version if possible, else fallback to latest
    candidates = get_top_scores_from_db(limit=limit * 10, sort_by='ai', version=version)
    
    if not candidates:
        # Fallback if DB is empty (first run)
        candidates = get_all_tw_stocks()[:100]
    
    from concurrent.futures import ThreadPoolExecutor
    
    def process_stock(stock):
        ticker = stock["ticker"] if "ticker" in stock else stock["code"]
        try:
            # 1. Name Lookup (Improved)
            from core.data import get_stock_name
            name = stock.get("name") or get_stock_name(ticker) or ticker
            
            df_full = fetch_stock_data(ticker, days=300, force_download=False)
            
            if df_full.empty or len(df_full) < (60 + days_ago):
                return None
                
            current_price = df_full.iloc[-1]['close']
            df_past = df_full.iloc[:-days_ago].copy()
            
            if df_past.empty or len(df_past) < 60:
                return None
                
            entry_date = df_past.iloc[-1]['date']
            simulated_date = entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date)
            entry_price = df_past.iloc[-1]['close']
            
            # --- STRATEGY FILTER: MIN PRICE $15 ---
            if entry_price < 15: # Strategy floor
                return None
            
            # --- Logic Selection ---
            if is_v4:
                from core.indicators_v2 import compute_v4_indicators
                from core.rise_score_v2 import calculate_rise_score_v2
                df_past = compute_v4_indicators(df_past)
                score_data = calculate_rise_score_v2(df_past)
                total_score = score_data.get('total_score_v2', 0)
            else:
                df_past = compute_all_indicators(df_past)
                score_data = calculate_rise_score(df_past)
                total_score = score_data.get('total_score', 0)
            
            # AI Probability (with version support)
            ai_result = predict_prob(df_past, version=version)
            ai_prob = ai_result.get('prob', 0.0) if isinstance(ai_result, dict) else (ai_result or 0.0)
            
            # --- STRATEGY FILTER: AI PROB > 0.40 ---
            if ai_prob < 0.40: # Strategy hurdle
                return None
            
            roi = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            
            return {
                "ticker": ticker,
                "name": name,
                "entry_date": simulated_date,
                "entry_price": entry_price,
                "current_price": current_price,
                "ai_prob_at_entry": ai_prob,
                "rise_score_at_entry": total_score,
                "actual_return": roi
            }
        except Exception as e:
            # print(f"❌ Error processing {ticker}: {e}")
            return None

    results = []
    # Use fewer workers since we are mostly reading DB now
    with ThreadPoolExecutor(max_workers=min(10, len(candidates))) as executor:
        futures = [executor.submit(process_stock, s) for s in candidates]
        for f in futures:
            res = f.result()
            if res: results.append(res)

    # 4. RANKING
    # We want to see: "If I bought the Top 10 High Confidence AI picks, what happened?"
    df_res = pd.DataFrame(results)
    
    if df_res.empty:
        return {"error": "No stocks met requirements", "summary": {"avg_return": 0, "win_rate": 0}}
        
    df_res = df_res.sort_values(by="ai_prob_at_entry", ascending=False)
    
    # NEW: Top 10 only as requested for "Concentrated Strategy"
    top_picks = df_res.head(10).to_dict('records')
    
    # Summary Stats for Top Picks
    top_df = df_res.head(10)
    avg_return = top_df['actual_return'].mean()
    win_count = len(top_df[top_df['actual_return'] > 0])
    
    return {
        "days_ago": days_ago,
        "model_version": version or "latest",
        "simulated_date": results[0]['entry_date'] if results else "N/A",
        "candidate_pool_size": len(results),
        "top_picks": top_picks,
        "summary": {
            "avg_return": avg_return,
            "win_rate": win_count / len(top_picks) if top_picks else 0,
            "best_stock": top_picks[0]['name'] if top_picks else "N/A",
            "best_return": top_picks[0]['actual_return'] if top_picks else 0
        }
    }

if __name__ == "__main__":
    # Test run
    print("Running test backtest (10 days ago)...")
    result = run_time_machine(days_ago=10)
    print(result['summary'])
