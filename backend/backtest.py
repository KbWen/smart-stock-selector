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

def run_time_machine(days_ago=30, limit=20):
    """
    Simulates Top Picks from 'days_ago' and calculates their actual return until now.
    """
    print(f"⏳ Time Machine Started: Traveling back {days_ago} days...")
    
    # OPTIMIZATION: Instead of scanning ALL stocks (which triggers massive API calls),
    # we only scan stocks that we track (in DB).
    # We increase the limit significantly (e.g. 300) to find the "True Top 20" from 30 days ago.
    from core.data import get_top_scores_from_db
    
    # Get 500 candidates to ensure we find those that were "Hot" 30 days ago
    candidates = get_top_scores_from_db(limit=500, sort_by='ai')
    
    if not candidates:
        # Fallback if DB is empty (first run)
        candidates = get_all_tw_stocks()[:100]
    
    from concurrent.futures import ThreadPoolExecutor
    
    def process_stock(stock):
        ticker = stock["ticker"] if "ticker" in stock else stock["code"]
        try:
            # 1. Name Lookup (Improved)
            from core.data import get_stock_name_from_db
            name = stock.get("name")
            if not name or name == ticker:
                name = get_stock_name_from_db(ticker)
                
            # Final Fallback if still None or ticker
            if not name or name == ticker:
                # Try to get from all stocks list (slow but reliable for backtest)
                all_stocks = get_all_tw_stocks() 
                for s in all_stocks:
                    if s['code'] == ticker:
                        name = s['name']
                        break
            
            name = name or ticker

            # FORCE DOWNLOAD = FALSE. Only use local data to be fast.
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
            if entry_price < 15:
                return None
            
            # Compute indicators on the PAST slice
            df_past = compute_all_indicators(df_past)
            score_data = calculate_rise_score(df_past)
            
            # AI Probability (P0 Fix: Improved checking)
            ai_result = predict_prob(df_past)
            if ai_result is None:
                ai_prob = 0.0
            elif isinstance(ai_result, dict):
                ai_prob = ai_result.get('prob', 0.0)
            else:
                ai_prob = float(ai_result) if ai_result else 0.0
            
            # --- STRATEGY FILTER: AI PROB > 0.40 ---
            if ai_prob < 0.40:
                return None
            
            roi = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            
            return {
                "ticker": ticker,
                "name": name,
                "entry_date": simulated_date,
                "entry_price": entry_price,
                "current_price": current_price,
                "ai_prob_at_entry": ai_prob,
                "rise_score_at_entry": score_data['total_score'],
                "actual_return": roi
            }
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            return None

    results = []
    # Use fewer workers since we are mostly reading DB now
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_stock, s) for s in candidates]
        for future in futures:
            res = future.result()
            if res:
                results.append(res)

    # 4. RANKING
    # We want to see: "If I bought the Top 10 High Confidence AI picks, what happened?"
    df_res = pd.DataFrame(results)
    
    if df_res.empty:
        return {"error": "No stocks met the strategy criteria (AI > 40%, Price > $15)"}
        
    df_res = df_res.sort_values(by="ai_prob_at_entry", ascending=False)
    
    # NEW: Top 10 only as requested for "Concentrated Strategy"
    top_picks = df_res.head(10).to_dict('records')
    
    # Summary Stats for Top Picks
    top_df = df_res.head(10)
    avg_return = top_df['actual_return'].mean()
    win_count = len(top_df[top_df['actual_return'] > 0])
    win_rate = win_count / len(top_picks) if len(top_picks) > 0 else 0
    
    return {
        "days_ago": days_ago,
        "simulated_date": results[0]['entry_date'] if results else "N/A",
        "candidate_pool_size": len(results),
        "ai_prob_max": float(df_res['ai_prob_at_entry'].max() * 100),
        "ai_prob_min": float(top_df['ai_prob_at_entry'].min() * 100),
        "top_picks": top_picks,
        "summary": {
            "avg_return": avg_return,
            "win_rate": win_rate,
            "best_stock": top_picks[0]['name'] if top_picks else "N/A",
            "best_return": top_picks[0]['actual_return'] if top_picks else 0
        }
    }

if __name__ == "__main__":
    # Test run
    print("Running test backtest (10 days ago)...")
    result = run_time_machine(days_ago=10)
    print(result['summary'])
