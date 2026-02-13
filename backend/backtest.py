import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import get_all_tw_stocks, fetch_stock_data
from core.analysis import (
    calculate_rise_score, calculate_rsi, calculate_macd, 
    calculate_smas, calculate_kd, calculate_bollinger, calculate_atr
)
from core.ai import predict_prob

def compute_all_indicators(df):
    """Compute all indicators on a specific dataframe slice."""
    df = df.copy()
    df['rsi'] = calculate_rsi(df)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    df = calculate_smas(df)
    df = calculate_kd(df)
    df = calculate_bollinger(df)
    df['atr'] = calculate_atr(df)
    return df

def run_time_machine(days_ago=30, limit=20):
    """
    Simulates Top Picks from 'days_ago' and calculates their actual return until now.
    """
    print(f"⏳ Time Machine Started: Traveling back {days_ago} days...")
    
    print(f"⏳ Time Machine Started: Traveling back {days_ago} days...")
    
    # OPTIMIZATION: Instead of scanning ALL stocks (which triggers massive API calls),
    # we only scan stocks that we track (in DB).
    # This prevents the "hanging" issue when data isn't cached.
    from core.data import get_top_scores_from_db
    
    # Get more candidates than limit to account for data gaps
    candidates = get_top_scores_from_db(limit=limit*2, sort_by='score')
    
    if not candidates:
        # Fallback if DB is empty (first run)
        candidates = get_all_tw_stocks()[:limit]
    
    from concurrent.futures import ThreadPoolExecutor
    
    def process_stock(stock):
        ticker = stock["ticker"] if "ticker" in stock else stock["code"]
        name = stock.get("name", ticker)
        try:
            # FORCE DOWNLOAD = FALSE. Only use local data to be fast.
            # If user wants full backtest, they must Sync first.
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
            
            df_past = compute_all_indicators(df_past)
            score_data = calculate_rise_score(df_past)
            ai_result = predict_prob(df_past)
            ai_prob = ai_result.get('prob', 0.0) if isinstance(ai_result, dict) else (ai_result or 0.0)
            
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
        except:
            return None

    results = []
    results = []
    # Use fewer workers since we are mostly reading DB now
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_stock, s) for s in candidates]
        for future in futures:
            res = future.result()
            if res:
                results.append(res)

    # 4. RANKING
    # We want to see: "If I bought the Top 10 AI picks, what happened?"
    # Sort by AI Probability (desc)
    df_res = pd.DataFrame(results)
    
    if df_res.empty:
        return {"error": "No data found for backtest"}
        
    df_res = df_res.sort_values(by="ai_prob_at_entry", ascending=False)
    
    top_picks = df_res.head(20).to_dict('records')
    
    # Summary Stats for Top Picks (Top 20)
    avg_return = df_res.head(20)['actual_return'].mean()
    win_count = len(df_res.head(20)[df_res.head(20)['actual_return'] > 0])
    win_rate = win_count / len(top_picks) if len(top_picks) > 0 else 0
    
    return {
        "days_ago": days_ago,
        "simulated_date": results[0]['entry_date'] if results else "N/A",
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
