import sys
import os
import pandas as pd
import time

# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import get_db_connection, save_score_to_db, load_from_db
from core.analysis import (
    calculate_rise_score, calculate_rsi, calculate_macd, 
    calculate_smas, calculate_kd, calculate_bollinger, calculate_atr
)
from core.ai import predict_prob

def compute_all_indicators(df):
    """Compute ALL indicators needed for both Score and AI."""
    df['rsi'] = calculate_rsi(df)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    df = calculate_smas(df)
    df = calculate_kd(df)
    df = calculate_bollinger(df)
    df['atr'] = calculate_atr(df)
    return df

def recalculate_all():
    print("=" * 60)
    print("Full Score & AI Recalculation")
    print("=" * 60)
    
    conn = get_db_connection()
    tickers = pd.read_sql("SELECT DISTINCT ticker FROM stock_history", conn)['ticker'].tolist()
    conn.close()
    
    print(f"Found {len(tickers)} stocks in history DB.")
    
    count = 0
    updated = 0
    errors = 0
    
    for ticker in tickers:
        try:
            df = load_from_db(ticker)
            if df.empty or len(df) < 60:
                continue
            
            # Compute ALL indicators
            df = compute_all_indicators(df)
            
            # 1. Rise Score (Rule-Based)
            score = calculate_rise_score(df)
            
            # 2. AI Probability (ML)
            ai_prob = predict_prob(df)
            
            # Save all stocks that have any signal
            save_score_to_db(ticker, score, ai_prob)
            updated += 1
                
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"Error {ticker}: {e}")
            
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{len(tickers)} (Updated: {updated})...")
            
    print(f"\nRecalculation complete!")
    print(f"  Processed: {count}")
    print(f"  Updated:   {updated}")
    print(f"  Errors:    {errors}")
    print("=" * 60)

if __name__ == "__main__":
    recalculate_all()
