import sys
import os
import pandas as pd

# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import get_db_connection, load_from_db
from core.analysis import (
    calculate_rsi, calculate_macd, calculate_smas,
    calculate_kd, calculate_bollinger, calculate_atr
)
from core.ai import train_and_save

def compute_all_indicators(df):
    """Compute ALL indicators needed for AI training."""
    df['rsi'] = calculate_rsi(df)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    df = calculate_smas(df)
    df = calculate_kd(df)
    df = calculate_bollinger(df)
    df['atr'] = calculate_atr(df)
    return df

def main():
    print("Loading stock data for AI training...")
    conn = get_db_connection()
    tickers = pd.read_sql("SELECT DISTINCT ticker FROM stock_history", conn)['ticker'].tolist()
    conn.close()
    
    print(f"Found {len(tickers)} stocks.")
    
    all_dfs = []
    count = 0
    
    for ticker in tickers:
        df = load_from_db(ticker)
        if df.empty or len(df) < 100:
            continue
            
        df = compute_all_indicators(df)
        all_dfs.append(df)
        count += 1
        
        if count % 100 == 0:
            print(f"Loaded {count} stocks...")
    
    print(f"Total stocks with sufficient data: {count}")
    
    train_and_save(all_dfs)
    print("Training complete!")

if __name__ == "__main__":
    main()
