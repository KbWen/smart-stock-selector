import sys
import os
import pandas as pd

# Add parent directory to path to find 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from core import config

from core.data import get_db_connection, load_from_db
from core.features import compute_all_indicators
from core.ai import train_and_save

# Setup Logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading stock data for AI training...")
    conn = get_db_connection()
    tickers = pd.read_sql("SELECT DISTINCT ticker FROM stock_history", conn)['ticker'].tolist()
    conn.close()
    
    logger.info(f"Found {len(tickers)} stocks.")
    
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
            logger.info(f"Loaded {count} stocks...")
    
    logger.info(f"Total stocks with sufficient data: {count}")
    
    train_and_save(all_dfs)
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
