import sys
import os
import pandas as pd

# Add parent directory to path to find 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from core import config

from core.data import get_db_connection, save_score_to_db, load_from_db
from core.analysis import (
    calculate_rise_score, generate_analysis_report
)
from core.features import compute_all_indicators
from core.ai import predict_prob, get_model_version

# Setup Logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def recalculate_all():
    logger.info("=" * 60)
    logger.info("Full Score & AI Recalculation")
    logger.info("=" * 60)
    
    conn = get_db_connection()
    tickers = pd.read_sql("SELECT DISTINCT ticker FROM stock_history", conn)['ticker'].tolist()
    conn.close()
    
    logger.info(f"Found {len(tickers)} stocks in history DB.")
    
    count = 0
    updated = 0
    errors = 0
    
    from core.rise_score_v2 import calculate_rise_score_v2
    from core.indicators_v2 import compute_v4_indicators
    
    current_model_version = get_model_version()
    is_v4 = current_model_version.startswith("v4")

    for ticker in tickers:
        try:
            df = load_from_db(ticker)
            if df.empty or len(df) < 60:
                continue
            
            if is_v4:
                # V4.1 Pipeline (Enhanced Sniper)
                df = compute_v4_indicators(df)
                score = calculate_rise_score_v2(df)
            else:
                # V1 Legacy Pipeline
                df = compute_all_indicators(df)
                score = calculate_rise_score(df)
            
            # --- Text Analysis ---
            # Analysis report might need update for V2, but keeping common for now
            prev_row = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
            analysis_report = generate_analysis_report(
                df.iloc[-1], prev_row, 
                score.get('trend_score_v2', score.get('trend_score', 0)),
                score.get('momentum_score_v2', score.get('momentum_score', 0)),
                score.get('volatility_score_v2', score.get('volatility_score', 0))
            )
            score['analysis'] = analysis_report
            
            # 2. AI Probability (ML)
            ai_result = predict_prob(df)
            ai_prob = 0.0
            if isinstance(ai_result, dict):
                ai_prob = ai_result.get('prob', 0.0)
                score['ai_details'] = ai_result.get('details', {})
            else:
                ai_prob = ai_result
            
            save_score_to_db(ticker, score, ai_prob, model_version=current_model_version)
            updated += 1
                
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.error(f"Error {ticker}: {e}")
            
        count += 1
        if count % 100 == 0:
            logger.info(f"Processed {count}/{len(tickers)} (Updated: {updated})...")
            
    logger.info(f"\nRecalculation complete!")
    logger.info(f"  Processed: {count}")
    logger.info(f"  Updated:   {updated}")
    logger.info(f"  Errors:    {errors}")
    logger.info("=" * 60)

if __name__ == "__main__":
    recalculate_all()
