from core import config

# ===== SNIPER STRATEGY PARAMETERS =====
PRED_DAYS = config.PRED_DAYS       # Look-ahead window (max 20 trading days)
TARGET_GAIN = config.TARGET_GAIN   # +15% profit target
STOP_LOSS = config.STOP_LOSS     # -5% stop loss

# Win = price hits +15% BEFORE it hits -5% within 20 days
# This is a 3:1 Risk/Reward ratio

MODEL_PATH = config.MODEL_PATH

# ===== FEATURE ENGINEERING =====
FEATURE_COLS = [
    'rsi', 'macd_rel', 'macd_hist_rel',
    'sma_diff', 'price_vs_sma20', 'price_vs_sma60',
    'sma20_slope', 'sma60_slope',
    'return_1d', 'return_5d', 'return_10d',
    'vol_ratio',
    'atr_norm',
    'bb_width', 'bb_percent',
    'k', 'd', 'kd_diff',
    'total_score', 'trend_score', 'momentum_score', 'volatility_score'
]
