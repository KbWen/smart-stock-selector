import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database
DB_PATH = os.path.join(BASE_DIR, "storage.db")
DB_TIMEOUT = 30.0

# AI Model
MODEL_PATH = os.path.join(BASE_DIR, "model_sniper.pkl")
PRED_DAYS = 20       # Look-ahead window
TARGET_GAIN = 0.15   # +15% profit target
STOP_LOSS = 0.05     # -5% stop loss

# System / Backend
CONCURRENCY_WORKERS = 5  # Reduced from 12 to avoid SQLite locks
CACHE_DURATION = 3600    # 1 hour
LOG_LEVEL = "INFO"
