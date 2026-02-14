import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "storage.db"))
DB_TIMEOUT = float(os.getenv("DB_TIMEOUT", 30.0))

# AI Model
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "model_sniper.pkl"))
PRED_DAYS = int(os.getenv("PRED_DAYS", 20))
TARGET_GAIN = float(os.getenv("TARGET_GAIN", 0.15))
STOP_LOSS = float(os.getenv("STOP_LOSS", 0.05))

# System / Backend
# Auto-detect CPU count for concurrency but cap at a safe level for SQLite
CPU_COUNT = os.cpu_count() or 4
CONCURRENCY_WORKERS = int(os.getenv("CONCURRENCY_WORKERS", 5)) 
CACHE_DURATION = int(os.getenv("CACHE_DURATION", 3600))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Rise Score Weights (Sum should be 1.0)
WEIGHT_TREND = float(os.getenv("WEIGHT_TREND", 0.40))
WEIGHT_MOMENTUM = float(os.getenv("WEIGHT_MOMENTUM", 0.30))
WEIGHT_VOLATILITY = float(os.getenv("WEIGHT_VOLATILITY", 0.30))

# Cache Paths
STOCK_LIST_CACHE = os.path.join(BASE_DIR, "stock_list_cache.json")
MARKET_HISTORY_PATH = os.path.join(BASE_DIR, "market_history.json")
MODELS_HISTORY_PATH = os.path.join(BASE_DIR, "models_history.json")
