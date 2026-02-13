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
