import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import os
import twstock
import time
import json
from datetime import datetime, timedelta
from functools import lru_cache
from core import config

def get_db_connection():
    conn = sqlite3.connect(config.DB_PATH, timeout=config.DB_TIMEOUT) 
    conn.row_factory = sqlite3.Row # Return dict-like rows
    return conn

def standardize_ticker(ticker: str) -> str:
    """Standardizes Taiwan stock tickers to numeric codes (e.g., 2454.TW -> 2454)."""
    if not ticker: return ticker
    # Strip common Taiwan suffixes
    for suffix in ['.TW', '.TWO']:
        if ticker.upper().endswith(suffix):
            return ticker[:-len(suffix)]
    return ticker

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Enable Write-Ahead Logging for concurrency
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_history (ticker, date)")
    
    # Ensure table exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_history (
            ticker TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
    ''')
    # Create scores table for fast ranking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_scores (
            ticker TEXT,
            total_score REAL,
            trend_score REAL,
            momentum_score REAL,
            volatility_score REAL,
            last_price REAL,
            change_percent REAL,
            ai_probability REAL,
            model_version TEXT,
            updated_at TIMESTAMP,
            PRIMARY KEY (ticker, model_version)
        )
    ''')
    
    # Create indicators table for caching computation results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_indicators (
            ticker TEXT PRIMARY KEY,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            ema_20 REAL,
            ema_50 REAL,
            sma_20 REAL,
            sma_60 REAL,
            k_val REAL,
            d_val REAL,
            atr REAL,
            model_version TEXT,
            updated_at TIMESTAMP
        )
    ''')
    
    # Migration: Check if columns exist
    try:
        cursor.execute("SELECT ai_probability FROM stock_scores LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE stock_scores ADD COLUMN ai_probability REAL")
        
    try:
        cursor.execute("SELECT model_version FROM stock_scores LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating DB: Adding model_version to stock_scores...")
        cursor.execute("ALTER TABLE stock_scores ADD COLUMN model_version TEXT")

    try:
        cursor.execute("SELECT model_version FROM stock_indicators LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating DB: Adding model_version to stock_indicators...")
        cursor.execute("ALTER TABLE stock_indicators ADD COLUMN model_version TEXT")
        
    conn.commit()
    conn.close()

def save_indicators_to_db(ticker, df, **kwargs):
    """Saves the last row of indicators to DB for fast scanning."""
    if df.empty: return
    last = df.iloc[-1]
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO stock_indicators (
                ticker, rsi, macd, macd_signal, ema_20, ema_50, sma_20, sma_60, k_val, d_val, atr, model_version, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker, 
            last.get('rsi'), last.get('macd'), last.get('macd_signal'),
            last.get('ema_20'), last.get('ema_50'),
            last.get('sma_20'), last.get('sma_60'),
            last.get('k'), last.get('d'),
            last.get('atr'),
            kwargs.get('model_version'),
            datetime.now()
        ))
        conn.commit()
    except Exception as e:
        print(f"Error saving indicators for {ticker}: {e}")
    finally:
        conn.close()

def load_indicators_from_db(ticker):
    """Loads cached indicators for a specific ticker."""
    conn = get_db_connection()
    query = 'SELECT * FROM stock_indicators WHERE ticker = ?'
    try:
        df = pd.read_sql(query, conn, params=(ticker,))
        if not df.empty:
            record = df.to_dict('records')[0]
            # Sanitize for JSON
            for k, v in record.items():
                if pd.isna(v): record[k] = None
            return record
        return None
    except Exception:
        return None
    finally:
        conn.close()

# Manual TTL Cache for TW Stocks
_tw_stocks_cache = {
    "data": None,
    "last_updated": 0
}

def get_all_tw_stocks():
    """Returns a list of all TWSE stock codes. Caches for 1 hour in memory and 24h in file."""
    now = time.time()
    
    # Memory Cache
    if _tw_stocks_cache["data"] and (now - _tw_stocks_cache["last_updated"] < config.CACHE_DURATION):
        return _tw_stocks_cache["data"]

    # File Cache
    if os.path.exists(config.STOCK_LIST_CACHE):
        try:
            mtime = os.path.getmtime(config.STOCK_LIST_CACHE)
            if now - mtime < 86400: # 1 Day TTL for file cache
                with open(config.STOCK_LIST_CACHE, 'r', encoding='utf-8') as f:
                    stocks = json.load(f)
                    _tw_stocks_cache["data"] = stocks
                    _tw_stocks_cache["last_updated"] = now
                    return stocks
        except Exception:
            pass

    stocks = []
    # Generation from twstock (Slow)
    for code, info in twstock.codes.items():
        if info.type == '股票' and info.market == '上市':
            stocks.append({
                "code": code,
                "name": info.name
            })
    
    # Save to File Cache
    try:
        with open(config.STOCK_LIST_CACHE, 'w', encoding='utf-8') as f:
            json.dump(stocks, f, ensure_ascii=False)
    except Exception:
        pass

    _tw_stocks_cache["data"] = stocks
    _tw_stocks_cache["last_updated"] = now
    return stocks

def get_stock_name_from_db(ticker: str) -> str:
    """Returns the name of a stock given its ticker."""
    # First try twstock dictionary (fastest)
    code_only = standardize_ticker(ticker)
    if code_only in twstock.codes:
        return twstock.codes[code_only].name
    return None

def save_to_db(ticker, df):
    if df.empty: return
    
    conn = get_db_connection()
    try:
        df = df.reset_index() if 'date' not in df.columns else df
        if not pd.api.types.is_string_dtype(df['date']):
             df['date'] = df['date'].dt.strftime('%Y-%m-%d')
             
        # Records
        records = df[['date', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
        # Add standardized ticker to each record
        std_ticker = standardize_ticker(ticker)
        for r in records: r['ticker'] = std_ticker
            
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO stock_history (ticker, date, open, high, low, close, volume)
            VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
        ''', records)
        conn.commit()
    except Exception as e:
        print(f"DB Error {ticker}: {e}")
    finally:
        conn.close()

def load_from_db(ticker: str, days: int = 365) -> pd.DataFrame:
    ticker = standardize_ticker(ticker)
    conn = get_db_connection()
    
    # Optimization: Filter by date to avoid loading full history
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    query = 'SELECT * FROM stock_history WHERE ticker = ? AND date >= ? ORDER BY date ASC'
    
    try:
        df = pd.read_sql(query, conn, params=(ticker, start_date))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_stock_data(ticker: str, days: int = 365, force_download: bool = False) -> pd.DataFrame:
    # 1. Try DB
    if not force_download:
        df = load_from_db(ticker, days)
        if not df.empty:
            last_date = df.iloc[-1]['date']
            # If data is fresh (updated today or yesterday), return it
            if datetime.now() - last_date < timedelta(hours=18): 
                return df

    # 2. Fetch from YFinance with Retry Logic (Exponential Backoff)
    yf_ticker = ticker
    if not ticker.endswith('.TW'):
        yf_ticker = f"{ticker}.TW"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(yf_ticker)
            df = stock.history(period=f"{days}d", timeout=10)
            
            if df.empty:
                # Try .TWO if .TW failed (First attempt only to switch suffix)
                if attempt == 0: 
                    yf_ticker = f"{ticker}.TWO"
                    stock = yf.Ticker(yf_ticker)
                    df = stock.history(period=f"{days}d", timeout=10)
                
            if not df.empty:
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                save_to_db(ticker, df)
                return df
                
        except Exception as e:
            wait_time = (2 ** attempt)  # 1s, 2s, 4s
            print(f"Fetch Error {ticker} (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    print(f"Failed to fetch {ticker} after {max_retries} attempts.")
    return pd.DataFrame()

def save_score_to_db(ticker, score_data, ai_prob=None, model_version=None):
    ticker = standardize_ticker(ticker)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO stock_scores (ticker, total_score, trend_score, momentum_score, volatility_score, last_price, change_percent, ai_probability, model_version, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        ticker, 
        score_data['total_score'],
        score_data['trend_score'],
        score_data['momentum_score'],
        score_data['volatility_score'],
        score_data.get('last_price', 0),
        score_data.get('change_percent', 0),
        ai_prob,
        model_version,
        datetime.now()
    ))
    conn.commit()
    conn.close()

def get_top_scores_from_db(limit=50, sort_by='score', version=None):
    conn = get_db_connection()
    
    order_clause = "total_score DESC"
    if sort_by == 'ai':
        order_clause = "ai_probability DESC"
        
    query = f"SELECT *, updated_at as last_sync FROM stock_scores"
    params = [limit]
    
    if version:
        query += " WHERE model_version = ?"
        params = [version, limit]
    else:
        # Default to latest version in the table if not specified
        query += " WHERE model_version = (SELECT model_version FROM stock_scores ORDER BY updated_at DESC LIMIT 1)"
        params = [limit]

    query += f" ORDER BY {order_clause} LIMIT ?"
    
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    
    # Robust NaN/NaT handling for JSON
    records = df.to_dict('records')
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
    return records

def search_stocks_global(query: str):
    """
    Searches for any stock in the database or twstock list by ticker or name.
    """
    q = f"%{query.upper()}%"
    conn = get_db_connection()
    # Search in historical data or scores to see if we have ANY info
    sql = """
        SELECT DISTINCT ticker FROM stock_history 
        WHERE ticker LIKE ? 
        UNION 
        SELECT ticker FROM stock_scores 
        WHERE ticker LIKE ? OR ticker LIKE ?
        LIMIT 10
    """
    try:
        tickers = [row['ticker'] for row in conn.execute(sql, (q, q, q)).fetchall()]
        results = []
        for t in tickers:
            name = get_stock_name_from_db(t)
            results.append({"ticker": t, "name": name or t})
        return results
    finally:
        conn.close()

init_db()
