import yfinance as yf
import pandas as pd
import sqlite3
import os
import twstock
import time
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "../storage.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
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
            ticker TEXT PRIMARY KEY,
            total_score REAL,
            trend_score REAL,
            momentum_score REAL,
            volatility_score REAL,
            last_price REAL,
            change_percent REAL,
            ai_probability REAL,
            updated_at TIMESTAMP
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
            updated_at TIMESTAMP
        )
    ''')
    
    # Migration: Check if ai_probability exists, if not add it
    try:
        cursor.execute("SELECT ai_probability FROM stock_scores LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating DB: Adding ai_probability column...")
        cursor.execute("ALTER TABLE stock_scores ADD COLUMN ai_probability REAL")
        
    conn.commit()
    conn.close()

def save_indicators_to_db(ticker, df):
    """Saves the last row of indicators to DB for fast scanning."""
    if df.empty: return
    last = df.iloc[-1]
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO stock_indicators (
                ticker, rsi, macd, macd_signal, ema_20, ema_50, sma_20, sma_60, k_val, d_val, atr, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker, 
            last.get('rsi'), last.get('macd'), last.get('macd_signal'),
            last.get('ema_20'), last.get('ema_50'),
            last.get('sma_20'), last.get('sma_60'),
            last.get('k'), last.get('d'), # KD columns are 'k' and 'd'
            last.get('atr'),
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
        return df.to_dict('records')[0] if not df.empty else None
    except Exception:
        return None
    finally:
        conn.close()

def get_all_tw_stocks():
    """Returns a list of all TWSE stock codes."""
    stocks = []
    # This loop is slow if run every time. Upper layer caches it.
    for code, info in twstock.codes.items():
        if info.type == '股票' and info.market == '上市':
            stocks.append({
                "code": code,
                "name": info.name
            })
    return stocks

def save_to_db(ticker, df):
    if df.empty: return
    
    conn = get_db_connection()
    try:
        df = df.reset_index() if 'date' not in df.columns else df
        if not pd.api.types.is_string_dtype(df['date']):
             df['date'] = df['date'].dt.strftime('%Y-%m-%d')
             
        # Records
        records = df[['date', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
        # Add ticker to each record
        for r in records: r['ticker'] = ticker
            
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
    conn = get_db_connection()
    query = 'SELECT * FROM stock_history WHERE ticker = ? ORDER BY date ASC'
    try:
        df = pd.read_sql(query, conn, params=(ticker,))
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
            if datetime.now() - last_date < timedelta(days=1):
                return df

    # 2. Fetch from YFinance
    yf_ticker = ticker
    if not ticker.endswith('.TW'):
        yf_ticker = f"{ticker}.TW"
    
    try:
        stock = yf.Ticker(yf_ticker)
        df = stock.history(period=f"{days}d")
        
        if df.empty:
            yf_ticker = f"{ticker}.TWO"
            stock = yf.Ticker(yf_ticker)
            df = stock.history(period=f"{days}d")
            
        if not df.empty:
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            save_to_db(ticker, df)
            
        return df
    except Exception as e:
        print(f"Fetch Error {ticker}: {e}")
        return pd.DataFrame()

def save_score_to_db(ticker, score_data, ai_prob=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO stock_scores (ticker, total_score, trend_score, momentum_score, volatility_score, last_price, change_percent, ai_probability, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        ticker, 
        score_data['total_score'],
        score_data['trend_score'],
        score_data['momentum_score'],
        score_data['volatility_score'],
        score_data['last_price'],
        score_data['change_percent'],
        ai_prob,
        datetime.now()
    ))
    conn.commit()
    conn.close()

def get_top_scores_from_db(limit=50, sort_by='score'):
    conn = get_db_connection()
    
    order_clause = "total_score DESC"
    if sort_by == 'ai':
        order_clause = "ai_probability DESC"
        
    df = pd.read_sql(f'''
        SELECT * FROM stock_scores 
        ORDER BY {order_clause}
        LIMIT ?
    ''', conn, params=(limit,))
    conn.close()
    return df.to_dict('records')

init_db()
