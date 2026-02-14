import pandas as pd
import numpy as np
from core import config

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Standard Wilder's RSI using EWMA."""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Wilder's smoothing: alpha = 1 / window
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.DataFrame) -> tuple:
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_smas(data: pd.DataFrame) -> pd.DataFrame:
    data['sma_5'] = data['close'].rolling(window=5).mean()
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_60'] = data['close'].rolling(window=60).mean()
    return data

def calculate_emas(data: pd.DataFrame) -> pd.DataFrame:
    data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
    return data

def calculate_kd(data: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    low_min = data['low'].rolling(window=period).min()
    high_max = data['high'].rolling(window=period).max()
    
    rsv = (data['close'] - low_min) / (high_max - low_min) * 100
    
    # Calculate K and D (using EWMA equivalent)
    # K = 2/3 * PrevK + 1/3 * RSV
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    
    data['k'] = k
    data['d'] = d
    return data

def calculate_bollinger(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    
    data['bb_upper'] = sma + (std * 2)
    data['bb_lower'] = sma - (std * 2)
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / sma
    data['bb_percent'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    return data

def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(window=window).mean()

def add_rise_scores_to_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized version of the Rise Score logic.
    Adds 'trend_score', 'momentum_score', 'volatility_score', and 'total_score' to the DataFrame.
    """
    if data.empty or len(data) < 60:
        for col in ['trend_score', 'momentum_score', 'volatility_score', 'total_score']:
            data[col] = 0.0
        return data
        
    df = data.copy()
    
    w_trend = config.WEIGHT_TREND * 100
    w_mom = config.WEIGHT_MOMENTUM * 100
    w_vol = config.WEIGHT_VOLATILITY * 100

    # 1. Trend (Configurable weight)
    trend = pd.Series(0.0, index=df.index)
    # Price > SMA20 > SMA60
    cond1 = (df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_60'])
    trend = np.where(cond1, trend + w_trend, trend)
    # Price > SMA20 only
    cond2 = (df['close'] > df['sma_20']) & ~cond1
    trend = np.where(cond2, trend + (w_trend * 0.5), trend)
    
    # SMA60 Slope (捕捉趨勢力度) - 5 day pct change
    sma60_slope = df['sma_60'].pct_change(5) * 100
    trend = np.where(sma60_slope > 0, trend + (w_trend * 0.25), trend)
    df['trend_score'] = trend.clip(0, w_trend)

    # 2. Momentum (Configurable weight)
    momentum = pd.Series(0.0, index=df.index)
    rsi = df['rsi'].fillna(50)
    momentum = np.where((rsi >= 55) & (rsi <= 75), momentum + (w_mom * 0.5), momentum)
    momentum = np.where(rsi > 80, momentum - (w_mom * 0.3), momentum)
    
    # MACD Bullish
    macd_bull = (df['macd'] > df['macd_signal']) & (df['macd'] > 0)
    momentum = np.where(macd_bull, momentum + (w_mom * 0.33), momentum)
    
    # KD Cross (Simplified version for vectorization)
    kd_bull = (df['k'] > df['d']) & (df['k'] < 80)
    # Check for actual cross by comparing with shifted row
    kd_prev_bull = (df['k'].shift(1) < df['d'].shift(1))
    momentum = np.where(kd_bull & kd_prev_bull, momentum + (w_mom * 0.5), np.where(kd_bull, momentum + (w_mom * 0.16), momentum))
    
    df['momentum_score'] = momentum.clip(0, w_mom)

    # 3. Volatility / Setup (Configurable weight)
    volatility = pd.Series(0.0, index=df.index)
    
    # Volume explosion
    vol_ma20 = df['volume'].rolling(window=20).mean()
    volatility = np.where(df['volume'] > vol_ma20 * 1.5, volatility + (w_vol * 0.5), 
                          np.where(df['volume'] > vol_ma20, volatility + (w_vol * 0.16), volatility))
                          
    # Bollinger Squeeze
    volatility = np.where(df['bb_width'] < 0.10, volatility + (w_vol * 0.33), volatility)
    
    # Bollinger Support
    bb_support = (df['bb_percent'] > 0) & (df['bb_percent'] < 0.2) & (df['close'] > df['open'])
    volatility = np.where(bb_support, volatility + (w_vol * 0.33), volatility)
    
    df['volatility_score'] = volatility.clip(0, w_vol)
    
    df['total_score'] = df['trend_score'] + df['momentum_score'] + df['volatility_score']
    
    return df

def calculate_rise_score(data: pd.DataFrame) -> dict:
    if data.empty or len(data) < 60:
        return {'total_score': 0, 'trend_score': 0, 'momentum_score': 0, 'volatility_score': 0, 'last_price': 0, 'change_percent': 0}

    # Use vectorized version to ensure consistency
    df_scored = add_rise_scores_to_df(data)
    last_row = df_scored.iloc[-1]
    prev_row = df_scored.iloc[-2]
    
    change = 0
    if len(df_scored) >= 2:
        change = ((last_row['close'] - prev_row['close']) / prev_row['close']) * 100

    return {
        'total_score': float(last_row['total_score']),
        'trend_score': float(last_row['trend_score']),
        'momentum_score': float(last_row['momentum_score']),
        'volatility_score': float(last_row['volatility_score']),
        'last_price': float(last_row['close']),
        'change_percent': float(change),
        'name': data.get('name', '')
    }

def generate_analysis_report(last_row, prev_row, trend_score, momentum_score, volatility_score) -> dict:
    """Generates text-based analysis for the UI."""
    report = {
        'trend': 'Neutral',
        'momentum': 'Neutral',
        'setup': 'None'
    }
    
    # 1. Trend Analysis
    if last_row['close'] > last_row['sma_20'] > last_row['sma_60']:
        report['trend'] = "Strong Bullish Alignment (Price > SMA20 > SMA60)."
    elif last_row['close'] < last_row['sma_20'] < last_row['sma_60']:
        report['trend'] = "Strong Bearish Downtrend."
    elif last_row['close'] > last_row['sma_20']:
        report['trend'] = "Short-term Bullish, reclaiming SMA20."
    else:
        report['trend'] = "Consolidation / No clear trend."
        
    # 2. Momentum Analysis
    rsi = last_row.get('rsi', 50)
    if rsi > 75:
        report['momentum'] = f"Overbought (RSI {rsi:.1f}). Caution advised."
    elif rsi < 25:
        report['momentum'] = f"Oversold (RSI {rsi:.1f}). Potential bounce."
    elif 55 <= rsi <= 75:
        report['momentum'] = "Healthy Bullish Momentum."
    else:
        report['momentum'] = "Neutral Momentum."
        
    # 3. Setup/Volatility
    bb_width = last_row.get('bb_width', 1.0)
    
    # Safe get for vol_ma20 (features.py ensures it, but safe fallback)
    vol_ma20 = last_row.get('vol_ma20')
    current_vol = last_row['volume']
    
    if bb_width < 0.10:
        report['setup'] = "Volatility Squeeze! Big move imminent."
    elif vol_ma20 and current_vol > vol_ma20 * 2:
        report['setup'] = "High Volume Event Detected."
    else:
        report['setup'] = "Normal market activity."
        
    return report

