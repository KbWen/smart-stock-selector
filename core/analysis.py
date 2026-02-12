import pandas as pd
import numpy as np

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
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

def calculate_rise_score(data: pd.DataFrame) -> dict:
    if data.empty or len(data) < 60:
        return {'total_score': 0, 'trend_score': 0, 'momentum_score': 0, 'volatility_score': 0, 'last_price': 0, 'change_percent': 0}

    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    
    # --- 1. Trend (40%) ---
    trend_score = 0
    # Price > SMA20 > SMA60 (Perfect Alignment)
    if last_row['close'] > last_row['sma_20'] > last_row['sma_60']:
        trend_score += 40
    elif last_row['close'] > last_row['sma_20']:
        trend_score += 20
        
    # SMA60 Styling Upwards
    sma60_slope = data['sma_60'].iloc[-1] - data['sma_60'].iloc[-5]
    if sma60_slope > 0:
        trend_score += 10
        
    trend_score = min(40, trend_score)

    # --- 2. Momentum (30%) ---
    momentum_score = 0
    # RSI Sweet Spot (50-70 is strong but not crazy overbought)
    rsi = last_row.get('rsi', 50)
    if 55 <= rsi <= 75:
        momentum_score += 15
    elif rsi > 80: # Overbought penalty
        momentum_score -= 10
        
    # MACD Golden Cross or Expanding
    macd = last_row.get('macd', 0)
    signal = last_row.get('macd_signal', 0)
    if macd > signal and macd > 0:
        momentum_score += 10
        
    # KD Golden Cross (K crosses above D)
    k = last_row.get('k', 50)
    d = last_row.get('d', 50)
    prev_k = prev_row.get('k', 50)
    prev_d = prev_row.get('d', 50)
    
    if prev_k < prev_d and k > d and k < 80: # Fresh Golden Cross
        momentum_score += 15
    elif k > d and k < 80: # Sustained stats
        momentum_score += 5
        
    momentum_score = min(30, max(0, momentum_score))

    # --- 3. Volatility / Setup (30%) ---
    volatility_score = 0
    
    # Volume Breakout
    vol_ma20 = data['volume'].rolling(window=20).mean().iloc[-1]
    if vol_ma20 > 0 and last_row['volume'] > vol_ma20 * 1.5:
        volatility_score += 15
    elif last_row['volume'] > vol_ma20:
        volatility_score += 5
        
    # Bollinger Squeeze (Compression before explosion)
    bb_width = last_row.get('bb_width', 1.0)
    if bb_width < 0.10: # Very tight squeeze
        volatility_score += 10
        
    # Bollinger Low Bounce ( %B < 0.2 but candle valid)
    bb_pct = last_row.get('bb_percent', 0.5)
    if 0 < bb_pct < 0.2 and last_row['close'] > last_row['open']: # Hammer at bottom
        volatility_score += 10

    volatility_score = min(30, volatility_score)

    total_score = trend_score + momentum_score + volatility_score
    
    change = 0
    if len(data) >= 2:
        change = ((last_row['close'] - prev_row['close']) / prev_row['close']) * 100

    return {
        'total_score': total_score,
        'trend_score': trend_score,
        'momentum_score': momentum_score,
        'volatility_score': volatility_score,
        'last_price': last_row['close'],
        'change_percent': change,
        'name': data.get('name', '') # Pass through if exists
    }
