# Whitepaper: Smart Stock Selector AI Sniper

## Executive Summary

The **Smart Stock Selector** is a specialized quantitative tool designed for the Taiwan Stock Market. It bridges the gap between traditional technical analysis and modern machine learning by focusing on asymmetric risk-reward profiles rather than simple price forecasts.

## 1. Technical Indicators (Rule-Based Engine)

The system calculates a multi-factor "Rise Score" (0-100) based on three pillars:

### A. Trend (40%)

Utilizes Moving Average ribbons (SMA20, SMA60). Full points are awarded when the price exhibits a perfect bullish alignment (Price > SMA20 > SMA60).

### B. Momentum (30%)

* **RSI (14)**: 採用標準的 **Wilder's Smoothing (EWMA)** 方法計算，捕捉更穩健的強弱訊號。
* **KD (Stochastics)**: Identifies fresh "Golden Crosses" where K crosses above D below the 80 level.
* **MACD**: Confirms directional strength via signal line divergence.

### C. Volatility & Setup (30%)

* **Bollinger Squeeze**: Detects periods of abnormally low volatility, often preceding explosive moves.
* **Volume Breakout**: Filters for price moves supported by heavy institutional-grade volume (>1.5x MA20).

## 2. The AI Sniper Model (Machine Learning V2)

Unlike "black box" models, our Sniper Model is trained with a specific exit strategy: **3:1 Risk/Reward**.

### Target Labeling

We label a "Win" ONLY if:

1. Price reaches **+15%** gain.
2. Condition 1 is met **BEFORE** the price reaches a **-5%** stop loss.
3. The outcome occurs within **20 trading days**.

### Advanced Feature Engineering (V2 Optimization)

* **Feature Normalization**: MACD 與 MACD Hist 皆經過價格標準化 (`Indicator / Price`)，消除股價絕對值對模型權重的影響。
* **Indicator Refinement**: 引入均線斜率 (`SMA Slope`) 與股價偏離度 (`Price Distance`)，捕捉趨勢的斜率與支撐強度。
* **Class Weighting**: 針對獲利樣本稀缺 (15.58%) 的特性，在訓練過程中施加類別權重，強制模型提升對「潛在贏家」的辨識召回率 (Recall)。

## 3. Data Integrity & "No Look-ahead" Policy

To prevent overfitting and survivorship bias:

* **Time-Series Splitting**: We use strict historical splitting for training. The model never "sees" the future during its training phase.
* **Normalized Features**: All price-based features are relative (percentages) to ensure the model generalizes across different stock price ranges (e.g., a $10 penny stock vs a $1000 blue chip).

## 4. Conclusion

The Smart Stock Selector does not aim to trade often; it aims to trade **well**. By filtering for high-probability setups where the math of the "Sniper Strategy" is in the user's favor, it provides a disciplined framework for successful swing trading.
