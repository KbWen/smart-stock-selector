# Smart Stock Selector 🚀 AI Sniper Edition

A quantitative stock analysis dashboard and AI prediction engine for the Taiwan Stock Exchange (TWSE).

## 🎯 Core Philosophy: The "Sniper" Strategy

Unlike traditional models that predict simple price direction, this system employs a **3:1 Risk/Reward Sniper Strategy**.

* **Win Condition**: Price gains **+15%** within 20 trading days **BEFORE** hitting a **-5%** stop loss.
* **Target**: The AI identifies "Golden Setups" where the probability of reaching the profit target is significantly higher than hitting the stop loss.
* **Result**: A highly selective, capital-preserving approach to swing trading.

## 🛠️ Tech Stack

* **Backend**: FastAPI (Python)
* **Frontend**: Vanilla JS, HTML5, CSS3 (Modern Glassmorphism UI)
* **Database**: SQLite (Persistent local storage)
* **Analysis**: Pandas, NumPy, TA indicators (KD, RSI, MACD, Bollinger Bands, ATR)
* **AI/ML**: GradientBoostingClassifier (Scikit-Learn)

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Sync

Initialize the local database by downloading historical data for all TWSE stocks:

```bash
# Via Web UI: Click "Sync Data" 
# OR via Code:
python backend/main.py
```

*(Syncing ~1000 stocks takes approx. 10-15 minutes)*

### 3. AI Training

Train the Sniper model on your local data:

```bash
python backend/train_ai.py
```

### 4. Recalculate Scores

Update the dashboard with latest technical scores and AI probabilities:

```bash
python backend/recalculate.py
```

### 5. Launch Dashboard

```bash
python backend/main.py
```

Visit: `http://localhost:8000/static/index.html`

## 📊 Features

* 🔥 **Technical Picks**: Rule-based screening using KD, RSI, and Bollinger Squeeze.
* 🤖 **AI Ranking**: Top 50 stocks sorted by Sniper Win Probability.
* 🔍 **Smart Search**: Search by ticker or name with real-time detail modal.
* 🛡️ **Risk Management**: Automatic calculation of Target and Stop prices.

## 📁 Project Structure

* `backend/`: FastAPI endpoints and background tasks.
* `core/`: Core logic for data fetching (`data.py`), technical analysis (`analysis.py`), and AI (`ai.py`).
* `frontend/`: Responsive dashboard files.
* `storage.db`: Local SQLite database (Auto-generated).

## 📜 License

MIT
