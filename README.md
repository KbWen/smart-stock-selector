# 智慧選股大師 🚀 AI 狙擊手版 (Smart Stock Selector)

這是一個專為台灣股市 (TWSE) 設計的技術分析儀表板與 AI 預測引擎。

## 🎯 核心理念：「狙擊手」策略

不同於單純預測漲跌的傳統模型，本系統採用 **3:1 損益比狙擊策略**。

* **獲利條件 (Win)**：在 20 個交易日內，股價先觸及 **+15%** (獲利點)，且過程中 **未曾** 觸及 **-5%** (停損點)。
* **AI 目標**：識別出「黃金勝率」的進場時機，即獲利機率遠高於停損機率的訊號。
* **核心價值**：極度挑剔的策略，旨在保護本金的同時追求高爆發性成長。

## 🏗️ 系統架構 (System Architecture)

```mermaid
graph TD
    A[資料來源: yfinance / twstock] --> B[核心資料層: core/data.py]
    B --> C[(本地資料庫: storage.db)]
    D[技術分析引擎: core/analysis.py]
    E[AI 預算引擎: core/ai/]
    C --> D
    C --> E
    D --> F[FastAPI 後端: backend/main.py]
    E --> F
    F --> G[現代化 Web UI]
    H[自動化腳本: daily_run.bat] --> B
    H --> E
    H --> I[重新計算分數: backend/recalculate.py]
    I --> C
```

## ⚙️ 運作流程 (Data Pipeline)

```mermaid
sequenceDiagram
    participant U as User / Batch
    participant S as Sync (Data)
    participant T as Train (AI)
    participant R as Recalc (Score)
    participant D as Dashboard
    
    U->>S: 下載歷史資料
    S->>T: 更新訓練集
    T->>T: 重新訓練 Ensemble V3
    T->>R: 更新預算結果
    R->>R: 計算所有指標與 Rise Score
    R->>D: 顯示最新排名與分析報告
```

## 🛠️ 技術棧

* **後端 (Backend)**: FastAPI (Python)
* **前端 (Frontend)**: 原生 JS, HTML5, CSS3 (現代化玻璃擬態 UI)
* **資料庫 (Database)**: SQLite (本地持久化存儲)
* **技術分析 (Analysis)**: Pandas, NumPy, 包含 KD, RSI (Wilder's), MACD (Normalized), 布林通道, ATR 等指標
* **人工智慧 (AI/ML)**: Ensemble V4 (結合 GradientBoosting, RandomForest 與 MLP 深度學習模型，並整合 Heuristic Rise Score 作為專家特徵)

## 🚀 快速上手 (Quick Start)

### 1. 安裝環境

確保您的電腦已安裝 Python，然後執行：

```bash
pip install -r requirements.txt
```

### 2. 資料同步 (資料庫初始化)

下載台股約 1000 檔股票的歷史資料到本地資料庫：

```bash
# 方法 A: 啟動後在網頁點擊 "Sync Data" 按鈕
# 方法 B: 直接執行伺服器程式碼 (伺服器啟動後才能同步)
python backend/main.py
```

*(同步過程約需 10-15 分鐘，請耐心等候)*

### 3. 每日自動更新 (推薦)

直接執行自動化腳本，即可完成「資料同步 -> AI 訓練 -> 分數重算」所有流程：

**Windows:**

```bash
./daily_run.bat
```

**Mac / Linux:**

```bash
chmod +x daily_run.sh
./daily_run.sh
```

### 4. 介面操作說明

* **🔄 Sync Data 按鈕**: 這是您的「數據心臟更新鍵」。點擊後，系統會即時從網路抓取最新的成交價，並利用現有的 AI 模型進行預測。適合在盤中或收盤後立即查看最新狀態。
* **🎯 狙擊手實戰觀察手冊**: 在儀表板中間新增了實戰指南，包含 **AI 共識檢核**、**技術面關聯性**與**大盤系統風險**三大量化觀察點，協助您分析訊號強度。

### 5. 分步手動執行 (選用)

如果您想手動控制流程：

1. **資料同步**: `python backend/main.py --sync`
2. **AI 訓練**: `python backend/train_ai.py`
3. **分數重算**: `python backend/recalculate.py`

### 5. 啟動儀表板

```bash
python backend/main.py
```

訪問網址：`http://localhost:8000/static/index.html`

## ✨ 亮點功能 (Feature Highlights)

| 功能 | 說明 |
| :--- | :--- |
| **Ensemble V4 AI** | 結合三種異質機器學習模型 (GB, RF, MLP)，並整合 **Rise Score** 技術指標分數作為訓練特徵，大幅提升預測穩定性。 |
| **Model Versioning** | 完整模型版本管理系統，自動追蹤訓練版本與同步時間，確保 UI 排名與策略回測結果 100% 一致。 |
| **Smart Sync 2.0** | 偵測模型更新或資料過期 (>6h) 自動觸發同步。並行同步技術 (`ThreadPoolExecutor`) 達成 10 倍速。 |
| **Stability Plus** | 啟用 SQLite WAL 模式與並行鎖定處理，確保在高強度同步與 API 請求同時發生時系統依然穩定不噴錯。 |
| **Backtest Lab** | 「時光機」功能。採用**大樣本搜尋 (300 檔)** 與**歷史機率排序**，真實模擬 30 天前的 AI Top 20 選股表現。 |
| **指標快取系統** | 將預計算的技術指標存儲於 `stock_indicators` 表，大幅提升掃描與回測速度。 |
| **AI 虛擬分析師** | 自動生成技術面解釋報告，解析 AI 預測背後的邏輯。 |
| **一鍵腳本** | `daily_run.bat` 讓每日資料更新與訓練變得極其簡單。 |

## 📁 專案結構

* `backend/`: FastAPI 接口與背景同步邏輯。
* `core/`: 核心邏輯，包含資料抓取 (`data.py`)、技術指標 (`analysis.py`)、與 AI 核心套件 (`ai/`)。
* `core/logger.py`: 全域日誌與警報系統。
* `frontend/`: 響應式儀表板前端檔案。
* `storage.db`: 本地 SQLite 資料庫 (自動生成)。

👉 **[查看完整資料字典 (Data Dictionary)](docs/DATA_DICTIONARY.md)**：包含資料庫定義、技術指標公式與 API 規格。

## 🔄 版本更新日誌 (Release Notes)

### v4.1.0 - The "Sniper Optimizer" Update (2026-02-14)

* **API 加速 (Score Persistence)**: 預先計算並儲存 V2 評分至資料庫，使狙擊手候選清單載入速度提升 90% 以上。
* **AI 回測版本切換**: 在時光機分頁新增模型版本選擇器，支援回溯並比較不同歷史模型的實戰表現。
* **台股核心優化**: 移除所有非台股相關的冗餘檔案與邏輯，全面調整為以 TWSE 為中心的資料處理流程。
* **代碼清理**: 移除過時的 `stock_data.db` 與 `index_legacy.html`，保持專案結構精簡效能極大化。

### v4.2.0 - The "Stability & Automation" Update (2026-02-14)

* **AI 核心模組化**: 將 `core/ai.py` 拆分為 `trainer.py` 與 `predictor.py`，優化訓練流程與預測快取。
* **全域日誌系統**: 引入 `core/logger.py` 支援 10MB 滾動記錄與警報勾點 (Alert Hooks)。
* **CI 自動化測試**: 整合 GitHub Actions，確保每次 Push/PR 均通過 `pytest` 認證。

## 📜 授權條款

MIT License
