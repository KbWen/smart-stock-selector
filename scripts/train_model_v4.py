import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import get_db_connection, get_all_tw_stocks
from core.ai_data_prep import prepare_v4_dataset

def train_sniper_v4():
    print("🚀 Starting V4.1 Sniper Model Training...")
    
    conn = get_db_connection()
    tickers = pd.read_sql("SELECT DISTINCT ticker FROM stock_history", conn)['ticker'].tolist()
    
    all_data = []
    
    print(f"📦 Loading and processing data for {len(tickers)} tickers...")
    for ticker in tickers:
        query = "SELECT * FROM stock_history WHERE ticker = ? ORDER BY date ASC"
        df = pd.read_sql(query, conn, params=(ticker,))
        
        if len(df) < 100: # Need enough history for indicators and look-ahead
            continue
            
        try:
            # prepare_v4_dataset handles indicators, scores, and labels
            dataset = prepare_v4_dataset(df)
            if not dataset.empty:
                all_data.append(dataset)
        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")
            
    if not all_data:
        print("❌ No data available for training.")
        return

    full_dataset = pd.concat(all_data).reset_index(drop=True)
    print(f"✅ Data processed. Total samples: {len(full_dataset)}")
    
    # Class Distribution
    win_count = full_dataset['label_win'].sum()
    loss_count = len(full_dataset) - win_count
    print(f"📊 Class Distribution: Win={win_count} ({win_count/len(full_dataset):.1%}), Loss={loss_count}")

    # Feature/Label split
    X = full_dataset.drop('label_win', axis=1)
    y = full_dataset['label_win']
    
    # Train/Test split (Simple for Lite version, but should use TimeSeriesSplit in prod)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🧠 Training Random Forest Classifier (Lite)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced' # Important for imbalanced classes
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    print("\n📈 Model Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n🌟 Top Features:")
    print(importance.head(10))

    # Save Model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"models/sniper_v4.1_{timestamp}.pkl"
    latest_name = "models/sniper_v4.1_latest.pkl"
    
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': model,
        'features': X.columns.tolist(),
        'version': f"v4.1.{timestamp}",
        'trained_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, model_name)
    joblib.dump(model_data, latest_name)
    
    print(f"\n✅ Model saved to {model_name}")
    print(f"🚀 Upgrade to V4.1 Core Complete!")

if __name__ == "__main__":
    train_sniper_v4()
