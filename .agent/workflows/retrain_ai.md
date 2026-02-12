---
description: How to retrain the AI and update the dashboard
---

// turbo-all

1. Ensure the database contains historical data (run Sync in UI if empty).
2. Run the training script: `python backend/train_ai.py`.
3. Verify the model saved as `model_sniper.pkl`.
4. Recalculate all scores to reflect new model logic: `python backend/recalculate.py`.
5. Restart the server: `python backend/main.py`.
