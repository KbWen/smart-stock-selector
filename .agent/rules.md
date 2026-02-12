# Antigravity Rules for Smart Stock Selector

1. **Code Consistency**: Always maintain the separation between `core/` (logic), `backend/` (API), and `frontend/` (UI).
2. **Scoring Logic**: Do not modify `core/analysis.py` scoring weights without explicit user approval.
3. **AI Strategy**: The current "Sniper Strategy" (+15%/-5% in 20 days) is the baseline. Any changes to `PRED_DAYS`, `TARGET_GAIN`, or `STOP_LOSS` require retraining.
4. **Data Persistence**: Use `core/data.py` for all DB interactions to ensure thread safety (SQLite sync).
5. **Environment**: Always use `requirements.txt` for dependencies. Avoid adding massive packages (like heavy deep learning frameworks) unless requested, preferring `scikit-learn` for efficiency.
