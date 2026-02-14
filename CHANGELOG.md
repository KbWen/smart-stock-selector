# Changelog

All notable changes to this project will be documented in this file.

## [v4.1.0] - 2026-02-14 "The Sniper Optimizer"

### Added

* **AI Model Versioning**: Added `Model Version Switcher` in Backtest UI to compare historical models.
* **Persistent Scoring**: V2 scores are now pre-calculated and stored in DB for <100ms API response.
* **Refined Backtest**: "Time Machine" simulation now supports selecting specific model versions.

### Changed

* **Data Pipeline**: Fully optimized for Taiwan Stock Market (TWSE).
* **Documentation**: Major overhaul of system docs (Architecture, Contributing, etc.).
* **Cleanup**: Removed legacy `stock_data.db` and old HTML files.

## [v2.1.0] - 2026-02-13 "The Market Pulse"

### Added

* **Global Search**: API endpoint `/api/search` for universal stock queries.
* **Market Pulse Chart**: Visualized 30-day market temperature and AI sentiment.
* **Interactive Backtest**: Clickable rows in backtest results to view details.

### Fixed

* **Search Scope**: Fixed issue where search was limited to local ranking list.
* **UI Bugs**: Fixed `handleSearch` scope and state dependency regressions.

## [v2.0.0] - 2026-02-12 "The React Pivot"

### Added

* **Modern Frontend**: Replaced vanilla JS with React + Vite + Tailwind.
* **Glassmorphism UI**: New dark mode design with translucent components.
* **New Components**: StockList, SniperCard, ChartContainer.

## [v1.0.0] - Initial Release

* Basic Technical Analysis Engine.
* Random Forest AI Model.
* Simple HTML Dashboard.
