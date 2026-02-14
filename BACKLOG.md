# Project Backlog - Sniper V4.1+

This document tracks upcoming tasks and optimizations that were not included in the current standardization wave.

## High Priority

- [ ] **AI Core Refactoring**: Split `core/ai.py` into `core/ai/trainer.py` and `core/ai/predictor.py`.
- [ ] **CI Integration**: Set up GitHub Actions for automated `pytest` on every PR.
- [ ] **Enhanced Error Logging**: Implement a more robust logging system with file rotation and Slack/Discord alerts for critical sync failures.

## Medium Priority

- [ ] **User Authentication**: Add a basic login layer to the React frontend.
- [ ] **Multi-Market Support**: Extend `core/data.py` to support US stocks (optional, low priority for now).
- [ ] **Real-time Search Filter**: Improve the frontend search to filter the local list more efficiently.

## Low Priority / Experimental

- [ ] **LLM Market Commentary**: Use an LLM to generate automated market sentiment analysis based on news feeds.
- [ ] **Dockerization**: Create a `Dockerfile` and `docker-compose.yml` for easy deployment.
