# Implementation Guide: StockTrust.ai

This document provides step-by-step instructions to set up and expand the StockTrust system.

## 1. Environment Setup

### Backend (Python 3.9+)
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (.env):
   ```
   DATABASE_URL=postgresql://user:password@localhost/stocks_db
   FIREBASE_KEY_PATH=path/to/firebase-key.json
   ```

### Database
1. Initialize PostgreSQL.
2. Run SQLAlchemy migrations or create tables:
   ```python
   from backend.database.db import engine
   from backend.database.models import Base
   Base.metadata.create_all(bind=engine)
   ```

## 2. Running the System

- **Start API**: `uvicorn backend.main:app --reload`
- **Run Daily Scan**: The system uses `APSchedule` (configured in `main.py`) to trigger `PreMarketScanner` at 08:45 IST daily.

## 3. 30-Day MVP Roadmap

### Phase 1: Data Integration (Days 1-7)
- Integrate `yfinance` and `nsepython` for live NSE feeds.
- Build the News Scraper using `BeautifulSoup` or `NewsAPI` for MoneyControl/EconomicTimes.
- Setup Redis for caching real-time prices.

### Phase 2: Core Engines (Days 8-15)
- Refine the Technical Engine with Support/Resistance detection.
- Implement Sentiment NLP using `textblob` or a pre-trained `Fine-tuned BERT` for financial sentiment.
- Finalize the `Trend Trust Score` algorithm weights.

### Phase 3: Alerts & Profile (Days 16-22)
- Integrate Firebase Cloud Messaging (FCM) for push notifications.
- Implement manual portfolio tracking with P/L calculations.
- Setup the 8:45 AM cron job for pre-market scans.

### Phase 4: Frontend & UX (Days 23-30)
- Build the React/Flutter mobile app using the provided glassmorphism design.
- Implement sentiment-over-time charts using Chart.js.
- Beta testing with retail investor group.

## 4. Market Psychology Scoring Matrix

| Signal | Source | Impact on Score |
|--------|--------|-----------------|
| Social Media Buzz Spike | Twitter / StockTwits | +15 (Hype) |
| News Sentiment (Positive) | MoneyControl / ET | +20 (Trust) |
| Google Trends Momentum | Search Volume | +10 (Awareness) |
| Large Block Deals | NSE Data | +15 (Smart Money) |
| Extreme RSI (>80) | Technical | -10 (Euphoria Risk) |

---

## 5. Security & Compliance
- **Financial Disclaimer**: "StockTrust is an advisory-only platform. Investments are subject to market risks. We do not execute trades."
- **Data Privacy**: All portfolio data is encrypted and accessible only to the user.
- **Read-Only**: The system never asks for broker login passwords.
