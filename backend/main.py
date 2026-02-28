from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
import uvicorn
import asyncio
import datetime
import threading
import time
import os
import sys
import platform

from backend.database import models, db
from backend.jobs.scanner import PreMarketScanner
from backend.jobs.notifier import WhatsAppNotifier

import yfinance as yf
from backend.engines.analyzer import StockAnalyzer
from backend.engines.portfolio_intelligence import PortfolioIntelligenceSystem
from backend.engines.performance_review import PerformanceReviewEngine

# Create database tables automatically
models.Base.metadata.create_all(bind=db.engine)

# --- Seed Default Portfolio (ensures data persists on cloud restarts) ---
def _seed_portfolio():
    """Seeds portfolio if empty (handles Railway's ephemeral SQLite)."""
    session = db.SessionLocal()
    try:
        # Check if user exists
        user = session.query(models.User).filter_by(id=1).first()
        if not user:
            user = models.User(id=1, email="praka@puniai.com", password_hash="seeded")
            session.add(user)
            session.commit()
        
        # Check if portfolio is empty
        count = session.query(models.Portfolio).count()
        if count == 0:
            holdings = [
                models.Portfolio(user_id=1, symbol="IRFC.NS", quantity=850, avg_price=24.14),
                models.Portfolio(user_id=1, symbol="SBIN.NS", quantity=15, avg_price=195.2),
                models.Portfolio(user_id=1, symbol="TMCV.NS", quantity=75, avg_price=27.07),
                models.Portfolio(user_id=1, symbol="TMPV.NS", quantity=75, avg_price=59.83),
                models.Portfolio(user_id=1, symbol="SILVERBEES.NS", quantity=97, avg_price=235.97),
                models.Portfolio(user_id=1, symbol="YESBANK.NS", quantity=450, avg_price=21.15),
            ]
            session.add_all(holdings)
            session.commit()
            print("[Seed] Portfolio seeded with 6 holdings")
        else:
            print(f"[Seed] Portfolio already has {count} holdings — skipping")
    except Exception as e:
        print(f"[Seed] Error: {e}")
        session.rollback()
    finally:
        session.close()

_seed_portfolio()


# --- Background Scheduler (Daemon Thread) ---
def _scheduler_thread():
    """Runs in a daemon thread. Automatically killed when the main process exits."""
    print("[Scheduler] Started (daemon thread). Waiting for 08:45 AM...")
    
    while True:
        try:
            now = datetime.datetime.now()
            if now.hour == 8 and now.minute == 45:
                # Run the notifier synchronously using asyncio.run()
                notifier = WhatsAppNotifier()
                asyncio.run(notifier.generate_and_send_daily_alert())
                time.sleep(61)  # Don't send twice in the same minute
            else:
                time.sleep(10)
        except Exception as e:
            print(f"[Scheduler] Error: {e}")
            time.sleep(30)

# Start the daemon thread
_sched_thread = threading.Thread(target=_scheduler_thread, daemon=True)
_sched_thread.start()


# --- Windows Hard Exit Workaround (Bypassing Uvicorn) ---
import signal
import sys

def _force_hard_exit():
    """Hard kill after 2.0s to ensure DB had time to flush but process doesn't hang."""
    time.sleep(2.0)
    print("\n[System] Forcing process exit to prevent Windows hang...")
    os._exit(0)

def custom_sigint_handler(signum, frame):
    print("\n[System] Caught Ctrl+C. Bypassing Uvicorn's signal handler for a clean Windows exit...")
    try:
        db.engine.dispose()
        print("[System] Database connections closed cleanly.")
    except Exception as e:
        print(f"[System] DB closure error: {e}")
    
    threading.Thread(target=_force_hard_exit, daemon=True).start()
    sys.exit(0)

# Register the signal handler only on Windows (Linux handles it fine)
if platform.system() == 'Windows':
    import signal
    signal.signal(signal.SIGINT, custom_sigint_handler)

app = FastAPI(title="Puni.ai Stock Intelligence")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    database = db.SessionLocal()
    try:
        yield database
    finally:
        database.close()


# --- Pydantic Models ---

class HoldingCreate(BaseModel):
    symbol: str
    quantity: int
    avg_price: float

class IntelligenceRequest(BaseModel):
    capital: float
    risk_profile: str
    horizon: str


# --- API Routes ---

@app.get("/trending")
async def get_trending_stocks():
    scanner = PreMarketScanner()
    top_5 = await scanner.run_daily_scan()
    return top_5

@app.post("/portfolio/add")
def add_holding(holding: HoldingCreate, db: Session = Depends(get_db)):
    # Ensure symbol is uppercase for consistency
    symbol = holding.symbol.upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"
        
    new_holding = models.Portfolio(
        user_id=1, # Mocked user for now
        symbol=symbol,
        quantity=holding.quantity,
        avg_price=holding.avg_price
    )
    db.add(new_holding)
    db.commit()
    db.refresh(new_holding)
    return {"status": "success", "data": new_holding}

@app.put("/portfolio/update/{holding_id}")
def update_holding(holding_id: int, holding: HoldingCreate, db: Session = Depends(get_db)):
    db_holding = db.query(models.Portfolio).filter(models.Portfolio.id == holding_id).first()
    if not db_holding:
        raise HTTPException(status_code=404, detail="Holding not found")
    
    # Update fields
    db_holding.quantity = holding.quantity
    db_holding.avg_price = holding.avg_price
    # Symbol is generally not edited, best to delete and re-add if symbol is wrong
    
    db.commit()
    return {"status": "success", "message": f"Holding {holding_id} updated"}

@app.get("/analyze/{symbol}")
def analyze_single_stock(symbol: str):
    symbol = symbol.upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    
    analyzer = StockAnalyzer()
    analysis = analyzer.analyze_stock(symbol)
    
    if not analysis:
        # Fallback if analysis fails (e.g. invalid symbol)
        return {"status": "error", "message": "Could not analyze stock. Check symbol."}
    
    return {"status": "success", "data": analysis}

@app.delete("/portfolio/delete/{holding_id}")
def delete_holding(holding_id: int, db: Session = Depends(get_db)):
    holding = db.query(models.Portfolio).filter(models.Portfolio.id == holding_id).first()
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")
    
    db.delete(holding)
    db.commit()
    return {"status": "success", "message": f"Holding {holding_id} deleted"}

@app.post("/portfolio/intelligence")
def generate_intelligence(req: IntelligenceRequest, db: Session = Depends(get_db)):
    # 1. Fetch Current Portfolio from DB
    holdings = db.query(models.Portfolio).filter(models.Portfolio.user_id == 1).all()
    
    portfolio_data = []
    for h in holdings:
        portfolio_data.append({
            "ticker": h.symbol,
            "avg_buy_price": h.avg_price,
            "quantity": h.quantity,
            "holding_period_months": 6 # Mocked as db doesn't have date yet
        })
        
    # 2. Prepare Input Payload
    sys_input = {
        "portfolio": portfolio_data,
        "capital_available": req.capital,
        "risk_profile": req.risk_profile,
        "investment_horizon": req.horizon,
        "focus": "Wealth growth" # Default
    }
    
    # 3. Run Analysis
    intel = PortfolioIntelligenceSystem()
    
    my_port_view = intel.analyze_portfolio(sys_input)
    new_buys_view = intel.generate_top_buys(sys_input)
    
    return {
        "status": "success",
        "data": {
            "portfolio_html": my_port_view,
            "new_buys_html": new_buys_view
        }
    }

@app.get("/review")
def get_performance_review():
    reviewer = PerformanceReviewEngine()
    html_report = reviewer.generate_review()
    return {"status": "success", "html": html_report}

@app.get("/portfolio")
def get_portfolio(db: Session = Depends(get_db)):
    # Check cache first
    from backend.jobs.scanner import _get_cached, _set_cached
    cached = _get_cached("portfolio_data")
    if cached:
        print("[Portfolio] Returning cached data")
        return cached
    
    holdings = db.query(models.Portfolio).filter(models.Portfolio.user_id == 1).all()
    
    results = []
    total_value = 0
    total_investment = 0
    
    # Batch fetch ALL portfolio prices in ONE call
    symbols = [h.symbol for h in holdings]
    prices = {}
    try:
        if symbols:
            data = yf.download(symbols, period="5d", group_by="ticker", progress=False, threads=False)
            for sym in symbols:
                try:
                    if len(symbols) > 1:
                        close_series = data[sym]['Close'].dropna()
                    else:
                        close_series = data['Close'].dropna()
                    if not close_series.empty:
                        prices[sym] = float(close_series.iloc[-1])
                except Exception:
                    pass
            print(f"[Portfolio] Batch fetched prices for {len(prices)}/{len(symbols)} stocks")
    except Exception as e:
        print(f"[Portfolio] Batch download error: {e}")
    
    for h in holdings:
        current_price = prices.get(h.symbol, h.avg_price)  # fallback to avg_price
        verdict = "HOLD"
        trust_score = 50
        
        if h.symbol in prices:
            change_pct = ((current_price - h.avg_price) / h.avg_price) * 100
            if change_pct > 10:
                verdict = "BUY"
                trust_score = 70
            elif change_pct > 0:
                verdict = "HOLD"
                trust_score = 55
            else:
                verdict = "HOLD"
                trust_score = 40
        
        current_price = float(current_price)
        investment = h.quantity * h.avg_price
        current_value = h.quantity * current_price
        pl = current_value - investment
        pl_percent = (pl / investment * 100) if investment > 0 else 0
        
        total_value += current_value
        total_investment += investment
        
        results.append({
            "id": h.id,
            "symbol": h.symbol,
            "quantity": h.quantity,
            "avg_price": h.avg_price,
            "current_price": round(current_price, 2),
            "pl": round(pl, 2),
            "pl_percent": round(pl_percent, 2),
            "verdict": verdict,
            "trust_score": trust_score,
            "advisory": {
                "entry": f"₹{h.avg_price}",
                "target": f"{round(current_price * 1.05, 1)}",
                "stop_loss": f"{round(current_price * 0.95, 1)}",
                "analyst_rating": f"{verdict} (Conf: {trust_score}%)"
            },
            "reasoning": f"Bought at ₹{h.avg_price}, now ₹{round(current_price, 2)}",
            "futures_data": {}
        })
        
    # Sector Map for Risk Analysis
    sector_map = {
        "RELIANCE": "Energy", "ONGC": "Energy", "NTPC": "Energy",
        "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
        "HDFCBANK": "Finance", "ICICIBANK": "Finance", "SBIN": "Finance",
        "IRFC": "Finance", "YESBANK": "Finance",
        "TATAMOTORS": "Auto", "MARUTI": "Auto", "TMPV": "Auto", "TMCV": "Auto",
        "SILVERBEES": "Commodities", "GOLDBEES": "Commodities"
    }
    
    sector_alloc = {}
    for r in results:
        base_sym = r['symbol'].split('.')[0]
        sec = sector_map.get(base_sym, "Others")
        val = r['current_price'] * r['quantity']
        sector_alloc[sec] = sector_alloc.get(sec, 0) + val
        
    risk_analysis = []
    if total_value > 0:
        for sec, val in sector_alloc.items():
            risk_analysis.append({
                "sector": sec,
                "value": round(val, 2),
                "percent": round((val/total_value)*100, 1)
            })

    result = {
        "holdings": results,
        "summary": {
            "total_value": round(total_value, 2),
            "total_investment": round(total_investment, 2),
            "total_pl": round(total_value - total_investment, 2),
            "total_pl_percent": round(((total_value - total_investment) / total_investment * 100), 2) if total_investment > 0 else 0,
            "risk_analysis": risk_analysis
        }
    }
    
    # Cache for 5 minutes
    _set_cached("portfolio_data", result)
    return result

@app.get("/alerts")
def get_alerts(db: Session = Depends(get_db)):
    """
    Dynamic alerts based on portfolio, scanner data, and market intelligence.
    """
    from backend.jobs.scanner import _get_cached
    import datetime
    
    alerts = []
    alert_id = 1
    now = datetime.datetime.now()
    time_str = now.strftime("%I:%M %p")
    
    # ── 1. Portfolio-Based Alerts ──
    portfolio_data = _get_cached("portfolio_data")
    if portfolio_data and 'holdings' in portfolio_data:
        for h in portfolio_data['holdings']:
            sym = h['symbol'].split('.')[0]
            pl_pct = h.get('pl_percent', 0)
            
            # Big winner alert
            if pl_pct > 50:
                alerts.append({
                    "id": alert_id, "symbol": h['symbol'],
                    "message": f"{sym} is up {pl_pct:.0f}% from your buy price! Consider booking partial profits.",
                    "time": time_str, "type": "positive", "category": "Portfolio"
                })
                alert_id += 1
            
            # Loss warning
            elif pl_pct < -15:
                alerts.append({
                    "id": alert_id, "symbol": h['symbol'],
                    "message": f"{sym} is down {abs(pl_pct):.0f}%. Review your stop-loss levels.",
                    "time": time_str, "type": "negative", "category": "Risk"
                })
                alert_id += 1
            
            # Near breakeven after being negative
            elif -2 < pl_pct < 2 and h.get('avg_price', 0) != h.get('current_price', 0):
                alerts.append({
                    "id": alert_id, "symbol": h['symbol'],
                    "message": f"{sym} is near your buy price (Rs.{h.get('current_price', 0)}). Watch for breakout.",
                    "time": time_str, "type": "neutral", "category": "Watch"
                })
                alert_id += 1
        
        # Total portfolio alert
        summary = portfolio_data.get('summary', {})
        total_pl = summary.get('total_pl_percent', 0)
        if total_pl > 10:
            alerts.insert(0, {
                "id": alert_id, "symbol": "PORTFOLIO",
                "message": f"Your portfolio is up {total_pl:.1f}% overall! Great performance.",
                "time": time_str, "type": "positive", "category": "Portfolio"
            })
            alert_id += 1
    
    # ── 2. Scanner/Trending Alerts ──
    trending = _get_cached("trending_scan")
    if trending:
        # High confidence BUY alerts
        for stock in trending:
            if stock.get('verdict') in ('BUY', 'STRONG BUY') and stock.get('trust_score', 0) >= 65:
                sym = stock['symbol'].split('.')[0]
                alerts.append({
                    "id": alert_id, "symbol": stock['symbol'],
                    "message": f"{sym} shows {stock['verdict']} signal with {stock['trust_score']}% confidence. Entry: Rs.{stock['price']}",
                    "time": time_str, "type": "positive", "category": "Signal"
                })
                alert_id += 1
            
            # Risk/SELL alerts
            elif stock.get('verdict') in ('SELL', 'STRONG SELL', 'AVOID'):
                sym = stock['symbol'].split('.')[0]
                alerts.append({
                    "id": alert_id, "symbol": stock['symbol'],
                    "message": f"{sym} shows {stock['verdict']} signal. Avoid new positions.",
                    "time": time_str, "type": "negative", "category": "Risk"
                })
                alert_id += 1
    
    # ── 3. Market Intelligence Alerts ──
    hour = now.hour
    if 9 <= hour <= 15:
        alerts.append({
            "id": alert_id, "symbol": "MARKET",
            "message": "Indian market is currently open. Live data is being tracked.",
            "time": time_str, "type": "neutral", "category": "Market"
        })
        alert_id += 1
    elif hour < 9:
        alerts.append({
            "id": alert_id, "symbol": "MARKET",
            "message": "Pre-market session. Check global cues before market opens at 9:15 AM.",
            "time": time_str, "type": "neutral", "category": "Market"
        })
        alert_id += 1
    else:
        alerts.append({
            "id": alert_id, "symbol": "MARKET",
            "message": "Market closed for today. Prices reflect last closing values.",
            "time": time_str, "type": "neutral", "category": "Market"
        })
        alert_id += 1
    
    # ── 4. Daily Tips ──
    day_tips = [
        "Diversify across sectors to reduce risk. Never put all eggs in one basket.",
        "Set stop-losses for every trade. Protect capital before chasing profits.",
        "Review your portfolio weekly. Cut losers early, let winners run.",
        "Avoid trading on emotions. Stick to your strategy and analysis.",
        "Blue chips for stability, small caps for growth. Balance both.",
        "Pre-market news can move stocks 3-5%. Check before placing orders.",
        "F&O positions require extra caution. Monitor OI changes daily."
    ]
    tip_index = now.day % len(day_tips)
    alerts.append({
        "id": alert_id, "symbol": "TIP",
        "message": day_tips[tip_index],
        "time": time_str, "type": "neutral", "category": "Tip"
    })
    
    # If no alerts generated (no data yet), add default ones
    if len(alerts) <= 2:
        alerts.extend([
            {"id": 100, "symbol": "SYSTEM", "message": "Dashboard is loading stock data. Alerts will appear shortly.", "time": time_str, "type": "neutral", "category": "System"},
            {"id": 101, "symbol": "SBIN.NS", "message": "Banking sector showing strength. Watch SBIN, HDFCBANK for breakouts.", "time": time_str, "type": "positive", "category": "Signal"},
            {"id": 102, "symbol": "IRFC.NS", "message": "Railway stocks in focus. Budget allocation may boost IRFC.", "time": time_str, "type": "positive", "category": "News"},
        ])
    
    return alerts

@app.get("/profile")
def get_profile():
    return {
        "name": "Puni",
        "email": "investor@puni.ai",
        "plan": "Premium (Advisory)",
        "joined": "Jan 2026",
        "risk_profile": "Balanced"
    }


@app.get("/share/whatsapp")
async def share_whatsapp():
    notifier = WhatsAppNotifier()
    # reuse the logic to populate top picks
    top_picks = await notifier.scanner.run_daily_scan()
    
    if not top_picks:
        return {"text": "No strong buy signals found today."}

    message = "🚀 *Puni.ai Morning Scan* 🚀\n"
    message += f"Date: {datetime.date.today()}\n\n"
    
    for stock in top_picks:
        icon = "💎" if stock['category'] == "Blue Chip" else "🪙"
        entry = stock['advisory'].get('entry', 'N/A')
        target = stock['advisory'].get('target', 'N/A')
        
        message += f"{icon} *{stock['symbol']}* ({stock['category']})\n"
        message += f"   • Price: ₹{round(float(stock['price']), 2)}\n"
        message += f"   • Entry: {entry}\n"
        message += f"   • Target: ₹{target}\n\n"
        
    message += "⚠️ _System Generated_"
    return {"text": message}


# Mount frontend directory to serve HTML/CSS/JS
# NOTE: This MUST be last — StaticFiles will catch all unmatched routes
# Use path relative to this file so it works on any OS / cloud
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)

