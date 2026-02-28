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
def get_alerts():
    """
    Returns active alerts for the user.
    Simulated using the database or trending logic.
    """
    return [
        {"id": 1, "symbol": "RELIANCE.NS", "message": "Crossed resistance at ₹2500", "time": "10:30 AM", "type": "positive"},
        {"id": 2, "symbol": "TCS.NS", "message": "High retail hype detected (Risk)", "time": "09:15 AM", "type": "negative"},
        {"id": 3, "symbol": "NIFTY50", "message": "Market entering volatile zone", "time": "09:00 AM", "type": "neutral"},
    ]

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

