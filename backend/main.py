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
    holdings = db.query(models.Portfolio).filter(models.Portfolio.user_id == 1).all()
    analyzer = StockAnalyzer()
    
    results = []
    total_value = 0
    total_investment = 0
    
    
    for h in holdings:
        # Analyze the stock to get real-time price and advisory
        print(f"DEBUG_PORTFOLIO: Analyzing {h.symbol}...")
        analysis = analyzer.analyze_stock(h.symbol)
        
        if analysis:
            current_price = analysis['price']
            advisory = analysis['advisory']
            verdict = analysis['verdict']
            trust_score = analysis['trust_score']
            reasoning = analysis['reasoning']
            futures_data = analysis.get('futures_data', {})
        else:
            # Fallback if analysis fails (e.g. no internet)
            current_price = h.avg_price
            advisory = {
                "entry": "N/A", "target": "N/A", "stop_loss": "N/A", "analyst_rating": "Neutral"
            }
            verdict = "HOLD"
            trust_score = 50
            reasoning = "Data unavailable"
            futures_data = {}

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
            "advisory": advisory,
            "reasoning": reasoning,
            "futures_data": futures_data
        })
        
    # Simple Sector Map for Demo Risk Analysis
    sector_map = {
        "RELIANCE": "Energy", "ONGC": "Energy", "NTPC": "Energy",
        "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
        "HDFCBANK": "Finance", "ICICIBANK": "Finance", "SBIN": "Finance",
        "TATAMOTORS": "Auto", "MARUTI": "Auto", "TMPV": "Auto",
        "SILVERBEES": "Commodities", "GOLDBEES": "Commodities"
    }
    
    sector_alloc = {}
    for r in results:
        # Extract base symbol from "RELIANCE.NS" -> "RELIANCE"
        base_sym = r['symbol'].split('.')[0]
        sec = sector_map.get(base_sym, "Others")
        val = r['current_price'] * r['quantity']
        sector_alloc[sec] = sector_alloc.get(sec, 0) + val
        
    # Convert to %
    risk_analysis = []
    if total_value > 0:
        for sec, val in sector_alloc.items():
            risk_analysis.append({
                "sector": sec,
                "value": round(val, 2),
                "percent": round((val/total_value)*100, 1)
            })

    return {
        "holdings": results,
        "summary": {
            "total_value": round(total_value, 2),
            "total_investment": round(total_investment, 2),
            "total_pl": round(total_value - total_investment, 2),
            "total_pl_percent": round(((total_value - total_investment) / total_investment * 100), 2) if total_investment > 0 else 0,
            "risk_analysis": risk_analysis
        }
    }

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

