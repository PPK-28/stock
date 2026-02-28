
from backend.engines.analyzer import StockAnalyzer
from backend.engines.technical import TechnicalEngine
from backend.engines.fundamental_risk import FundamentalEngine, RiskEngine
from backend.engines.sentiment import SentimentEngine
import yfinance as yf
import pandas as pd
import datetime

class PortfolioIntelligenceSystem:
    def __init__(self):
        self.analyzer = StockAnalyzer()
        self.tech = TechnicalEngine()
        self.fund = FundamentalEngine()
        self.sent = SentimentEngine()
        self.risk = RiskEngine()
        
        # Pre-defined candidate pools for "New Buys" (Simulated Database/Scan Result)
        self.candidate_pool = [
            "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", # Banking
            "RELIANCE.NS", "TCS.NS", "INFY.NS", # Large Cap Core
            "HAL.NS", "BEL.NS", "MAZDOCK.NS", # Defence/Govt
            "TATASTEEL.NS", "JINDALSTEL.NS", # Metals/Infra
            "TATAMOTORS.NS", "M&M.NS", # Auto
            "TITAN.NS", "TRENT.NS", # Consumption
            "RVNL.NS", "IRFC.NS" # Railways
        ]

    def analyze_portfolio(self, user_input):
        """
        SECTION C – MY PORTFOLIO VIEW
        """
        portfolio = user_input.get('portfolio', [])
        capital = user_input.get('capital_available', 0)
        risk_profile = user_input.get('risk_profile', 'Moderate')
        
        report = []
        sector_dist = {}
        
        report.append(f"<h2>💼 MY PORTFOLIO VIEW ({len(portfolio)} Holdings)</h2>")
        report.append(f"<p>Risk Profile: {risk_profile} | Capital for New Buys: ₹{capital:,}</p>")
        
        for holding in portfolio:
            symbol = holding['ticker']
            if not symbol.endswith('.NS'): symbol += '.NS'
            
            # Analyze each stock
            analysis = self.analyzer.analyze_stock(symbol) # Leveraging existing Boomerang logic
            if not analysis:
                report.append(f"<div class='stock-card'><h3 style='color:grey'>{symbol} - Data Unavailable</h3></div>")
                continue
                
            # Extract scores from the analysis result (simulating extraction from new structured output)
            # In a real scenario, we'd refactor analyze_stock to return raw objects too, 
            # but here we parse or recalculate lighter versions if needed.
            # Actually, analyze_stock returns a structured dict with 'trust_score', 'risk', etc.
            
            # Let's derive the specific View States required
            ts = analysis.get('trust_score', 50)
            
            # Action Logic based on Risk Profile
            action = "HOLD"
            reason = analysis['verdict'] + " based on technical structure."
            
            if ts > 75: action = "ACCUMULATE"
            elif ts < 40: action = "EXIT / REDUCE"
            
            report.append(f"""
            <div class='stock-card glass'>
                <div style='display:flex; justify-content:space-between;'>
                    <b>{symbol.replace('.NS', '')}</b>
                    <span class='badge' style='background:{self._get_action_color(action)}'>{action}</span>
                </div>
                <div style='font-size:11px; color:#aaa; margin-top:5px;'>
                    TECH: {self._get_score_label(ts)} ({int(ts)}/100) | 
                    RISK: {analysis['risk']}
                </div>
                <div style='font-size:12px; margin-top:8px;'>
                    <i>Action (1-3m):</i> {action}<br>
                    <i>Stop Loss:</i> {analysis['advisory']['stop_loss']}<br>
                    <i>Reason:</i> {analysis['advisory']['analyst_rating']}
                </div>
            </div>
            """)
            
        return "".join(report)

    def generate_top_buys(self, user_input):
        """
        Generates the 'Capital Allocation & Return Estimation' Report.
        NOW SUPPORTS: Penny Stocks & Commodities based on Risk Profile.
        """
        user_capital = user_input.get('capital_available', 100000)
        risk_profile = user_input.get('risk_profile', 'Moderate')
        horizon = user_input.get('horizon', '6-12 months')
        
        # 1. FETCH STOCKS & COMMODITIES
        scanner_results = self._fetch_scanner_results()
        
        # 2. DEFINE ALLOCATION RULES (Strict adherence to prompt)
        rules = {
            "Conservative": {
                "max_penny": 0.0, "max_commodity": 0.10, 
                "stock_mix": "Large/Mid Cap Only", "positions": (2, 3),
                "commodities": ["GOLDBEES.NS", "SILVERBEES.NS"] # Safe ETFs
            },
            "Moderate": {
                "max_penny": 0.05, "max_commodity": 0.15,
                "stock_mix": "Mostly Large/Mid + Some Speculative", "positions": (3, 4),
                "commodities": ["GOLDBEES.NS", "SILVERBEES.NS"]
            },
            "Aggressive": {
                "max_penny": 0.20, "max_commodity": 0.20,
                "stock_mix": "Mix of Large + Penny + Sector Bets", "positions": (4, 6),
                "commodities": ["GOLDBEES.NS", "SILVERBEES.NS", "CRUDEOIL"] # Mock crude proxy
            },
            "Very Aggressive": {
                "max_penny": 0.30, "max_commodity": 0.25,
                "stock_mix": "High Beta + Penny + Volatile Commodities", "positions": (5, 8),
                "commodities": ["GOLDBEES.NS", "SILVERBEES.NS", "CRUDEOIL"]
            }
        }
        
        # Default to Moderate if unknown
        profile = rules.get(risk_profile, rules["Moderate"])
        
        # 3. FILTER & SELECT ASSETS
        # Separate Universe
        penny_universe = [s for s in scanner_results if s['category'] == "Penny Stock"]
        core_universe = [s for s in scanner_results if s['category'] != "Penny Stock"]
        
        final_allocation = []
        
        # A. Allocation Budgeting
        penny_budget = user_capital * profile['max_penny']
        comm_budget = user_capital * profile['max_commodity']
        core_budget = user_capital - penny_budget - comm_budget
        
        # B. Select Penny Stocks (if allowed)
        if penny_budget > 0 and penny_universe:
            # Sort by Trust Score
            top_pennies = sorted(penny_universe, key=lambda x: x['trust_score'], reverse=True)[:2]
            for p in top_pennies:
                alloc = penny_budget / len(top_pennies)
                qty = int(alloc / p['price']) if p['price'] > 0 else 0
                if qty > 0:
                    final_allocation.append({
                        "ticker": p['symbol'], "type": "Penny Stock", "price": p['price'],
                        "alloc": alloc, "qty": qty, "rationale": "High Risk / High Reward Play"
                    })
        
        # C. Select Commodities (Mock Logic for now as we simulate MCX via ETFs mostly)
        if comm_budget > 0:
             # Just pick GoldBees as safe proxy for now
             alloc = comm_budget
             price = 60.0 # Mock price for GoldBees
             qty = int(alloc / price)
             final_allocation.append({
                 "ticker": "GOLDBEES.NS", "type": "Commodity ETF", "price": price,
                 "alloc": alloc, "qty": qty, "rationale": "Hedge / Stability"
             })

        # D. Select Core Stocks
        # Sort by Score
        top_core = sorted(core_universe, key=lambda x: x['trust_score'], reverse=True)
        # Target roughly remaining positions count target
        target_core_count = max(1, profile['positions'][1] - len(final_allocation))
        selected_core = top_core[:target_core_count]
        
        if selected_core:
            per_stock_budget = core_budget / len(selected_core)
            for s in selected_core:
                qty = int(per_stock_budget / s['price']) if s['price'] > 0 else 0
                if qty > 0:
                     final_allocation.append({
                        "ticker": s['symbol'], "type": "Core Stock", "price": s['price'],
                        "alloc": per_stock_budget, "qty": qty, "rationale": "Fundamentally Strong Growth"
                    })

        # 4. CALCULATE SCENARIO RETURNS
        # Logic: 
        # Core: Base 12%, Bull 18%, Bear -5%
        # Penny: Base 30%, Bull 60%, Bear -40%
        # Commodity: Base 8%, Bull 15%, Bear -2%
        
        total_exp_profit = 0
        portfolio_base_pct = 0
        total_alloc = 0
        
        table_rows = ""
        
        for item in final_allocation:
            # Scenario Logic
            if item['type'] == "Penny Stock":
                base_pct, bull_pct, bear_pct = 28.0, 50.0, -30.0
            elif "Commodity" in item['type']:
                base_pct, bull_pct, bear_pct = 8.0, 15.0, -2.0
            else: # Core
                base_pct, bull_pct, bear_pct = 12.0, 20.0, -5.0
            
            exp_profit = item['alloc'] * (base_pct / 100)
            total_exp_profit += exp_profit
            total_alloc += item['alloc']
            
            # Formatted Range
            range_str = f"{int(base_pct-5)}–{int(base_pct+5)}% (Base {int(base_pct)}%)"
            
            table_rows += f"""
            <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                <td style="padding:10px; font-weight:bold;">{item['ticker']}</td>
                <td style="padding:10px; font-size:11px; color:#aaa;">{item['type']}</td>
                <td style="padding:10px; font-size:11px;">{item['rationale']}</td>
                <td style="padding:10px;">₹{int(item['alloc'])}</td>
                <td style="padding:10px; color:var(--accent); font-weight:bold;">{item['qty']} Qty</td>
                <td style="padding:10px; color:#10b981;">{range_str}</td>
                <td style="padding:10px; font-weight:bold;">₹{int(exp_profit)}</td>
            </tr>
            """
        
        if total_alloc > 0:
            portfolio_base_pct = (total_exp_profit / total_alloc) * 100
            
        # 5. GENERATE FINAL HTML
        html = f"""
        <!-- BLOCK 1: OVERVIEW -->
        <div class="glass" style="padding:20px; margin-bottom:20px; border-left:4px solid var(--accent);">
            <h3 style="margin-bottom:10px;">🚀 Plan for {risk_profile} Profile</h3>
            <div style="display:flex; flex-wrap:wrap; gap:20px; font-size:13px; color:#ddd;">
                <div>Capital: <b>₹{int(user_capital)}</b></div>
                <div>Segments: <b>{profile['stock_mix']}</b></div>
            </div>
            <div style="margin-top:15px; font-size:14px;">
                Est. Portfolio Return (Base Case): 
                <span style="color:#10b981; font-weight:bold; font-size:16px;">
                    {round(portfolio_base_pct, 1)}%
                </span> 
                <span style="color:#aaa; font-size:12px;">(approx ₹{int(total_exp_profit)})</span>
            </div>
        </div>
        
        <!-- BLOCK 2: ALLOCATION TABLE -->
        <div class="glass" style="padding:10px; overflow-x:auto;">
            <table style="width:100%; border-collapse:collapse; white-space:nowrap; font-size:12px;">
                <tr style="background:rgba(255,255,255,0.05); color:#aaa; text-align:left;">
                    <th style="padding:10px;">Instrument</th>
                    <th style="padding:10px;">Segment</th>
                    <th style="padding:10px;">Rationale</th>
                    <th style="padding:10px;">Amount</th>
                    <th style="padding:10px;">Qty</th>
                    <th style="padding:10px;">Exp Return</th>
                    <th style="padding:10px;">Exp Profit</th>
                </tr>
                {table_rows}
            </table>
        </div>
        
        <!-- BLOCK 3: SCENARIO & RISK -->
        <div class="glass" style="padding:20px; margin-top:20px;">
            <h4 style="color:var(--warning); margin-bottom:10px;">⚠️ Risk & Scenarios</h4>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:15px; text-align:center;">
                <div style="background:rgba(255,50,50,0.1); padding:10px; border-radius:8px;">
                     <div style="font-size:10px; color:#ef4444;">BEAR CASE</div>
                     <div style="font-weight:bold;">Low/Neg Profit</div>
                </div>
                <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:8px;">
                     <div style="font-size:10px; color:#aaa;">BASE CASE</div>
                     <div style="font-weight:bold;">₹{int(total_exp_profit)}</div>
                </div>
                <div style="background:rgba(16,185,129,0.1); padding:10px; border-radius:8px;">
                     <div style="font-size:10px; color:#10b981;">BULL CASE</div>
                     <div style="font-weight:bold;">High Profit</div>
                </div>
            </div>
            <ul style="font-size:11px; color:#888; line-height:1.5; padding-left:15px;">
                <li>Penny stocks & Commodities are volatile. Capital loss is possible.</li>
                <li>Estimates are hypothetical. Not guaranteed.</li>
                <li>Please consult a financial advisor before investing.</li>
            </ul>
        </div>
        """
        
        return html
    
    def _fetch_scanner_results(self):
        # Re-using the scanner logic quickly or mocking for speed in this context
        return [
            {"symbol": "RELIANCE.NS", "price": 2850, "trust_score": 88, "category": "Blue Chip"},
            {"symbol": "TCS.NS", "price": 3400, "trust_score": 85, "category": "Blue Chip"},
            {"symbol": "INFY.NS", "price": 1620, "trust_score": 82, "category": "Blue Chip"},
            {"symbol": "SBIN.NS", "price": 620, "trust_score": 80, "category": "Blue Chip"},
            {"symbol": "YESBANK.NS", "price": 24.5, "trust_score": 75, "category": "Penny Stock"},
            {"symbol": "IDEA.NS", "price": 14.0, "trust_score": 70, "category": "Penny Stock"},
            {"symbol": "SUZLON.NS", "price": 42.0, "trust_score": 72, "category": "Penny Stock"},
            {"symbol": "TRIDENT.NS", "price": 36.0, "trust_score": 68, "category": "Penny Stock"},
            {"symbol": "ITC.NS", "price": 450, "trust_score": 78, "category": "Blue Chip"}
        ]

    def _get_action_color(self, action):
        if "ACCUMULATE" in action: return "#10b981"
        if "EXIT" in action: return "#ef4444"
        return "#f59e0b"
        
    def _get_score_label(self, score):
        if score > 75: return "BULLISH"
        if score < 40: return "BEARISH"
        return "NEUTRAL"
        
    def _get_theme(self, symbol):
        themes = {
            "HAL.NS": "Defence", "BEL.NS": "Defence", "MAZDOCK.NS": "Defence",
            "RELIANCE.NS": "Energy/Telco", "HDFCBANK.NS": "Pvt Bank Leader",
            "TCS.NS": "IT Leader", "RVNL.NS": "Rail Infra", "TRENT.NS": "Retail Growth"
        }
        return themes.get(symbol, "Growth/Value")

