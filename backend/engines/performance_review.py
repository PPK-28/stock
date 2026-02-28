
from backend.engines.analyzer import StockAnalyzer
from backend.jobs.scanner import _get_cached, _set_cached
import yfinance as yf
import datetime

class PerformanceReviewEngine:
    def __init__(self):
        self.analyzer = StockAnalyzer()

    def generate_review(self, benchmark_index="NIFTY 50"):
        # Check cache first
        cached = _get_cached("review_data")
        if cached:
            print("[Review] Returning cached data")
            return cached
        
        # Portfolio recommendations to review
        mock_portfolio = [
            {"ticker": "RELIANCE", "reco_price": 1350, "action": "BUY", "target": 1550, "stop_loss": 1280, "date": "20-Feb"},
            {"ticker": "HDFCBANK", "reco_price": 860, "action": "BUY", "target": 950, "stop_loss": 820, "date": "21-Feb"},
            {"ticker": "YESBANK", "reco_price": 19.5, "action": "BUY", "target": 24, "stop_loss": 17, "date": "22-Feb"},
            {"ticker": "SBIN", "reco_price": 780, "action": "BUY", "target": 850, "stop_loss": 740, "date": "24-Feb"},
            {"ticker": "INFY", "reco_price": 1580, "action": "BUY", "target": 1750, "stop_loss": 1500, "date": "24-Feb"},
        ]
        
        benchmark_return = 1.8
        
        # Batch fetch ALL review stock prices in ONE call
        symbols = [item['ticker'] + ".NS" for item in mock_portfolio]
        prices = {}
        try:
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
            print(f"[Review] Batch fetched {len(prices)}/{len(symbols)} prices")
        except Exception as e:
            print(f"[Review] Batch error: {e}")
        
        results = {"table": [], "app_score": 0, "avg_return": 0, "hit_rate": 0, "outperform_rate": 0, "total": 0}
        total_pnl = 0
        hits = 0
        outperforms = 0
        winners = []
        losers = []
        
        for item in mock_portfolio:
            sym = item['ticker'] + ".NS"
            if sym not in prices:
                continue
            
            price_now = prices[sym]
            
            pnl_pct = ((price_now - item['reco_price']) / item['reco_price']) * 100
            rel_perf = pnl_pct - benchmark_return
            
            if rel_perf > 3: tag = "OUTPERFORM"
            elif rel_perf < -3: tag = "UNDERPERFORM"
            else: tag = "IN LINE"
            
            if "OUTPERFORM" in tag: outperforms += 1
            
            idea_score = 50
            is_correct_dir = False
            
            if item['action'] == "BUY":
                if pnl_pct > 0: 
                    is_correct_dir = True
                    idea_score += 20
                else:
                    idea_score -= 20
            
            if rel_perf > 0: idea_score += 15
            elif rel_perf < -2: idea_score -= 15
            
            idea_score = max(0, min(100, idea_score))
            
            if is_correct_dir: hits += 1
            total_pnl += pnl_pct
            
            if pnl_pct > 0:
                winners.append({"ticker": item['ticker'], "pnl": round(pnl_pct, 1)})
            else:
                losers.append({"ticker": item['ticker'], "pnl": round(pnl_pct, 1)})
            
            results['table'].append({
                "date": item['date'],
                "ticker": item['ticker'],
                "action": item['action'],
                "entry": item['reco_price'],
                "now": round(price_now, 2),
                "pnl": round(pnl_pct, 1),
                "vs_bench": round(rel_perf, 1),
                "tag": tag,
                "score": int(idea_score)
            })
                
        count = len(results['table'])
        if count > 0:
            avg_ret = total_pnl / count
            hit_rate = (hits / count) * 100
            outperform_rate = (outperforms / count) * 100
            
            app_score = 50
            if avg_ret >= benchmark_return + 3: app_score += 25
            elif avg_ret >= benchmark_return - 3: app_score += 10
            else: app_score -= 10
            
            if hit_rate >= 70: app_score += 25
            elif hit_rate >= 50: app_score += 15
            
            results['app_score'] = int(max(0, min(100, app_score)))
            results['avg_return'] = round(avg_ret, 1)
            results['hit_rate'] = round(hit_rate, 1)
            results['total'] = count

        html = self._generate_html_report(results, benchmark_return, winners, losers)
        _set_cached("review_data", html)
        return html

    def _generate_html_report(self, data, bench_ret, winners, losers):
        s = data['app_score']
        score_color = "#10b981" if s >= 70 else ("#f59e0b" if s >= 50 else "#ef4444")
        
        html = f"""
        <div class="glass" style="padding:20px; margin-bottom:20px;">
            <div style="font-size:12px; color:#aaa; margin-bottom:5px;">LAST 7 DAYS REVIEW</div>
            <div style="font-size:14px; margin-bottom:15px;">
                Suggested <b>{data['total']} ideas</b>. Avg Return: <b>{data['avg_return']}%</b> vs Nifty {bench_ret}%.
            </div>
            <div style="display:flex; align-items:center; gap:15px;">
                <div style="font-size:16px;">Overall App Score:</div>
                <div style="font-size:24px; font-weight:bold; color:{score_color}; border:2px solid {score_color}; padding:5px 15px; border-radius:12px;">
                    {s}/100
                </div>
            </div>
        </div>
        """
        
        html += '<div style="display:flex; flex-direction:column; gap:12px; margin-bottom:20px;">'
        
        for r in data['table']:
            pnl_c = "#10b981" if r['pnl'] > 0 else "#ef4444"
            bench_c = "#10b981" if r['vs_bench'] > 0 else "#ef4444"
            pnl_sign = "+" if r['pnl'] > 0 else ""
            bench_sign = "+" if r['vs_bench'] > 0 else ""
            
            tag_bg = "rgba(16,185,129,0.15)" if r['tag'] == "OUTPERFORM" else (
                "rgba(239,68,68,0.15)" if r['tag'] == "UNDERPERFORM" else "rgba(255,255,255,0.05)"
            )
            tag_color = "#10b981" if r['tag'] == "OUTPERFORM" else (
                "#ef4444" if r['tag'] == "UNDERPERFORM" else "#aaa"
            )
            sc_color = "#10b981" if r['score'] >= 65 else ("#f59e0b" if r['score'] >= 45 else "#ef4444")
            
            html += f"""
            <div class="glass" style="padding:15px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <div>
                        <span style="font-weight:700; font-size:15px;">{r['ticker']}</span>
                        <span class="badge" style="margin-left:8px; font-size:9px;">{r['action']}</span>
                        <div style="font-size:11px; color:#888; margin-top:2px;">Rec: {r['date']}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:18px; font-weight:700; color:{pnl_c};">{pnl_sign}{r['pnl']}%</div>
                        <div style="font-size:10px; color:{bench_c};">vs Nifty: {bench_sign}{r['vs_bench']}%</div>
                    </div>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; font-size:12px;">
                    <div style="color:#aaa;">Entry<br><span style="color:#fff; font-weight:600;">Rs.{r['entry']}</span></div>
                    <div style="color:#aaa;">Now<br><span style="color:#fff; font-weight:600;">Rs.{r['now']}</span></div>
                    <div style="color:#aaa;">Score<br><span style="color:{sc_color}; font-weight:700;">{r['score']}/100</span></div>
                </div>
                <div style="margin-top:8px;">
                    <span style="font-size:10px; padding:3px 8px; border-radius:6px; background:{tag_bg}; color:{tag_color};">{r['tag']}</span>
                </div>
            </div>
            """
        
        html += '</div>'
        
        # Performance Notes
        if winners:
            winners_sorted = sorted(winners, key=lambda x: x['pnl'], reverse=True)
            top_winner = winners_sorted[0]
            what_worked = f"{top_winner['ticker']} led with +{top_winner['pnl']}% gain."
            if len(winners_sorted) > 1:
                what_worked += f" {winners_sorted[1]['ticker']} also positive (+{winners_sorted[1]['pnl']}%)."
        else:
            what_worked = "No winning picks this week. Market conditions were challenging."
        
        if losers:
            losers_sorted = sorted(losers, key=lambda x: x['pnl'])
            worst_loser = losers_sorted[0]
            what_failed = f"{worst_loser['ticker']} dropped {worst_loser['pnl']}%."
            if len(losers_sorted) > 1:
                what_failed += f" {losers_sorted[1]['ticker']} also underperformed ({losers_sorted[1]['pnl']}%)."
        else:
            what_failed = "All picks were positive this week."
        
        hit_rate = data.get('hit_rate', 0)
        if hit_rate >= 60:
            adjustment = "Maintaining current strategy. Hit rate is strong."
        elif hit_rate >= 40:
            adjustment = "Will tighten stop-losses and focus on stronger setups."
        else:
            adjustment = "Reviewing strategy. Will reduce position sizes and shift to defensive picks."
        
        html += f"""
        <div class="glass" style="padding:20px; margin-top:10px;">
            <h4 style="margin-bottom:10px;">Performance Notes</h4>
            <ul style="font-size:12px; color:#ccc; line-height:1.8; padding-left:20px;">
                <li><b>What Worked:</b> {what_worked}</li>
                <li><b>What Failed:</b> {what_failed}</li>
                <li><b>Adjustment:</b> {adjustment}</li>
            </ul>
            <div style="font-size:10px; color:#666; margin-top:15px; text-align:center;">
                Past performance metrics are for review only and do not guarantee future results.
            </div>
        </div>
        """
        
        return html
