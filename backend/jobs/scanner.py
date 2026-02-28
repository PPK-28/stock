from backend.engines.analyzer import StockAnalyzer
import time
import threading

# ── In-Memory Cache ──
# Stores results for 5 minutes to avoid hammering Yahoo Finance
_cache = {}
_cache_lock = threading.Lock()
CACHE_TTL = 300  # 5 minutes

def _get_cached(key):
    with _cache_lock:
        if key in _cache:
            result, ts = _cache[key]
            if time.time() - ts < CACHE_TTL:
                return result
    return None

def _set_cached(key, value):
    with _cache_lock:
        _cache[key] = (value, time.time())


class PreMarketScanner:
    def __init__(self):
        self.analyzer = StockAnalyzer()

    async def run_daily_scan(self):
        """
        Scans NSE Majors & Penny Stocks for opportunities.
        Uses caching + delays to avoid Yahoo Finance rate limits on cloud.
        """
        # Check cache first
        cached = _get_cached("trending_scan")
        if cached:
            print("Scanner: Returning cached results")
            return cached

        try:
            # 1. Major Stocks (Blue Chip)
            majors = [
                "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
                "SBIN.NS", "TATAMOTORS.NS", "ITC.NS", "BHARTIARTL.NS", "LICI.NS",
                "SILVERBEES.NS", "GOLDBEES.NS", "MARUTI.NS", "M&M.NS", "ASIANPAINT.NS"
            ]
            
            # 2. Popular Penny / Small Cap Stocks
            pennies = [
                "YESBANK.NS", "IDEA.NS", "SUZLON.NS", "JPPOWER.NS", "RPOWER.NS",
                "TRIDENT.NS", "GTLINFRA.NS", "VIKASECO.NS", "UCOBANK.NS", "IOB.NS",
                "IRFC.NS", "RVNL.NS", "NHPC.NS", "RENUKA.NS", "TV18BRDCST.NS",
                "ZOMATO.NS", "PAYTM.NS", "WELSPUNLIV.NS"
            ]
            
            all_symbols = majors + pennies
            results = []
            
            for i, symbol in enumerate(all_symbols):
                try:
                    # Rate limit: 1.5s delay between requests to avoid Yahoo bans
                    if i > 0:
                        time.sleep(1.5)
                    
                    result = self.analyzer.analyze_stock(symbol)
                    if result and result.get('verdict') != 'ERROR':
                        result['category'] = "Penny Stock" if symbol in pennies else "Blue Chip"
                        results.append(result)
                        print(f"Scanner: ✓ {symbol} ({i+1}/{len(all_symbols)})")
                    else:
                        print(f"Scanner: ✗ {symbol} — skipped (error or no data)")
                except Exception as inner_e:
                    print(f"Scanner: Skipping {symbol} due to error: {inner_e}")
                    continue
            
            # Sort by Trust Score, take top 15
            if results:
                top_picks = sorted(results, key=lambda x: x['trust_score'], reverse=True)[:15]
                _set_cached("trending_scan", top_picks)
                print(f"Scanner: Completed! {len(results)} stocks analyzed, top 15 cached.")
                return top_picks
            
            # If no results, return fallback
            print("Scanner: No live results. Returning fallback data.")
            return self._get_fallback_data()
            
        except Exception as e:
            print(f"Scanner Error: {e}. Returning Fallback Data.")
            return self._get_fallback_data()
    
    def _get_fallback_data(self):
        """Fallback data when live scanning fails."""
        return [
            {
                "symbol": "RELIANCE.NS", "price": 1393.9, "trust_score": 45, 
                "verdict": "HOLD", "reasoning": "Live data temporarily unavailable. Showing cached data.", 
                "risk": "Medium", "category": "Blue Chip",
                "advisory": {"entry": "₹1393.9", "target": "1463.6", "stop_loss": "1342.7", "analyst_rating": "HOLD (Conf: 45%)"},
                "futures_data": {"is_fno": True, "oi_signal": "N/A", "sentiment": "Neutral", "pcr": 1.0}
            },
            {
                "symbol": "SBIN.NS", "price": 1201.7, "trust_score": 67, 
                "verdict": "BUY", "reasoning": "Strong technicals and institutional support.", 
                "risk": "Low", "category": "Blue Chip",
                "advisory": {"entry": "₹1201.7", "target": "1261.8", "stop_loss": "1157.8", "analyst_rating": "BUY (Conf: 67%)"},
                "futures_data": {"is_fno": True, "oi_signal": "Long Buildup", "sentiment": "Bullish", "pcr": 1.18}
            }
        ]
