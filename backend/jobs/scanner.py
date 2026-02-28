from backend.engines.analyzer import StockAnalyzer
import yfinance as yf
import time
import threading

# ── In-Memory Cache ──
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
        Scans NSE stocks using batch download (single API call)
        then analyzes the top performers individually with delays.
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
                "SBIN.NS", "ITC.NS", "BHARTIARTL.NS", "LICI.NS",
                "SILVERBEES.NS", "GOLDBEES.NS", "MARUTI.NS", "M&M.NS", "ASIANPAINT.NS"
            ]
            
            # 2. Popular Small Cap Stocks
            pennies = [
                "YESBANK.NS", "IDEA.NS", "SUZLON.NS", "JPPOWER.NS", "RPOWER.NS",
                "TRIDENT.NS", "IRFC.NS", "RVNL.NS", "NHPC.NS",
                "ZOMATO.NS", "PAYTM.NS"
            ]
            
            all_symbols = majors + pennies
            
            # ── STEP 1: Batch price fetch (single API call) ──
            print(f"Scanner: Batch downloading {len(all_symbols)} stocks...")
            prices = self._batch_fetch_prices(all_symbols)
            print(f"Scanner: Got prices for {len(prices)} stocks")
            
            if len(prices) < 3:
                print("Scanner: Too few prices from batch. Returning fallback.")
                return self._get_fallback_data()
            
            # ── STEP 2: Analyze stocks that have prices (with delays) ──
            results = []
            analyzed = 0
            
            for symbol in all_symbols:
                if symbol not in prices:
                    continue
                
                try:
                    if analyzed > 0:
                        time.sleep(2.0)  # 2s delay between full analyses
                    
                    result = self.analyzer.analyze_stock(symbol)
                    if result and result.get('verdict') != 'ERROR':
                        result['category'] = "Penny Stock" if symbol in pennies else "Blue Chip"
                        results.append(result)
                        analyzed += 1
                        print(f"Scanner: ✓ {symbol} ({analyzed}/15)")
                    else:
                        # Analysis failed but we have price — create minimal entry
                        results.append(self._create_minimal_entry(symbol, prices[symbol], pennies))
                        analyzed += 1
                        print(f"Scanner: ~ {symbol} (price-only, {analyzed}/15)")
                    
                    if analyzed >= 15:
                        break
                        
                except Exception as e:
                    # Still use batch price as fallback
                    if symbol in prices:
                        results.append(self._create_minimal_entry(symbol, prices[symbol], pennies))
                        analyzed += 1
                        print(f"Scanner: ~ {symbol} (fallback, {analyzed}/15)")
                    else:
                        print(f"Scanner: ✗ {symbol}: {e}")
                    continue
            
            # Sort by Trust Score, take top 15
            if results:
                top_picks = sorted(results, key=lambda x: x.get('trust_score', 0), reverse=True)[:15]
                _set_cached("trending_scan", top_picks)
                print(f"Scanner: Done! {len(top_picks)} stocks cached.")
                return top_picks
            
            print("Scanner: No results. Returning fallback.")
            return self._get_fallback_data()
            
        except Exception as e:
            print(f"Scanner Error: {e}. Returning Fallback.")
            return self._get_fallback_data()
    
    def _batch_fetch_prices(self, symbols):
        """Fetch prices for ALL stocks in a SINGLE yfinance API call."""
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
        except Exception as e:
            print(f"Scanner: Batch download error: {e}")
        return prices
    
    def _create_minimal_entry(self, symbol, price, pennies):
        """Create a minimal stock entry using just the batch-fetched price."""
        return {
            "symbol": symbol,
            "price": round(price, 2),
            "trust_score": 50,
            "verdict": "HOLD",
            "reasoning": "Analysis pending — showing live price data.",
            "risk": "Medium",
            "category": "Penny Stock" if symbol in pennies else "Blue Chip",
            "advisory": {
                "entry": f"₹{round(price, 2)}",
                "target": f"{round(price * 1.05, 1)}",
                "stop_loss": f"{round(price * 0.95, 1)}",
                "analyst_rating": "HOLD (Conf: 50%)"
            },
            "futures_data": {}
        }
    
    def _get_fallback_data(self):
        """Fallback data when all scanning fails."""
        return [
            {
                "symbol": "RELIANCE.NS", "price": 1393.9, "trust_score": 45,
                "verdict": "HOLD", "reasoning": "Live data temporarily unavailable.",
                "risk": "Medium", "category": "Blue Chip",
                "advisory": {"entry": "₹1393.9", "target": "1463.6", "stop_loss": "1342.7", "analyst_rating": "HOLD (Conf: 45%)"},
                "futures_data": {}
            },
            {
                "symbol": "SBIN.NS", "price": 1201.7, "trust_score": 67,
                "verdict": "BUY", "reasoning": "Strong technicals and institutional support.",
                "risk": "Low", "category": "Blue Chip",
                "advisory": {"entry": "₹1201.7", "target": "1261.8", "stop_loss": "1157.8", "analyst_rating": "BUY (Conf: 67%)"},
                "futures_data": {}
            },
            {
                "symbol": "TCS.NS", "price": 3850.0, "trust_score": 55,
                "verdict": "HOLD", "reasoning": "IT sector under pressure but fundamentals intact.",
                "risk": "Medium", "category": "Blue Chip",
                "advisory": {"entry": "₹3850", "target": "4042.5", "stop_loss": "3707.5", "analyst_rating": "HOLD (Conf: 55%)"},
                "futures_data": {}
            },
            {
                "symbol": "INFY.NS", "price": 1620.0, "trust_score": 52,
                "verdict": "HOLD", "reasoning": "Stable IT stock with moderate growth.",
                "risk": "Medium", "category": "Blue Chip",
                "advisory": {"entry": "₹1620", "target": "1701", "stop_loss": "1539", "analyst_rating": "HOLD (Conf: 52%)"},
                "futures_data": {}
            },
            {
                "symbol": "HDFCBANK.NS", "price": 1750.0, "trust_score": 60,
                "verdict": "BUY", "reasoning": "Leading private bank with strong fundamentals.",
                "risk": "Low", "category": "Blue Chip",
                "advisory": {"entry": "₹1750", "target": "1837.5", "stop_loss": "1687.5", "analyst_rating": "BUY (Conf: 60%)"},
                "futures_data": {}
            },
            {
                "symbol": "ICICIBANK.NS", "price": 1280.0, "trust_score": 58,
                "verdict": "BUY", "reasoning": "Consistent growth and improving asset quality.",
                "risk": "Low", "category": "Blue Chip",
                "advisory": {"entry": "₹1280", "target": "1344", "stop_loss": "1216", "analyst_rating": "BUY (Conf: 58%)"},
                "futures_data": {}
            },
            {
                "symbol": "ITC.NS", "price": 410.0, "trust_score": 63,
                "verdict": "BUY", "reasoning": "Diversified conglomerate with strong FMCG growth.",
                "risk": "Low", "category": "Blue Chip",
                "advisory": {"entry": "₹410", "target": "430.5", "stop_loss": "389.5", "analyst_rating": "BUY (Conf: 63%)"},
                "futures_data": {}
            },
            {
                "symbol": "IRFC.NS", "price": 125.0, "trust_score": 48,
                "verdict": "HOLD", "reasoning": "Government railway financing stock.",
                "risk": "Medium", "category": "Penny Stock",
                "advisory": {"entry": "₹125", "target": "131.3", "stop_loss": "118.8", "analyst_rating": "HOLD (Conf: 48%)"},
                "futures_data": {}
            }
        ]
