import urllib.parse
import urllib.request
import asyncio
import datetime
from backend.jobs.scanner import PreMarketScanner

class WhatsAppNotifier:
    def __init__(self):
        # REPLACE THESE WITH YOUR DETAILS
        self.phone_number = "+919988776655" # Example: +919876543210
        self.api_key = "123456"            # Get this from CallMeBot
        self.scanner = PreMarketScanner()

    async def generate_and_send_daily_alert(self):
        print(f"[Notifier] Running Daily Scan at {datetime.datetime.now()}")
        
        # 1. Run Scan
        top_picks = await self.scanner.run_daily_scan()
        
        if not top_picks:
            print("[Notifier] No strong buy signals today.")
            return

        # 2. Format Message
        message = "🚀 *Market Morning Alert* 🚀\n"
        message += f"Date: {datetime.date.today()}\n\n"
        
        for stock in top_picks:
            icon = "💎" if stock['category'] == "Blue Chip" else "🪙"
            entry_price = stock['advisory'].get('entry', 'N/A')
            target = stock['advisory'].get('target', 'N/A')
            
            message += f"{icon} *{stock['symbol']}* ({stock['category']})\n"
            message += f"   • Price: ₹{stock['price']}\n"
            message += f"   • Entry: {entry_price}\n"
            message += f"   • Target: ₹{target}\n"
            message += f"   • Rating: {stock['verdict']} ({stock['trust_score']}%)\n\n"
            
        message += "⚠️ _Trade responsibly. System generated._"
        
        # 3. Send via CallMeBot
        self.send_message(message)

    def send_message(self, text):
        if self.api_key == "123456":
            print("❌ [Notifier] API Key not set! Message NOT sent.")
            print("To enable WhatsApp:\n1. Add +34 644 10 55 84 to contacts.\n2. Send 'I allow callmebot' to get API Key.\n3. Update backend/jobs/notifier.py")
            return

        encoded_text = urllib.parse.quote(text)
        url = f"https://api.callmebot.com/whatsapp.php?phone={self.phone_number}&text={encoded_text}&apikey={self.api_key}"
        
        try:
            with urllib.request.urlopen(url) as response:
                if response.getcode() == 200:
                    print("✅ [Notifier] WhatsApp Alert Sent Successfully!")
                else:
                    print(f"❌ [Notifier] Failed to send. Code: {response.getcode()}")
        except Exception as e:
            print(f"❌ [Notifier] Error connecting to WhatsApp API: {e}")

# Test Run (Uncomment to test immediately)
# if __name__ == "__main__":
#     notifier = WhatsAppNotifier()
#     import asyncio
#     asyncio.run(notifier.generate_and_send_daily_alert())
