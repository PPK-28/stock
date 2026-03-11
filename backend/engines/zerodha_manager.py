import asyncio
import threading
import sys
import json
from mcp.client.sse import sse_client
from mcp import ClientSession

class ZerodhaManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ZerodhaManager, cls).__new__(cls)
            cls._instance._init_state()
        return cls._instance
        
    def _init_state(self):
        self.status = "DISCONNECTED" # "DISCONNECTED", "NEEDS_LOGIN", "CONNECTED"
        self.login_url = None
        self.session = None
        self.loop = None
        self.thread = None
        
    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.thread.start()
            
    def _run_async_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._connect_mcp())
        except Exception as e:
            print(f"[Zerodha] Connection error: {e}")
            self.status = "DISCONNECTED"
            
    async def _connect_mcp(self):
        url = "https://mcp.kite.trade/sse"
        self.status = "CONNECTING"
        print(f"[Zerodha] Connecting to {url}...")
        
        try:
            async with sse_client(url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self.session = session
                    
                    # Check if login needed by calling get_ltp on dummy
                    result = await session.call_tool("get_ltp", arguments={"instruments": ["NSE:INFY"]})
                    if result.isError and "log in" in result.content[0].text.lower():
                        await self._fetch_login_url()
                    else:
                        self.status = "CONNECTED"
                        print("[Zerodha] Successfully connected and authenticated!")
                        
                    # Keep the connection alive
                    while True:
                        await asyncio.sleep(10)
        except Exception as e:
            print(f"[Zerodha] SSE Error: {e}")
            self.status = "DISCONNECTED"
            
    async def _fetch_login_url(self):
        print("[Zerodha] Fetching login URL...")
        login_result = await self.session.call_tool("login", arguments={})
        for content in login_result.content:
            try:
                text = content.text
            except UnicodeEncodeError:
                text = content.text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
                
            # Extract URL heuristically
            if "https://kite.zerodha.com" in text:
                import re
                urls = re.findall(r'(https?://[^\s)\]]+)', text)
                if urls:
                    self.login_url = urls[-1] # take the last one which is usually the plain URL
                    break
        
        self.status = "NEEDS_LOGIN"
        print(f"[Zerodha] Needs Login. URL: {self.login_url}")

    def call_tool_sync(self, name, arguments):
        """Helper to invoke MCP tools synchronously from FastAPI threads."""
        if not self.loop or not self.session:
            return {"error": "Zerodha MCP disconnected"}
        
        future = asyncio.run_coroutine_threadsafe(
            self.session.call_tool(name, arguments=arguments), 
            self.loop
        )
        try:
            result = future.result(timeout=10)
            if result.isError:
                text = result.content[0].text
                # Auto detect session drops
                if "log in" in text.lower() and self.status == "CONNECTED":
                    self.status = "NEEDS_LOGIN"
                    asyncio.run_coroutine_threadsafe(self._fetch_login_url(), self.loop)
                return {"error": text}
                
            # Parse JSON if possible
            text = result.content[0].text
            try:
                return {"data": json.loads(text)}
            except:
                return {"data": text}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    # --- Domain Specific Helpers ---
    def get_ltp(self, symbols: list):
        if self.status != "CONNECTED":
            return {"error": "Not logged in"}
        res = self.call_tool_sync("get_ltp", {"instruments": symbols})
        return res
        
    def check_auth(self):
        """Forces a check on auth status by calling a ping"""
        if self.status == "NEEDS_LOGIN":
            # Attempt to call LTP to see if user magically logged in browser
            res = self.call_tool_sync("get_ltp", {"instruments": ["NSE:INFY"]})
            if "error" not in res:
                self.status = "CONNECTED"
                self.login_url = None
        return {"status": self.status, "login_url": self.login_url}
