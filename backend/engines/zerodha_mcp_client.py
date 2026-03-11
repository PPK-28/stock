import asyncio
import sys
from mcp.client.sse import sse_client
from mcp import ClientSession

async def setup_zerodha_mcp():
    """
    Connects to the Zerodha Kite MCP Server via Server-Sent Events (SSE).
    This allows you to securely access market data, portfolio, and place orders.
    """
    url = "https://mcp.kite.trade/sse"
    print(f"Connecting to Zerodha MCP at {url}...")
    
    # Establish SSE connection
    async with sse_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize connection
            await session.initialize()
            print("Successfully connected!")
            
            # Fetch available tools
            tools_result = await session.list_tools()
            print("\nAvailable Zerodha MCP Tools:")
            for t in tools_result.tools:
                print(f" - {t.name}: {t.description}")

            
            # Example usage to get Last Traded Price (LTP) for an instrument:
            print("\nAttempting to fetch LTP for NSE:INFY and NSE:SBIN...")
            result = await session.call_tool("get_ltp", arguments={"instruments": ["NSE:INFY", "NSE:SBIN"]})
            
            # Check if login is required
            if result.isError and "log in" in result.content[0].text.lower():
                print("\n[Auth] Authentication required. Fetching login link...")
                login_result = await session.call_tool("login", arguments={})
                print("\n=== LOGIN REQUIRED ===")
                for content in login_result.content:
                    try:
                        print(content.text)
                    except UnicodeEncodeError:
                        print(content.text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
                print("======================\n")
                print("Note: Once you click the link and log in, your session is authenticated.")
                input("\nPress ENTER here in the console AFTER you have successfully logged in in the browser...")
                
                print("\nRetrying to fetch LTP...")
                result = await session.call_tool("get_ltp", arguments={"instruments": ["NSE:INFY", "NSE:SBIN"]})
                if result.isError:
                    print(f"Still getting error: {result.content}")
                else:
                    print("\n[Success] LTP Result:")
                    for content in result.content:
                        print(content.text)
                    
            else:
                print("\n[Success] LTP Result:")
                for content in result.content:
                    print(content.text)

if __name__ == "__main__":
    # Note: Requires `pip install mcp`
    asyncio.run(setup_zerodha_mcp())
