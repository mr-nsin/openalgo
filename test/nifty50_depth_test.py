"""
Nifty50 Options Market Depth Test Script

This script tests market depth subscriptions for multiple Nifty50 option strikes.

DEPTH LEVEL SUPPORT:
-------------------
1. Regular Depth (5 levels): Available for all exchanges (NSE, NFO, BSE, etc.)
   - Default depth level for most brokers
   - Works for both equity and options

2. 50-Level Depth (TBT): Available only for NSE equity via Fyers TBT WebSocket
   - Use :50 suffix on symbol (e.g., "TCS:50") to request 50-level depth
   - Only supported for NSE equity symbols, NOT for NFO options
   - Requires Fyers broker with TBT WebSocket support

IMPORTANT NOTES:
---------------
- For NFO options (like Nifty50 options), 50-level depth is NOT available via TBT
- Fyers TBT WebSocket only supports NSE equity symbols
- For NFO options, the script will use regular depth (typically 5 levels)
- To test 50-level depth, use NSE equity symbols with use_50_level=True

USAGE:
------
python nifty50_depth_test.py

The script will:
1. Get current Nifty50 expiry
2. Calculate ATM strike
3. Subscribe to multiple strikes around ATM (both CE and PE)
4. Display real-time market depth data
"""

import sys
import os
import time
import json
import threading
import websocket
from typing import List, Dict, Any, Callable, Optional
from queue import Queue
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.symbol import SymToken, db_session
from services.option_symbol_service import (
    get_available_strikes,
    construct_option_symbol,
    find_option_in_database,
    find_atm_strike_from_actual
)
from services.quotes_service import get_quotes
from services.expiry_service import get_expiry_dates

class Nifty50DepthFeed:
    """A wrapper around the OpenAlgo WebSocket client for Market Depth data for Nifty50 options"""
    
    def __init__(self, host: str = "localhost", port: int = 8765, api_key: Optional[str] = None):
        """
        Initialize the Nifty50DepthFeed
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
            api_key: API key for authentication (loads from .env if not provided)
        """
        self.ws_url = f"ws://{host}:{port}"
        self.api_key = api_key
        
        if not self.api_key:
            # Try to load from .env file
            try:
                from dotenv import load_dotenv
                load_dotenv()
                self.api_key = os.getenv("API_KEY")
            except ImportError:
                print("python-dotenv not installed. Please provide API key explicitly.")
                
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.pending_auth = False
        self.message_queue = Queue()
        
        # Storage for market data
        self.market_data = {}
        self.lock = threading.Lock()
        
        # Callbacks
        self.on_data_callback = None
        
    def connect(self) -> bool:
        """Connect to the WebSocket server"""
        try:
            def on_message(ws, message):
                self.message_queue.put(message)
                self._process_message(message)
                
            def on_error(ws, error):
                print(f"WebSocket error: {error}")
                
            def on_open(ws):
                print(f"Connected to {self.ws_url}")
                self.connected = True
                
            def on_close(ws, close_status_code, close_reason):
                print(f"Disconnected from {self.ws_url}")
                self.connected = False
                self.authenticated = False
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_open=on_open,
                on_close=on_close
            )
            
            # Start WebSocket connection in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection to establish
            timeout = 5
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not self.connected:
                print("Failed to connect to the WebSocket server")
                return False
                
            # Now authenticate
            return self._authenticate()
        except Exception as e:
            print(f"Error connecting to WebSocket: {e}")
            return False
            
    def disconnect(self) -> None:
        """Disconnect from the WebSocket server"""
        if self.ws:
            self.ws.close()
            # Wait for websocket to close
            timeout = 2
            start_time = time.time()
            while self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            self.ws = None
            
    def _authenticate(self) -> bool:
        """Authenticate with the WebSocket server"""
        if not self.connected or not self.api_key:
            print("Cannot authenticate: not connected or no API key")
            return False
            
        auth_msg = {
            "action": "authenticate",
            "api_key": self.api_key
        }
        
        print(f"Authenticating with API key: {self.api_key[:8]}...{self.api_key[-8:]}")
        self.ws.send(json.dumps(auth_msg))
        self.pending_auth = True
        
        # Wait for authentication response synchronously
        timeout = 5
        start_time = time.time()
        while not self.authenticated and time.time() - start_time < timeout:
            # Process any messages in the queue
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get(block=False)
                    self._process_message(message)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error processing message: {e}")
                time.sleep(0.1)
                
        if self.authenticated:
            print("Authentication successful!")
            return True
        else:
            print("Authentication failed or timed out")
            return False
        
    def _process_message(self, message_str: str) -> None:
        """Process incoming WebSocket messages"""
        try:
            message = json.loads(message_str)
            
            # Handle authentication response
            if message.get("type") == "auth":
                if message.get("status") == "success":
                    print(f"Authentication response: {message}")
                    self.authenticated = True
                    self.pending_auth = False
                else:
                    print(f"Authentication failed: {message}")
                    self.pending_auth = False
                return
                
            # Handle subscription response
            if message.get("type") == "subscribe":
                print(f"Subscription response: {message}")
                return
                
            # Handle market data for market depth
            if message.get("type") == "market_data":
                exchange = message.get("exchange")
                symbol = message.get("symbol")
                if exchange and symbol:
                    symbol_key = f"{exchange}:{symbol}"
                    mode = message.get("mode")
                    market_data = message.get("data", {})
                    
                    if mode == 3 and "depth" in market_data:  # Depth mode
                        # Extract depth data
                        depth_data = {
                            'ltp': market_data.get("ltp", 0),
                            'open': market_data.get("open", 0),
                            'high': market_data.get("high", 0),
                            'low': market_data.get("low", 0),
                            'close': market_data.get("close", 0),
                            'depth': market_data.get("depth", {"buy": [], "sell": []})
                        }
                        
                        # Store the depth data in our cache
                        with self.lock:
                            self.market_data[symbol_key] = depth_data
                        
                        # Print when depth data is received
                        self._print_depth(symbol_key, depth_data)
                        
                        # Invoke callback if set
                        if self.on_data_callback:
                            self.on_data_callback(message)
        except json.JSONDecodeError:
            print(f"Invalid JSON message: {message_str}")
        except Exception as e:
            print(f"Error handling message: {e}")
    
    def _print_depth(self, symbol_key: str, depth_data: Dict[str, Any]) -> None:
        """Print market depth data in formatted form"""
        buy_depth = depth_data.get('depth', {}).get('buy', [])
        sell_depth = depth_data.get('depth', {}).get('sell', [])
        
        # Count actual depth levels received
        buy_levels = len(buy_depth) if buy_depth else 0
        sell_levels = len(sell_depth) if sell_depth else 0
        max_levels = max(buy_levels, sell_levels)
        
        print(f"\n{'='*70}")
        print(f"Depth {symbol_key} - LTP: {depth_data.get('ltp')}")
        print(f"📊 Depth Levels: {max_levels} levels received (Buy: {buy_levels}, Sell: {sell_levels})")
        print(f"{'='*70}")
        
        # Print all buy depth levels
        print("\nBUY DEPTH (Bids):")
        print("-" * 70)
        print(f"{'Level':<8} {'Price':<15} {'Quantity':<15} {'Orders':<12}")
        print("-" * 70)
        
        if buy_depth:
            for i, level in enumerate(buy_depth):
                price = level.get('price', 'N/A')
                qty = level.get('quantity', 'N/A')
                orders = level.get('orders', 'N/A')
                print(f"{i+1:<8} {price:<15} {qty:<15} {orders:<12}")
        else:
            print("No buy depth data available")
            
        # Print all sell depth levels
        print("\nSELL DEPTH (Asks):")
        print("-" * 70)
        print(f"{'Level':<8} {'Price':<15} {'Quantity':<15} {'Orders':<12}")
        print("-" * 70)
        
        if sell_depth:
            for i, level in enumerate(sell_depth):
                price = level.get('price', 'N/A')
                qty = level.get('quantity', 'N/A')
                orders = level.get('orders', 'N/A')
                print(f"{i+1:<8} {price:<15} {qty:<15} {orders:<12}")
        else:
            print("No sell depth data available")
            
        print("-" * 70)
            
    def subscribe_depth(self, instruments: List[Dict[str, str]], on_data_received: Optional[Callable] = None, use_50_level: bool = False) -> bool:
        """
        Subscribe to Market Depth updates for instruments
        
        Args:
            instruments: List of instrument dictionaries with keys exchange, symbol/exchange_token
            on_data_received: Callback function for data updates
            use_50_level: If True, add :50 suffix to symbol for 50-level depth (NSE equity only via Fyers TBT)
                          Note: NFO options don't support 50-level depth via TBT
        """
        if not self.connected:
            print("Not connected to WebSocket server")
            return False
            
        if not self.authenticated:
            print("Not authenticated with WebSocket server")
            return False
            
        self.on_data_callback = on_data_received
        
        for instrument in instruments:
            exchange = instrument.get("exchange")
            symbol = instrument.get("symbol")
            exchange_token = instrument.get("exchange_token")
            
            # If only exchange_token is provided, we need to map it to a symbol
            if not symbol and exchange_token:
                symbol = exchange_token
                
            if not exchange or not symbol:
                print(f"Invalid instrument: {instrument}")
                continue
            
            # For 50-level depth: Add :50 suffix to symbol (OpenAlgo convention)
            # This triggers TBT WebSocket for Fyers (NSE equity only)
            # Note: TBT only supports NSE equity, not NFO options
            original_symbol = symbol
            if use_50_level:
                if exchange == "NSE":
                    # Add :50 suffix for 50-level depth via TBT
                    symbol = f"{symbol}:50"
                    print(f"📊 Requesting 50-level depth (TBT) for {exchange}:{original_symbol}")
                elif exchange == "NFO":
                    print(f"⚠️  Warning: 50-level depth (TBT) not supported for NFO options. Using regular depth for {exchange}:{original_symbol}")
                    # For NFO, we can't use TBT, so keep original symbol
                    symbol = original_symbol
                else:
                    print(f"⚠️  Warning: 50-level depth may not be supported for {exchange}. Using regular depth for {exchange}:{original_symbol}")
                    symbol = original_symbol
            else:
                print(f"📊 Using regular depth subscription for {exchange}:{symbol}")
            
            # Note: When using :50 suffix, the depth parameter is ignored by OpenAlgo
            # It automatically routes to TBT WebSocket which provides 50 levels
            subscription_msg = {
                "action": "subscribe",
                "symbol": symbol,  # May include :50 suffix for 50-level depth
                "exchange": exchange,
                "mode": 3,  # 3 for Depth
                "depth": 5  # This is ignored when :50 suffix is used (TBT provides 50 levels)
            }
            
            print(f"Subscribing to {exchange}:{symbol} Market Depth")
            self.ws.send(json.dumps(subscription_msg))
            
            # Small delay to ensure the message is processed separately
            time.sleep(0.1)
            
        return True
        
    def unsubscribe_depth(self, instruments: List[Dict[str, str]]) -> bool:
        """
        Unsubscribe from Market Depth updates for instruments
        
        Note: Use the same symbol format (with or without :50 suffix) as used in subscribe
        """
        if not self.connected or not self.authenticated:
            print("Not connected or authenticated")
            return False
            
        for instrument in instruments:
            exchange = instrument.get("exchange")
            symbol = instrument.get("symbol")
            exchange_token = instrument.get("exchange_token")
            
            # If only exchange_token is provided, we need to map it to a symbol
            if not symbol and exchange_token:
                symbol = exchange_token
                
            if not exchange or not symbol:
                print(f"Invalid instrument: {instrument}")
                continue
            
            # Use the symbol as-is (may include :50 suffix if it was used in subscribe)
            unsubscription_msg = {
                "action": "unsubscribe",
                "symbol": symbol,  # Keep :50 suffix if present
                "exchange": exchange,
                "mode": 3  # 3 for Depth
            }
            
            print(f"Unsubscribing from {exchange}:{symbol}")
            self.ws.send(json.dumps(unsubscription_msg))
            
        return True
        
    def get_depth(self, symbol: str = None, exchange: str = None) -> Dict[str, Any]:
        """
        Get the latest market depth data for a symbol or all symbols
        
        Args:
            symbol: Symbol to get data for, or None for all symbols
            exchange: Exchange to get data for
            
        Returns:
            Dict: Market depth data
        """
        depth_data = {}
        with self.lock:
            if symbol and exchange:
                symbol_key = f"{exchange}:{symbol}"
                return self.market_data.get(symbol_key, {})
            else:
                # Return all market data
                depth_data = self.market_data.copy()
        return depth_data
    
    def print_current_depth(self, symbol: str = None, exchange: str = None) -> None:
        """
        Print the current market depth for a symbol or all symbols
        
        Args:
            symbol: Symbol to print depth for, or None for all symbols
            exchange: Exchange to print depth for
        """
        if symbol and exchange:
            symbol_key = f"{exchange}:{symbol}"
            depth_data = self.get_depth(symbol, exchange)
            if depth_data:
                self._print_depth(symbol_key, depth_data)
            else:
                print(f"No depth data available for {symbol_key}")
        else:
            # Print all symbols
            depth_data = self.get_depth()
            if depth_data:
                for symbol_key, data in depth_data.items():
                    self._print_depth(symbol_key, data)
            else:
                print("No depth data available")


def get_nifty50_ltp(api_key: str) -> Optional[float]:
    """Get the current LTP of Nifty50 index"""
    try:
        success, quote_response, status_code = get_quotes(
            symbol="NIFTY",
            exchange="NSE_INDEX",
            api_key=api_key
        )
        
        if success and quote_response.get('data', {}).get('ltp'):
            ltp = quote_response['data']['ltp']
            print(f"✅ Nifty50 LTP: {ltp}")
            return ltp
        else:
            print(f"❌ Failed to get Nifty50 LTP: {quote_response.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"❌ Error getting Nifty50 LTP: {e}")
        return None


def get_current_expiry(api_key: str) -> Optional[str]:
    """Get the current expiry date for Nifty50 options"""
    try:
        success, response, status_code = get_expiry_dates(
            symbol="NIFTY",
            exchange="NFO",
            instrumenttype="options",
            api_key=api_key
        )
        
        if success and response.get('data'):
            expiry_dates = response['data']
            if expiry_dates:
                # Get the first (nearest) expiry
                # Convert from "DD-MMM-YY" to "DDMMMYY" format
                expiry_str = expiry_dates[0]
                # Parse and reformat
                from datetime import datetime
                expiry_date = datetime.strptime(expiry_str, "%d-%b-%y")
                expiry_formatted = expiry_date.strftime("%d%b%y").upper()
                print(f"✅ Current Nifty50 expiry: {expiry_formatted}")
                return expiry_formatted
        print(f"❌ Failed to get expiry dates: {response.get('message', 'Unknown error')}")
        return None
    except Exception as e:
        print(f"❌ Error getting expiry dates: {e}")
        return None


def get_strikes_around_atm(ltp: float, expiry_date: str, num_strikes: int = 5) -> List[float]:
    """
    Get strikes around ATM for Nifty50 options
    
    Args:
        ltp: Last traded price of Nifty50
        expiry_date: Expiry date in DDMMMYY format
        num_strikes: Number of strikes on each side of ATM (default: 5)
    
    Returns:
        List of strike prices around ATM
    """
    try:
        # Get available strikes for CE (both CE and PE have same strikes)
        available_strikes = get_available_strikes("NIFTY", expiry_date, "CE", "NFO")
        
        if not available_strikes:
            print(f"❌ No strikes found for NIFTY {expiry_date}")
            return []
        
        # Find ATM strike
        atm_strike = find_atm_strike_from_actual(ltp, available_strikes)
        if not atm_strike:
            print(f"❌ Could not determine ATM strike")
            return []
        
        print(f"✅ ATM Strike: {atm_strike}")
        
        # Find index of ATM in the list
        atm_index = available_strikes.index(atm_strike)
        
        # Get strikes around ATM
        start_index = max(0, atm_index - num_strikes)
        end_index = min(len(available_strikes), atm_index + num_strikes + 1)
        
        strikes = available_strikes[start_index:end_index]
        print(f"✅ Selected {len(strikes)} strikes around ATM: {strikes[0]} to {strikes[-1]}")
        
        return strikes
    except Exception as e:
        print(f"❌ Error getting strikes: {e}")
        return []


def construct_option_instruments(strikes: List[float], expiry_date: str, option_types: List[str] = ["CE", "PE"], use_50_level: bool = False) -> List[Dict[str, str]]:
    """
    Construct option instruments for subscription
    
    Args:
        strikes: List of strike prices
        expiry_date: Expiry date in DDMMMYY format
        option_types: List of option types (default: ["CE", "PE"])
        use_50_level: If True, add :50 suffix (Note: NFO options don't support 50-level TBT)
    
    Returns:
        List of instrument dictionaries with exchange and symbol
    """
    instruments = []
    
    for strike in strikes:
        for option_type in option_types:
            # Construct option symbol
            option_symbol = construct_option_symbol("NIFTY", expiry_date, strike, option_type)
            
            # Verify symbol exists in database
            option_details = find_option_in_database(option_symbol, "NFO")
            
            if option_details:
                # Note: Fyers TBT only supports NSE equity, not NFO options
                # So :50 suffix won't work for NFO, but we'll add it if requested for consistency
                final_symbol = option_symbol
                if use_50_level:
                    final_symbol = f"{option_symbol}:50"
                    print(f"⚠️  Note: {final_symbol} - 50-level depth (TBT) not supported for NFO options. Will use regular depth.")
                
                instruments.append({
                    "exchange": "NFO",
                    "symbol": final_symbol
                })
                print(f"✅ Added: {final_symbol} (Strike: {strike}, Type: {option_type})")
            else:
                print(f"⚠️  Symbol not found in database: {option_symbol}")
    
    return instruments


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("="*60)
    print("OpenAlgo Nifty50 Options Market Depth Test")
    print("="*60)
    
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("API_KEY not found in .env file")
        api_key = input("Enter your API key: ")
    
    # Step 1: Get current expiry
    print("\n📅 Step 1: Getting current expiry date...")
    expiry_date = get_current_expiry(api_key)
    if not expiry_date:
        print("❌ Could not get expiry date. Exiting.")
        sys.exit(1)
    
    # Step 2: Get Nifty50 LTP
    print("\n💰 Step 2: Getting Nifty50 LTP...")
    nifty_ltp = get_nifty50_ltp(api_key)
    if not nifty_ltp:
        print("❌ Could not get Nifty50 LTP. Exiting.")
        sys.exit(1)
    
    # Step 3: Get strikes around ATM
    print("\n🎯 Step 3: Getting strikes around ATM...")
    num_strikes = 3  # 3 strikes on each side of ATM (total 7 strikes)
    strikes = get_strikes_around_atm(nifty_ltp, expiry_date, num_strikes=num_strikes)
    if not strikes:
        print("❌ Could not get strikes. Exiting.")
        sys.exit(1)
    
    # Step 4: Construct option instruments
    print("\n🔧 Step 4: Constructing option instruments...")
    # Note: For NFO options, 50-level depth via TBT is not supported by Fyers
    # TBT only works for NSE equity. For NFO, regular depth (typically 5 levels) will be used.
    use_50_level = False  # Set to True only if testing with NSE equity symbols
    instruments = construct_option_instruments(strikes, expiry_date, option_types=["CE", "PE"], use_50_level=use_50_level)
    if not instruments:
        print("❌ No valid instruments found. Exiting.")
        sys.exit(1)
    
    print(f"\n✅ Total instruments to subscribe: {len(instruments)}")
    print(f"ℹ️  Note: NFO options use regular depth (typically 5 levels).")
    print(f"   50-level depth (TBT) is only available for NSE equity symbols.")
    
    # Step 5: Create the feed and connect
    print("\n🔌 Step 5: Connecting to WebSocket server...")
    feed = Nifty50DepthFeed(api_key=api_key)
    
    if not feed.connect():
        print("❌ Failed to connect or authenticate with WebSocket server")
        sys.exit(1)
    
    # Step 6: Subscribe to depth data
    print("\n📡 Step 6: Subscribing to market depth...")
    # Note: For NFO options, 50-level depth (TBT) is not supported by Fyers
    # TBT only supports NSE equity. For NFO options, regular depth (5 levels) will be used.
    # Set use_50_level=True only if you want to test with NSE equity symbols
    use_50_level = False  # Set to True for NSE equity to get 50-level depth via TBT
    feed.subscribe_depth(instruments, use_50_level=use_50_level)
    
    # Step 7: Receive data for specified duration
    print("\n" + "="*60)
    print(f"📊 Receiving market depth data for 30 seconds...")
    print("="*60)
    print("(Press Ctrl+C to stop early)\n")
    
    try:
        for i in range(30):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    # Step 8: Unsubscribe and disconnect
    print("\n🔌 Unsubscribing...")
    feed.unsubscribe_depth(instruments)
    time.sleep(1)
    
    print("\n🔌 Disconnecting...")
    feed.disconnect()
    
    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print("="*60)

