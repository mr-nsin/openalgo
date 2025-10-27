#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIFTY ATM Options Tracker - WebSocket Version (Zerodha-style Option Chain Console, Fixed Layout)

What’s improved
---------------
• Correct, fixed-width columns (handles large comma-separated numbers).
• Pad first, then color — ANSI colors no longer break alignment.
• Accurate header/ATM/separator lengths from computed table width.
• Still handles WebSocket mode 1/2/3 (Full) with robust bid/ask extraction.

Usage:
    python nifty_atm_websocket_tracker.py
"""

import os
import sys
import time
import json
import threading
import websocket
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openalgo import api
from database.symbol import SymToken, db_session, enhanced_search_symbols
from loguru import logger

# =========================
# Configuration
# =========================
API_KEY = os.getenv("OPENALGO_API_KEY", "4729b0f39e8d894b6667b95ab31994b06a8ce1f2fd40e8e4d1f9a74051d7c75f")
API_HOST = os.getenv("OPENALGO_API_HOST", "http://127.0.0.1:5000")
WS_HOST = os.getenv("OPENALGO_WS_HOST", "127.0.0.1")
WS_PORT = int(os.getenv("OPENALGO_WS_PORT", "8765"))

STRIKE_RANGE = 20        # strikes above and below ATM
STRIKE_STEP = 50         # NIFTY strike step
REFRESH_INTERVAL = 0.1   # 100ms between display updates (10 Hz - ultra fast)
NIFTY_EXPIRY = "28OCT25" # Example expiry (adjust as needed)

# Ultra-fast WebSocket configuration
WEBSOCKET_PING_INTERVAL = 10      # 10 seconds between pings
WEBSOCKET_BATCH_SIZE = 50         # Larger batches for faster subscription
WEBSOCKET_BATCH_DELAY = 0.05      # 50ms delay between batches
MAX_WEBSOCKET_SUBSCRIPTIONS = 3000 # Maximum subscriptions per connection

# Use ANSI color in terminals (disable if you want plain text)
USE_COLOR = True

# =========================
# Loguru Configuration
# =========================
# Configure loguru for async logging and performance
logger.remove()  # Remove default handler

# Console handler with colors and performance info
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
    enqueue=True  # Async logging
)

# File handler for detailed logs with enhanced formatting
log_file = f"logs/nifty_atm_websocket_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("logs", exist_ok=True)
logger.add(
    log_file,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    rotation="100 MB",
    retention="7 days",
    enqueue=True,  # Async logging
    colorize=True  # Enable colors in log file
)

# Performance metrics logger
perf_logger = logger.bind(component="performance")
data_logger = logger.bind(component="data_flow")
atm_logger = logger.bind(component="atm_tracking")

# =========================
# Models
# =========================
@dataclass
class OptionInfo:
    symbol: str
    strike: float
    option_type: str  # "CE" or "PE"
    expiry: str
    ltp: float = 0.0
    volume: int = 0
    oi: int = 0
    bid: float = 0.0
    ask: float = 0.0
    change: float = 0.0
    timestamp: int = 0
    last_update: Optional[datetime] = None
    update_count: int = 0  # Track number of updates
    last_price: float = 0.0  # Track price changes
    # Enhanced bid/ask levels
    bid_levels: List[Tuple[float, int]] = None  # [(price, quantity), ...]
    ask_levels: List[Tuple[float, int]] = None  # [(price, quantity), ...]
    
    def __post_init__(self):
        if self.bid_levels is None:
            self.bid_levels = []
        if self.ask_levels is None:
            self.ask_levels = []

# =========================
# WebSocket Feed
# =========================
class WebSocketFeed:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, api_key: str = None):
        self.ws_url = f"ws://{host}:{port}"
        self.api_key = api_key
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.pending_auth = False
        self.message_queue = Queue(maxsize=10000)  # Larger queue for high-frequency data
        self.market_data = {'ltp': {}}
        self.lock = threading.Lock()
        self.on_data_callback = None
        self.ws_thread = None
        
        # Enhanced performance tracking
        self.message_count = 0
        self.start_time = time.time()
        self.last_message_time = 0
        self.messages_per_second = 0.0
        self.last_calculation_time = time.time()
        self.data_processing_times = []
        self.connection_start_time = 0
        self.last_heartbeat = 0
        self.reconnect_count = 0
        self.total_bytes_received = 0
        self.bytes_per_second = 0.0
        
        # ATM tracking metrics
        self.atm_updates = 0
        self.last_atm_price = 0.0
        self.atm_price_changes = []
        self.option_updates = 0
        self.last_option_update_time = 0
        self.atm_strike = 0  # Add missing atm_strike attribute
        
        logger.info(f"WebSocketFeed initialized for {self.ws_url}")

    def connect(self) -> bool:
        try:
            self.connection_start_time = time.time()
            logger.info(f"Attempting to connect to WebSocket at {self.ws_url}")
            
            def on_message(ws, message):
                try:
                    self.message_queue.put_nowait(message)  # Non-blocking for speed
                except:
                    pass  # Drop message if queue is full
                self._process_message(message)

            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
                self.reconnect_count += 1

            def on_open(ws):
                connection_time = time.time() - self.connection_start_time
                logger.success(f"📡 Connected to {self.ws_url} in {connection_time:.3f}s")
                self.connected = True
                self.last_heartbeat = time.time()

            def on_close(ws, close_status_code, close_reason):
                logger.warning(f"📡 Disconnected from {self.ws_url} - Code: {close_status_code}, Reason: {close_reason}")
                self.connected = False
                self.authenticated = False

            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_open=on_open,
                on_close=on_close
            )
            self.ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval": WEBSOCKET_PING_INTERVAL})
            self.ws_thread.daemon = True
            self.ws_thread.start()

            start_time = time.time()
            while not self.connected and time.time() - start_time < 5:
                time.sleep(0.1)
            if not self.connected:
                logger.error("❌ Failed to connect to the WebSocket server")
                return False

            return self._authenticate()
        except Exception as e:
            logger.error(f"❌ Error connecting to WebSocket: {e}")
            return False

    def disconnect(self) -> None:
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
            self.connected = False
            self.authenticated = False

    def _authenticate(self) -> bool:
        if not self.connected or not self.api_key:
            logger.error("❌ Cannot authenticate: not connected or no API key")
            return False

        auth_msg = {"action": "authenticate", "api_key": self.api_key}
        logger.info(f"🔐 Authenticating with API key: {self.api_key[:8]}...{self.api_key[-8:]}")
        self.ws.send(json.dumps(auth_msg))
        self.pending_auth = True

        start_time = time.time()
        while not self.authenticated and time.time() - start_time < 5:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get(block=False)
                    self._process_message(message)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"❌ Error processing message: {e}")
                time.sleep(0.1)

        if self.authenticated:
            auth_time = time.time() - start_time
            logger.success(f"✅ Authentication successful in {auth_time:.3f}s!")
            return True
        logger.error("❌ Authentication failed or timed out")
        return False

    def _extract_bid_ask_levels(self, md: dict) -> Tuple[float, float, List[Tuple[float, int]], List[Tuple[float, int]]]:
        """
        Extract best bid/ask and all 5 levels of depth:
        Returns: (best_bid, best_ask, bid_levels, ask_levels)
        """
        bid = md.get("bid", None)
        ask = md.get("ask", None)
        bid_levels = []
        ask_levels = []

        if bid is None:
            bid = md.get("best_bid", None)
        if ask is None:
            ask = md.get("best_ask", None)

        try:
            depth = md.get("depth") or {}
            bids = depth.get("bids") or depth.get("bid") or []
            asks = depth.get("asks") or depth.get("ask") or []
            
            # Extract bid levels (up to 5)
            if isinstance(bids, list):
                for i, bid_item in enumerate(bids[:5]):
                    try:
                        if isinstance(bid_item, dict):
                            price = float(bid_item.get("price", 0))
                            qty = int(bid_item.get("quantity", bid_item.get("qty", 0)))
                        else:
                            price = float(bid_item[0]) if len(bid_item) > 0 else 0
                            qty = int(bid_item[1]) if len(bid_item) > 1 else 0
                        if price > 0 and qty > 0:
                            bid_levels.append((price, qty))
                    except (ValueError, IndexError, TypeError):
                        continue
                        
            # Extract ask levels (up to 5)
            if isinstance(asks, list):
                for i, ask_item in enumerate(asks[:5]):
                    try:
                        if isinstance(ask_item, dict):
                            price = float(ask_item.get("price", 0))
                            qty = int(ask_item.get("quantity", ask_item.get("qty", 0)))
                        else:
                            price = float(ask_item[0]) if len(ask_item) > 0 else 0
                            qty = int(ask_item[1]) if len(ask_item) > 1 else 0
                        if price > 0 and qty > 0:
                            ask_levels.append((price, qty))
                    except (ValueError, IndexError, TypeError):
                        continue
                        
            # Set best bid/ask from levels if not available directly
            if bid is None and bid_levels:
                bid = bid_levels[0][0]
            if ask is None and ask_levels:
                ask = ask_levels[0][0]
                
        except Exception as e:
            data_logger.debug(f"Error extracting depth levels: {e}")

        try:
            bid = float(bid) if bid is not None else 0.0
        except Exception:
            bid = 0.0
        try:
            ask = float(ask) if ask is not None else 0.0
        except Exception:
            ask = 0.0
            
        return bid, ask, bid_levels, ask_levels

    def _process_message(self, message_str: str) -> None:
        process_start = time.time()
        try:
            # Track message performance
            current_time = time.time()
            self.message_count += 1
            self.last_message_time = current_time
            self.total_bytes_received += len(message_str.encode('utf-8'))
            
            # Calculate performance metrics every 5 seconds
            if current_time - self.last_calculation_time >= 5.0:
                elapsed = current_time - self.start_time
                self.messages_per_second = self.message_count / elapsed if elapsed > 0 else 0
                self.bytes_per_second = self.total_bytes_received / elapsed if elapsed > 0 else 0
                self.last_calculation_time = current_time
                
                # Log performance metrics
                perf_logger.info(f"Performance: {self.messages_per_second:.1f} msg/sec, "
                               f"{self.bytes_per_second/1024:.1f} KB/sec, "
                               f"Total: {self.message_count:,} msgs, "
                               f"Reconnects: {self.reconnect_count}")
            
            message = json.loads(message_str)

            if message.get("type") == "auth":
                if message.get("status") == "success":
                    logger.success(f"✅ Authentication response: {message.get('message', 'Success')}")
                    self.authenticated = True
                    self.pending_auth = False
                else:
                    logger.error(f"❌ Authentication failed: {message}")
                    self.pending_auth = False
                return

            if message.get("type") == "subscribe":
                if message.get("status") == "success":
                    subs = message.get('subscriptions', [])
                    success_count = len([s for s in subs if s.get('status') == 'success'])
                    logger.success(f"✅ Subscriptions: {success_count}/{len(subs)} successful")
                else:
                    logger.error(f"❌ Subscription failed: {message}")
                return

            if message.get("type") == "market_data":
                exchange = message.get("exchange")
                symbol = message.get("symbol")
                if exchange and symbol:
                    if exchange == "INDEX" and message.get("broker"):
                        exchange = "NSE_INDEX"

                    mode = message.get("mode")
                    md = message.get("data", {})
                    
                    # Accept modes 1 (LTP), 2 (Quote), and 3 (Full)
                    if mode in [1, 2, 3] and "ltp" in md:
                        ltp = md.get("ltp", 0.0)
                        ts = md.get("timestamp", int(time.time() * 1000))
                        vol = md.get("volume", 0)
                        oi = md.get("oi", 0)
                        bid, ask, bid_levels, ask_levels = self._extract_bid_ask_levels(md)

                        with self.lock:
                            if exchange not in self.market_data['ltp']:
                                self.market_data['ltp'][exchange] = {}
                            rec = self.market_data['ltp'][exchange].get(symbol, {})
                            rec.update({
                                'timestamp': ts,
                                'ltp': ltp,
                                'volume': vol,
                                'oi': oi,
                                'bid': bid if bid else rec.get('bid', 0.0),
                                'ask': ask if ask else rec.get('ask', 0.0),
                                'change': md.get('change', rec.get('change', 0.0)),
                                'open': md.get('open', rec.get('open', 0.0)),
                                'high': md.get('high', rec.get('high', 0.0)),
                                'low': md.get('low', rec.get('low', 0.0)),
                                'mode': mode,
                                'bid_levels': bid_levels,
                                'ask_levels': ask_levels
                            })
                            self.market_data['ltp'][exchange][symbol] = rec

                        # Enhanced NIFTY logging with performance metrics
                        if symbol == "NIFTY":
                            latency = (time.time() * 1000) - ts
                            self.atm_updates += 1
                            price_change = ltp - self.last_atm_price if self.last_atm_price > 0 else 0
                            self.atm_price_changes.append(price_change)
                            if len(self.atm_price_changes) > 100:  # Keep only last 100 changes
                                self.atm_price_changes.pop(0)
                            
                            atm_logger.info(f"📈 NIFTY: ₹{ltp:.2f} | Vol: {vol:,} | Latency: {latency:.1f}ms | "
                                          f"Msg/sec: {self.messages_per_second:.1f} | "
                                          f"Updates: {self.atm_updates:,} | "
                                          f"Change: {price_change:+.2f} | "
                                          f"Time: {datetime.fromtimestamp(ts/1000).strftime('%H:%M:%S.%f')[:-3]} | "
                                          f"ATM: {self.atm_strike} | "
                                          f"Processing: {time.time() - current_time:.3f}s")
                            
                            self.last_atm_price = ltp
                            
                        # Track option updates
                        if exchange == "NFO":
                            self.option_updates += 1
                            self.last_option_update_time = current_time
                            
                        if self.on_data_callback:
                            md_enriched = dict(md)
                            md_enriched["bid"], md_enriched["ask"] = bid, ask
                            md_enriched["bid_levels"] = bid_levels
                            md_enriched["ask_levels"] = ask_levels
                            self.on_data_callback(f"{exchange}:{symbol}", md_enriched)
                            
            # Track processing time
            process_time = time.time() - process_start
            self.data_processing_times.append(process_time)
            if len(self.data_processing_times) > 1000:  # Keep only last 1000 processing times
                self.data_processing_times.pop(0)
                
        except json.JSONDecodeError as e:
            data_logger.debug(f"Invalid JSON received: {e}")
        except Exception as e:
            data_logger.error(f"Error processing message: {e}")

    def subscribe_ltp(self, instruments: List[Dict[str, str]], on_data_received=None) -> bool:
        if not self.connected or not self.authenticated:
            logger.error("❌ Not connected or authenticated")
            return False

        if len(instruments) > MAX_WEBSOCKET_SUBSCRIPTIONS:
            logger.warning(f"⚠️  Too many instruments ({len(instruments)}) for WebSocket (max: {MAX_WEBSOCKET_SUBSCRIPTIONS})")
            return False

        self.on_data_callback = on_data_received
        batch_size = WEBSOCKET_BATCH_SIZE
        
        logger.info(f"📡 Subscribing to {len(instruments)} instruments in batches of {batch_size}...")
        start_time = time.time()
        
        for i in range(0, len(instruments), batch_size):
            batch = instruments[i:i + batch_size]
            req = {"action": "subscribe", "symbols": batch, "mode": 3}  # LTP mode for speed
            logger.debug(f"📡 Batch {i//batch_size + 1}/{(len(instruments)-1)//batch_size + 1}: {len(batch)} instruments")
            self.ws.send(json.dumps(req))
            time.sleep(WEBSOCKET_BATCH_DELAY)  # Minimal delay between batches
        
        elapsed = time.time() - start_time
        logger.success(f"✅ Subscription completed in {elapsed:.2f}s")
        return True

    def get_ltp_data(self) -> Dict:
        with self.lock:
            return dict(self.market_data)

# =========================
# Tracker
# =========================
class NiftyATMWebSocketTracker:
    def __init__(self):
        self.client = api(api_key=API_KEY, host=API_HOST)
        self.ws_feed = WebSocketFeed(host=WS_HOST, port=WS_PORT, api_key=API_KEY)
        self.nifty_price = 0.0
        self.atm_strike = 0
        self.options: Dict[str, OptionInfo] = {}
        self.current_expiry = ""
        
        # Enhanced ATM tracking
        self.atm_history = []
        self.atm_change_count = 0
        self.last_atm_change_time = 0
        self.option_update_stats = {}
        self.display_refresh_count = 0
        self.last_display_time = 0
        
        logger.info("🚀 Initializing NIFTY ATM WebSocket Tracker...")
        atm_logger.info(f"Configuration: Strike Range: ±{STRIKE_RANGE}, Step: {STRIKE_STEP}, "
                       f"Refresh: {1/REFRESH_INTERVAL:.1f}Hz, Expiry: {NIFTY_EXPIRY}")

    # -------- Color/format helpers --------
    def _clr(self, text, color=None, bold=False):
        if not USE_COLOR or not sys.stdout.isatty():
            return text
        colors = {
            "red": "\033[31m", "green": "\033[32m",
            "yellow": "\033[33m", "cyan": "\033[36m",
            "magenta": "\033[35m", "blue": "\033[34m",
            "grey": "\033[90m"
        }
        reset = "\033[0m"
        b = "\033[1m" if bold else ""
        c = colors.get(color, "")
        return f"{b}{c}{text}{reset}"

    def _fmt_num_plain(self, n, width, decimals=2, comma=True, prefix=""):
        try:
            if isinstance(n, (int, float)):
                s = f"{n:,.{decimals}f}" if comma else f"{n:.{decimals}f}"
            else:
                s = str(n)
        except Exception:
            s = "-"
        if prefix and s not in ("-", ""):
            s = prefix + s
        return f"{s:>{width}}"

    def _fmt_int_plain(self, n, width):
        try:
            s = f"{int(n):,}"
        except Exception:
            s = "-"
        return f"{s:>{width}}"

    def _fmt_chg(self, chg: float, width: int) -> str:
        """Pad first, then color entire padded string so visible width stays constant."""
        s = f"{chg:+.2f}"
        padded = f"{s:>{width}}"
        if chg > 0:
            return self._clr(padded, "green")
        if chg < 0:
            return self._clr(padded, "red")
        return padded

    def _fmt_bid_ask_levels(self, levels: List[Tuple[float, int]], width: int) -> str:
        """Format bid/ask levels for display (up to 5 levels)"""
        if not levels:
            return f"{'-':>{width}}"
        
        # Format up to 5 levels more compactly
        level_strs = []
        for i, (price, qty) in enumerate(levels[:5]):
            if qty > 0:  # Only show levels with quantity
                level_strs.append(f"{price:.1f}@{qty}")
        
        if not level_strs:
            return f"{'-':>{width}}"
        
        # Join with spaces and pad
        result = " ".join(level_strs)
        return f"{result:>{width}}"

    # -------- Data plumbing --------
    def get_current_nifty_price(self) -> float:
        try:
            logger.info("📈 Fetching initial NIFTY price from API...")
            start_time = time.time()
            resp = self.client.quotes(symbol="NIFTY", exchange="NSE_INDEX")
            api_time = time.time() - start_time
            
            if resp.get("status") == "success":
                price = float(resp["data"]["ltp"])
                logger.success(f"✅ Initial NIFTY Price: ₹{price:.2f} (API time: {api_time:.3f}s)")
                atm_logger.info(f"Initial NIFTY: ₹{price:.2f} | API Response Time: {api_time:.3f}s")
                return price
            logger.error(f"❌ Error fetching NIFTY price: {resp.get('message', 'Unknown error')}")
            return 0.0
        except Exception as e:
            logger.error(f"❌ Exception getting NIFTY price: {e}")
            return 0.0

    def get_nifty_price_from_websocket(self) -> float:
        try:
            ws_data = self.ws_feed.get_ltp_data()
            nse_idx = ws_data.get('ltp', {}).get('NSE_INDEX', {})
            nifty = nse_idx.get('NIFTY', {})
            if nifty and 'ltp' in nifty:
                return float(nifty['ltp'])
            return self.nifty_price
        except Exception:
            return self.nifty_price

    def calculate_atm_strike(self, nifty_price: float) -> int:
        atm = int(round(nifty_price / STRIKE_STEP) * STRIKE_STEP)
        old_atm = self.atm_strike
        
        if old_atm != 0 and atm != old_atm:
            self.atm_change_count += 1
            self.last_atm_change_time = time.time()
            atm_logger.info(f"🎯 ATM Strike changed: {old_atm} → {atm} (NIFTY: ₹{nifty_price:.2f})")
            self.atm_history.append({
                'timestamp': time.time(),
                'nifty_price': nifty_price,
                'old_atm': old_atm,
                'new_atm': atm,
                'change': atm - old_atm
            })
            # Keep only last 50 ATM changes
            if len(self.atm_history) > 50:
                self.atm_history.pop(0)
        else:
            atm_logger.debug(f"🎯 ATM Strike: {atm} (NIFTY: ₹{nifty_price:.2f})")
            
        return atm

    def get_strike_range(self) -> List[int]:
        strikes = []
        for i in range(-STRIKE_RANGE, STRIKE_RANGE + 1):
            s = self.atm_strike + i * STRIKE_STEP
            if s > 0:
                strikes.append(s)
        print(f"📊 Generated {len(strikes)} strikes: {strikes[0]} to {strikes[-1]}")
        return strikes

    def discover_current_expiry(self) -> str:
        print(f"📅 Using NIFTY expiry: {NIFTY_EXPIRY}")
        return NIFTY_EXPIRY

    def get_actual_strikes_from_database(self) -> List[float]:
        try:
            print(f"🔍 Getting actual NIFTY strikes from database for expiry: {self.current_expiry}")
            pattern = f"NIFTY{self.current_expiry}"
            results = enhanced_search_symbols(pattern, exchange="NFO")
            if not results:
                print(f"❌ No options found for pattern {pattern}")
                return []
            nifty = []
            for r in results:
                if (r.symbol.startswith(f"NIFTY{self.current_expiry}")
                    and not r.symbol.startswith(f"FINNIFTY{self.current_expiry}")
                    and not r.symbol.startswith(f"BANKNIFTY{self.current_expiry}")
                    and not r.symbol.startswith(f"MIDCPNIFTY{self.current_expiry}")
                    and r.instrumenttype in ["CE", "PE"]):
                    nifty.append(r)
            if not nifty:
                print(f"❌ No NIFTY options found (from {len(results)} results)")
                return []
            strikes = sorted({r.strike for r in nifty if r.strike})
            print(f"✅ Found {len(strikes)} unique NIFTY strikes in database")
            print(f"📊 NIFTY strike range: {min(strikes)} to {max(strikes)}")
            if len(strikes) > 1:
                step = strikes[1] - strikes[0]
                print(f"📏 NIFTY strike step: {step} points")
            return strikes
        except Exception as e:
            print(f"❌ Error getting strikes from database: {e}")
            logger.exception("Error in get_actual_strikes_from_database")
            return []

    def get_strikes_around_atm(self, all_strikes: List[float]) -> List[float]:
        if not all_strikes:
            return []
        atm = min(all_strikes, key=lambda x: abs(x - self.nifty_price))
        idx = all_strikes.index(atm)
        start = max(0, idx - STRIKE_RANGE)
        end = min(len(all_strikes), idx + STRIKE_RANGE + 1)
        sel = all_strikes[start:end]
        print(f"🎯 ATM Strike from database: {atm}")
        print(f"📊 Selected {len(sel)} strikes around ATM: {sel[0]} to {sel[-1]}")
        self.atm_strike = int(atm)
        return sel

    def discover_option_symbols_from_database(self, strikes: List[float]) -> Dict[str, OptionInfo]:
        options = {}
        found = 0
        try:
            pattern = f"NIFTY{self.current_expiry}"
            all_results = enhanced_search_symbols(pattern, exchange="NFO")
            nifty = []
            for r in all_results:
                if (r.symbol.startswith(f"NIFTY{self.current_expiry}")
                    and not r.symbol.startswith(f"FINNIFTY{self.current_expiry}")
                    and not r.symbol.startswith(f"BANKNIFTY{self.current_expiry}")
                    and not r.symbol.startswith(f"MIDCPNIFTY{self.current_expiry}")
                    and r.instrumenttype in ["CE", "PE"]):
                    nifty.append(r)
            if not nifty:
                print(f"❌ No NIFTY options found (from {len(all_results)} results)")
                return {}

            strike_map = {}
            for r in nifty:
                if r.strike and r.instrumenttype in ["CE", "PE"]:
                    strike_map[f"{r.strike}_{r.instrumenttype}"] = r

            for s in strikes:
                for t in ["CE", "PE"]:
                    k = f"{s}_{t}"
                    if k in strike_map:
                        r = strike_map[k]
                        options[k] = OptionInfo(
                            symbol=r.symbol, strike=s, option_type=t,
                            expiry=r.expiry or self.current_expiry
                        )
                        found += 1
                    else:
                        print(f"⚠️  Not found in database: NIFTY {s} {t}")

            print(f"✅ Found {found} NIFTY option symbols out of {len(strikes)*2} expected")
            return options
        except Exception as e:
            print(f"❌ Error discovering option symbols: {e}")
            logger.exception("Error in discover_option_symbols_from_database")
            return {}

    # -------- WebSocket callbacks / subs --------
    def on_websocket_data(self, symbol_key: str, md: dict):
        try:
            if ":" in symbol_key:
                exchange, symbol = symbol_key.split(":", 1)
            else:
                exchange, symbol = "NFO", symbol_key

            # Index updates
            if symbol == "NIFTY" and exchange == "NSE_INDEX":
                new_px = float(md.get("ltp", 0))
                if new_px > 0 and abs(new_px - self.nifty_price) > 0.01:
                    old = self.nifty_price
                    self.nifty_price = new_px
                    new_atm = self.calculate_atm_strike(self.nifty_price)
                    if new_atm != self.atm_strike:
                        self.atm_strike = new_atm
                return

            # Option updates with enhanced tracking
            if exchange == "NFO":
                for opt in self.options.values():
                    if opt.symbol == symbol:
                        old_price = opt.ltp
                        opt.ltp = float(md.get("ltp", opt.ltp))
                        opt.volume = int(md.get("volume", opt.volume))
                        opt.oi = int(md.get("oi", opt.oi))
                        opt.bid = float(md.get("bid", opt.bid))
                        opt.ask = float(md.get("ask", opt.ask))
                        opt.change = float(md.get("change", opt.change))
                        opt.timestamp = int(md.get("timestamp", int(time.time() * 1000)))
                        opt.last_update = datetime.now()
                        opt.update_count += 1
                        opt.last_price = old_price
                        
                        # Update bid/ask levels
                        opt.bid_levels = md.get("bid_levels", [])
                        opt.ask_levels = md.get("ask_levels", [])
                        
                        # Track option update statistics
                        opt_key = f"{opt.strike}_{opt.option_type}"
                        if opt_key not in self.option_update_stats:
                            self.option_update_stats[opt_key] = {
                                'updates': 0,
                                'last_update': 0,
                                'price_changes': []
                            }
                        
                        self.option_update_stats[opt_key]['updates'] += 1
                        self.option_update_stats[opt_key]['last_update'] = time.time()
                        
                        if old_price > 0 and abs(opt.ltp - old_price) > 0.01:
                            price_change = opt.ltp - old_price
                            self.option_update_stats[opt_key]['price_changes'].append(price_change)
                            if len(self.option_update_stats[opt_key]['price_changes']) > 20:
                                self.option_update_stats[opt_key]['price_changes'].pop(0)
                        
                        # Log significant option updates with enhanced formatting
                        if opt.strike == self.atm_strike and abs(opt.ltp - old_price) > 0.1:
                            # Format bid/ask levels for logging
                            bid_levels_str = " | ".join([f"{p:.1f}@{q}" for p, q in opt.bid_levels[:3] if q > 0])
                            ask_levels_str = " | ".join([f"{p:.1f}@{q}" for p, q in opt.ask_levels[:3] if q > 0])
                            
                            atm_logger.info(f"🎯 ATM {opt.option_type} {opt.strike}: ₹{old_price:.2f} → ₹{opt.ltp:.2f} | "
                                          f"Bid: {opt.bid:.2f} | Ask: {opt.ask:.2f} | "
                                          f"BidLevels: {bid_levels_str or 'N/A'} | "
                                          f"AskLevels: {ask_levels_str or 'N/A'} | "
                                          f"Vol: {opt.volume:,} | OI: {opt.oi:,}")
                        
                        break
        except Exception as e:
            data_logger.error(f"Error processing WebSocket data for {symbol_key}: {e}")

    def setup_websocket_subscriptions(self):
        instruments = [{"exchange": "NSE_INDEX", "symbol": "NIFTY"}]
        for opt in self.options.values():
            instruments.append({"exchange": "NFO", "symbol": opt.symbol})
        print(f"📡 Setting up WebSocket subscriptions for {len(instruments)} instruments...")
        ok = self.ws_feed.subscribe_ltp(instruments, on_data_received=self.on_websocket_data)
        print("✅ WebSocket subscriptions setup complete" if ok else "❌ Failed to setup subscriptions")
        return ok

    def validate_nifty_data_flow(self) -> bool:
        print("🔍 Validating NIFTY data flow...")
        start = time.time()
        while time.time() - start < 10:
            px = self.get_nifty_price_from_websocket()
            if px > 0 and abs(px - self.nifty_price) < 1000:
                print(f"✅ NIFTY WebSocket data validated: ₹{px:.2f}")
                self.nifty_price = px
                return True
            print(f"⏳ Waiting for NIFTY data... ({int(time.time() - start)}s)")
            time.sleep(1)
        print("⚠️  NIFTY WebSocket data not received, using API price")
        return False

    def classify_option(self, option: OptionInfo) -> str:
        if option.strike == self.atm_strike:
            return "ATM"
        if option.option_type == "CE":
            return "ITM" if option.strike < self.nifty_price else "OTM"
        return "ITM" if option.strike > self.nifty_price else "OTM"

    # -------- Zerodha-style Chain Renderer --------
    def display_option_chain(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Track display performance
        current_time = time.time()
        self.display_refresh_count += 1
        if self.last_display_time > 0:
            display_interval = current_time - self.last_display_time
            actual_fps = 1.0 / display_interval if display_interval > 0 else 0
        else:
            actual_fps = 0
        self.last_display_time = current_time

        # Column plan (wider OI/VOL to fit commas comfortably)
        colw = {
            "oi": 12, "vol": 11, "ltp": 8, "chg": 8, "bid": 8, "ask": 8, "strike": 7,
            "bid1": 8, "bid2": 8, "bid3": 8, "bid4": 8, "bid5": 8,
            "ask1": 8, "ask2": 8, "ask3": 8, "ask4": 8, "ask5": 8,
            "timestamp": 12  # For individual bid/ask levels and timestamp
        }

        def hdr_side(side):
            if side == "CALLS":
                return f"{'OI':>{colw['oi']}} {'VOL':>{colw['vol']}} {'LTP':>{colw['ltp']}} {'CHG':>{colw['chg']}} {'BID1':>{colw['bid1']}} {'BID2':>{colw['bid2']}} {'BID3':>{colw['bid3']}} {'BID4':>{colw['bid4']}} {'BID5':>{colw['bid5']}} {'ASK1':>{colw['ask1']}} {'ASK2':>{colw['ask2']}} {'ASK3':>{colw['ask3']}} {'ASK4':>{colw['ask4']}} {'ASK5':>{colw['ask5']}}"
            return f"{'BID1':>{colw['bid1']}} {'BID2':>{colw['bid2']}} {'BID3':>{colw['bid3']}} {'BID4':>{colw['bid4']}} {'BID5':>{colw['bid5']}} {'ASK1':>{colw['ask1']}} {'ASK2':>{colw['ask2']}} {'ASK3':>{colw['ask3']}} {'ASK4':>{colw['ask4']}} {'ASK5':>{colw['ask5']}} {'CHG':>{colw['chg']}} {'LTP':>{colw['ltp']}} {'VOL':>{colw['vol']}} {'OI':>{colw['oi']}}"

        calls_hdr = hdr_side("CALLS")
        puts_hdr  = hdr_side("PUTS")

        # Compute total table width for separators (no ANSI here)
        sep_gap = " | "
        row_sep_gap = " | "
        table_width = len(calls_hdr) + len(sep_gap) + colw['strike'] + len(sep_gap) + colw['timestamp'] + len(sep_gap) + len(puts_hdr)

        now = datetime.now().strftime('%H:%M:%S')
        
        # Enhanced header with performance metrics
        ws_data = self.ws_feed.get_ltp_data()
        msg_rate = getattr(self.ws_feed, 'messages_per_second', 0.0)
        bytes_rate = getattr(self.ws_feed, 'bytes_per_second', 0.0)
        atm_changes = len(self.atm_history)
        
        header = f"🧿 NIFTY OPTION CHAIN (WebSocket)  |  NIFTY: ₹{self.nifty_price:.2f}  |  ATM: {self.atm_strike}  |  Expiry: {self.current_expiry}  |  ⏰ {now}"
        perf_line = f"📊 Performance: {actual_fps:.1f} FPS | {msg_rate:.1f} msg/sec | {bytes_rate/1024:.1f} KB/sec | ATM Changes: {atm_changes} | Updates: {self.display_refresh_count:,}"
        
        print("=" * max(len(header), table_width))
        print(header)
        print(perf_line)
        print("=" * max(len(header), table_width))

        print(calls_hdr + sep_gap + f"{'STRK':^{colw['strike']}}" + sep_gap + f"{'TIME':^{colw['timestamp']}}" + sep_gap + puts_hdr)
        print("-" * (table_width + colw['timestamp'] + len(sep_gap)))

        # Build a map strike -> {CE, PE}
        by_strike: Dict[float, Dict[str, OptionInfo]] = {}
        for k, v in self.options.items():
            by_strike.setdefault(v.strike, {})[v.option_type] = v

        strikes_sorted = sorted(by_strike.keys())

        tot_call_oi = 0
        tot_put_oi = 0
        atm_line_plain = "━" * (table_width + colw['timestamp'] + len(sep_gap))
        atm_line_colored = self._clr(atm_line_plain, "yellow")

        for s in strikes_sorted:
            ce = by_strike[s].get("CE")
            pe = by_strike[s].get("PE")

            def fmt_side(opt: Optional[OptionInfo], left=True):
                if not opt or not opt.last_update:
                    if left:
                        return " ".join([
                            f"{'-':>{colw['oi']}}", f"{'-':>{colw['vol']}}",
                            f"{'-':>{colw['ltp']}}", f"{'-':>{colw['chg']}}",
                            f"{'-':>{colw['bid1']}}", f"{'-':>{colw['bid2']}}", f"{'-':>{colw['bid3']}}", f"{'-':>{colw['bid4']}}", f"{'-':>{colw['bid5']}}",
                            f"{'-':>{colw['ask1']}}", f"{'-':>{colw['ask2']}}", f"{'-':>{colw['ask3']}}", f"{'-':>{colw['ask4']}}", f"{'-':>{colw['ask5']}}"
                        ])
                    else:
                        return " ".join([
                            f"{'-':>{colw['bid1']}}", f"{'-':>{colw['bid2']}}", f"{'-':>{colw['bid3']}}", f"{'-':>{colw['bid4']}}", f"{'-':>{colw['bid5']}}",
                            f"{'-':>{colw['ask1']}}", f"{'-':>{colw['ask2']}}", f"{'-':>{colw['ask3']}}", f"{'-':>{colw['ask4']}}", f"{'-':>{colw['ask5']}}",
                            f"{'-':>{colw['chg']}}", f"{'-':>{colw['ltp']}}",
                            f"{'-':>{colw['vol']}}", f"{'-':>{colw['oi']}}"
                        ])

                chg_padded = self._fmt_chg(opt.change, colw['chg'])
                
                # Format individual bid/ask levels
                bid_levels = opt.bid_levels[:5] if opt.bid_levels else []
                ask_levels = opt.ask_levels[:5] if opt.ask_levels else []
                
                bid_cols = []
                for i in range(5):
                    if i < len(bid_levels) and bid_levels[i][1] > 0:  # Check quantity > 0
                        bid_cols.append(f"{bid_levels[i][0]:.1f}@{bid_levels[i][1]}")
                    else:
                        bid_cols.append("-")
                
                ask_cols = []
                for i in range(5):
                    if i < len(ask_levels) and ask_levels[i][1] > 0:  # Check quantity > 0
                        ask_cols.append(f"{ask_levels[i][0]:.1f}@{ask_levels[i][1]}")
                    else:
                        ask_cols.append("-")

                if left:
                    return " ".join([
                        self._fmt_int_plain(opt.oi, colw['oi']),
                        self._fmt_int_plain(opt.volume, colw['vol']),
                        self._fmt_num_plain(opt.ltp, colw['ltp']),
                        chg_padded,
                        f"{bid_cols[0]:>{colw['bid1']}}", f"{bid_cols[1]:>{colw['bid2']}}", f"{bid_cols[2]:>{colw['bid3']}}", f"{bid_cols[3]:>{colw['bid4']}}", f"{bid_cols[4]:>{colw['bid5']}}",
                        f"{ask_cols[0]:>{colw['ask1']}}", f"{ask_cols[1]:>{colw['ask2']}}", f"{ask_cols[2]:>{colw['ask3']}}", f"{ask_cols[3]:>{colw['ask4']}}", f"{ask_cols[4]:>{colw['ask5']}}"
                    ])
                else:
                    return " ".join([
                        f"{bid_cols[0]:>{colw['bid1']}}", f"{bid_cols[1]:>{colw['bid2']}}", f"{bid_cols[2]:>{colw['bid3']}}", f"{bid_cols[3]:>{colw['bid4']}}", f"{bid_cols[4]:>{colw['bid5']}}",
                        f"{ask_cols[0]:>{colw['ask1']}}", f"{ask_cols[1]:>{colw['ask2']}}", f"{ask_cols[2]:>{colw['ask3']}}", f"{ask_cols[3]:>{colw['ask4']}}", f"{ask_cols[4]:>{colw['ask5']}}",
                        chg_padded,
                        self._fmt_num_plain(opt.ltp, colw['ltp']),
                        self._fmt_int_plain(opt.volume, colw['vol']),
                        self._fmt_int_plain(opt.oi, colw['oi']),
                    ])

            left = fmt_side(ce, left=True)
            right = fmt_side(pe, left=False)

            # ATM highlight & strike
            strike_txt_plain = f"{int(s):^{colw['strike']}}"
            strike_txt = strike_txt_plain
            if s == self.atm_strike:
                print(atm_line_colored)
                strike_txt = self._clr(strike_txt_plain, "yellow", bold=True)

            # Format timestamp
            timestamp_str = "-"
            if ce and ce.last_update:
                timestamp_str = ce.last_update.strftime('%H:%M:%S')
            elif pe and pe.last_update:
                timestamp_str = pe.last_update.strftime('%H:%M:%S')
            
            timestamp_txt = f"{timestamp_str:^{colw['timestamp']}}"

            print(left + row_sep_gap + strike_txt + row_sep_gap + timestamp_txt + row_sep_gap + right)

            if s == self.atm_strike:
                print(atm_line_colored)

            if ce and ce.last_update:
                tot_call_oi += ce.oi
            if pe and pe.last_update:
                tot_put_oi += pe.oi

        # Footer totals with performance metrics
        pcr = (tot_put_oi / tot_call_oi) if tot_call_oi else 0.0
        pcr_txt = f"{pcr:.2f}"
        pcr_txt = self._clr(pcr_txt, "green") if pcr >= 1 else self._clr(pcr_txt, "red")
        
        # Calculate performance metrics
        total_updates = sum(opt.update_count for opt in self.options.values())
        avg_updates_per_option = total_updates / len(self.options) if self.options else 0
        
        print("-" * (table_width + colw['timestamp'] + len(sep_gap)))
        print(
            f"{'Totals  CALL OI: '}{tot_call_oi:,}    {'PUT OI: '}{tot_put_oi:,}    {'PCR: '}{pcr_txt}"
        )
        print("=" * (table_width + colw['timestamp'] + len(sep_gap)))
        
        # Enhanced performance metrics
        print(f"📊 PERFORMANCE: Updates/sec: {msg_rate:.1f} | Total Updates: {total_updates:,} | "
              f"Avg/option: {avg_updates_per_option:.1f} | Refresh: {1/REFRESH_INTERVAL:.1f}Hz")
        
        print("📡 WebSocket Connected | Ctrl+C to stop")
        
        # Log the option chain display to file
        self._log_option_chain_display()

    def _log_option_chain_display(self):
        """Log the option chain display similar to console format"""
        try:
            # Build option chain data for logging
            by_strike: Dict[float, Dict[str, OptionInfo]] = {}
            for k, v in self.options.items():
                by_strike.setdefault(v.strike, {})[v.option_type] = v

            strikes_sorted = sorted(by_strike.keys())
            
            # Log header
            now = datetime.now().strftime('%H:%M:%S')
            atm_logger.info("=" * 80)
            atm_logger.info(f"🧿 NIFTY OPTION CHAIN (WebSocket) | NIFTY: ₹{self.nifty_price:.2f} | ATM: {self.atm_strike} | Expiry: {self.current_expiry} | ⏰ {now}")
            
            # Log ATM options specifically
            atm_ce = by_strike.get(self.atm_strike, {}).get("CE")
            atm_pe = by_strike.get(self.atm_strike, {}).get("PE")
            
            if atm_ce and atm_ce.last_update:
                bid_levels_str = " | ".join([f"{p:.1f}@{q}" for p, q in atm_ce.bid_levels[:5] if q > 0])
                ask_levels_str = " | ".join([f"{p:.1f}@{q}" for p, q in atm_ce.ask_levels[:5] if q > 0])
                atm_logger.info(f"📞 ATM CALL {atm_ce.strike}: LTP={atm_ce.ltp:.2f} | CHG={atm_ce.change:+.2f} | "
                              f"Bid={atm_ce.bid:.2f} | Ask={atm_ce.ask:.2f} | "
                              f"BidLevels=[{bid_levels_str or 'N/A'}] | AskLevels=[{ask_levels_str or 'N/A'}] | "
                              f"Vol={atm_ce.volume:,} | OI={atm_ce.oi:,}")
            
            if atm_pe and atm_pe.last_update:
                bid_levels_str = " | ".join([f"{p:.1f}@{q}" for p, q in atm_pe.bid_levels[:5] if q > 0])
                ask_levels_str = " | ".join([f"{p:.1f}@{q}" for p, q in atm_pe.ask_levels[:5] if q > 0])
                atm_logger.info(f"📞 ATM PUT {atm_pe.strike}: LTP={atm_pe.ltp:.2f} | CHG={atm_pe.change:+.2f} | "
                              f"Bid={atm_pe.bid:.2f} | Ask={atm_pe.ask:.2f} | "
                              f"BidLevels=[{bid_levels_str or 'N/A'}] | AskLevels=[{ask_levels_str or 'N/A'}] | "
                              f"Vol={atm_pe.volume:,} | OI={atm_pe.oi:,}")
            
            atm_logger.info("=" * 80)
        except Exception as e:
            atm_logger.error(f"Error logging option chain display: {e}")

    def _display_atm_bid_ask_levels(self):
        """Display detailed bid/ask levels for ATM options"""
        atm_ce = None
        atm_pe = None
        
        # Find ATM options
        for opt in self.options.values():
            if opt.strike == self.atm_strike:
                if opt.option_type == "CE":
                    atm_ce = opt
                elif opt.option_type == "PE":
                    atm_pe = opt
        
        if atm_ce or atm_pe:
            print("\n" + "=" * 80)
            print(f"🎯 ATM {self.atm_strike} - DETAILED BID/ASK LEVELS")
            print("=" * 80)
            
            if atm_ce and atm_ce.last_update:
                print(f"\n📞 CALL {atm_ce.strike} (LTP: ₹{atm_ce.ltp:.2f})")
                print("BID LEVELS:")
                for i, (price, qty) in enumerate(atm_ce.bid_levels[:5], 1):
                    print(f"  {i}. ₹{price:.2f} @ {qty:,}")
                print("ASK LEVELS:")
                for i, (price, qty) in enumerate(atm_ce.ask_levels[:5], 1):
                    print(f"  {i}. ₹{price:.2f} @ {qty:,}")
            
            if atm_pe and atm_pe.last_update:
                print(f"\n📞 PUT {atm_pe.strike} (LTP: ₹{atm_pe.ltp:.2f})")
                print("BID LEVELS:")
                for i, (price, qty) in enumerate(atm_pe.bid_levels[:5], 1):
                    print(f"  {i}. ₹{price:.2f} @ {qty:,}")
                print("ASK LEVELS:")
                for i, (price, qty) in enumerate(atm_pe.ask_levels[:5], 1):
                    print(f"  {i}. ₹{price:.2f} @ {qty:,}")
            
            print("=" * 80)

    # -------- Run loop --------
    def run_websocket_monitoring(self):
        print("\n" + "=" * 80)
        print("📡 WEBSOCKET CONTINUOUS MONITORING")
        print("=" * 80)
        try:
            if not self.ws_feed.connect():
                print("❌ Failed to connect to WebSocket")
                return
            if not self.setup_websocket_subscriptions():
                print("❌ Failed to setup subscriptions")
                return

            if self.validate_nifty_data_flow():
                print("✅ NIFTY data flow validated - starting live monitoring")
            else:
                print("⚠️  NIFTY data flow validation failed - continuing with API price")

            time.sleep(1)  # Reduced initial wait
            print(f"🚀 Starting ultra-fast monitoring at {1/REFRESH_INTERVAL:.1f}Hz refresh rate...")
            
            while True:
                self.display_option_chain()
                time.sleep(REFRESH_INTERVAL)
        except KeyboardInterrupt:
            print("\n⏹️ WebSocket monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error in WebSocket monitoring: {e}")
        finally:
            self.ws_feed.disconnect()

# =========================
# Main
# =========================
def main():
    print("=" * 80)
    print("🚀 NIFTY ATM OPTIONS TRACKER - ULTRA FAST WEBSOCKET VERSION")
    print("=" * 80)
    print(f"⚡ ULTRA FAST CONFIGURATION:")
    print(f"• Display Refresh: {1/REFRESH_INTERVAL:.1f}Hz ({REFRESH_INTERVAL*1000:.0f}ms)")
    print(f"• WebSocket Batch Size: {WEBSOCKET_BATCH_SIZE} instruments")
    print(f"• WebSocket Batch Delay: {WEBSOCKET_BATCH_DELAY*1000:.0f}ms")
    print(f"• Max Subscriptions: {MAX_WEBSOCKET_SUBSCRIPTIONS}")
    print(f"• Ping Interval: {WEBSOCKET_PING_INTERVAL}s")
    print(f"• Log File: {log_file}")
    print("=" * 80)
    
    logger.info("Starting NIFTY ATM WebSocket Tracker")
    logger.info(f"Configuration: Refresh={1/REFRESH_INTERVAL:.1f}Hz, Batch={WEBSOCKET_BATCH_SIZE}, "
               f"Delay={WEBSOCKET_BATCH_DELAY*1000:.0f}ms, MaxSubs={MAX_WEBSOCKET_SUBSCRIPTIONS}")
    try:
        tracker = NiftyATMWebSocketTracker()

        tracker.nifty_price = tracker.get_current_nifty_price()
        if tracker.nifty_price == 0:
            print("❌ Failed to get NIFTY price. Exiting.")
            return

        tracker.atm_strike = tracker.calculate_atm_strike(tracker.nifty_price)
        tracker.current_expiry = tracker.discover_current_expiry()

        print("\n" + "=" * 80)
        print("🔍 GETTING ACTUAL STRIKES FROM DATABASE")
        print("=" * 80)
        all_strikes = tracker.get_actual_strikes_from_database()
        if not all_strikes:
            print("❌ No strikes found in database. Exiting.")
            return

        selected = tracker.get_strikes_around_atm(all_strikes)
        if not selected:
            print("❌ No strikes selected around ATM. Exiting.")
            return

        print("\n" + "=" * 80)
        print("🔍 OPTION SYMBOL DISCOVERY FROM DATABASE")
        print("=" * 80)
        tracker.options = tracker.discover_option_symbols_from_database(selected)
        if not tracker.options:
            print("❌ No option symbols found. Exiting.")
            return

        print(f"✅ Found {len(tracker.options)} option symbols")
        tracker.run_websocket_monitoring()

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error in main: {e}")
        logger.exception("Error in main function")

if __name__ == "__main__":
    main()
