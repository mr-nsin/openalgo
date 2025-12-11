"""
OpenAlgo-Integrated Historical Data Fetcher - Main Orchestrator

This is the main entry point for the historical data fetcher that integrates
with OpenAlgo's existing infrastructure, database, and authentication system.
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime, timedelta
<<<<<<< HEAD
from typing import Dict, Any, List, Optional
=======
from typing import Dict, Any, List
>>>>>>> 98cb17d (Fix historicalfetcher)
import time

# Ensure historical_fetcher local packages (database/, models/, etc.) are importable
# Must be done BEFORE any imports that use these packages
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Add the OpenAlgo root directory to Python path (for main OpenAlgo modules)
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))
if _project_root not in sys.path:
    sys.path.append(_project_root)

from historicalfetcher.utils.async_logger import setup_async_logger, get_async_logger
<<<<<<< HEAD
from historicalfetcher.utils.fetch_metrics import FetchMetricsTracker, FailureReason
=======
>>>>>>> 98cb17d (Fix historicalfetcher)

# Initialize async logger early for module-level logging
# Default logger will be reconfigured in __init__ with settings
_early_logger = get_async_logger()
logger = _early_logger.get_logger()

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load from historical_fetcher directory first, then from OpenAlgo root
    env_paths = [
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    ]
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path, override=False)
            logger.info(f"Loaded environment variables from: {env_path}")
            break
    else:
        logger.warning("No .env file found. Using system environment variables only.")
        
except ImportError:
    logger.warning("python-dotenv not installed. Using system environment variables only.")
# Import from historicalfetcher local packages
from historicalfetcher.config.openalgo_settings import OpenAlgoSettings, InstrumentType, TimeFrame
<<<<<<< HEAD
from historicalfetcher.models.data_models import SymbolInfo
=======
>>>>>>> 98cb17d (Fix historicalfetcher)
from historicalfetcher.fetchers.openalgo_zerodha_fetcher import OpenAlgoZerodhaHistoricalFetcher, OpenAlgoSymbolManager
from historicalfetcher.database.optimized_questdb_client import OptimizedQuestDBClient
from historicalfetcher.database.models import FetchSummaryModel
from historicalfetcher.notifications.notification_manager import NotificationManager, NotificationFormatter
from historicalfetcher.utils.performance_monitor import PerformanceMonitor, AsyncTimer

class OpenAlgoHistoricalDataFetcher:
    """
    Main orchestrator for historical data fetching process integrated with OpenAlgo
    """
    
    def __init__(self):
        """Initialize the historical data fetcher with OpenAlgo integration"""
        
        # Load configuration
        self.settings = OpenAlgoSettings()
        
        # Setup async logging using async_logger from utils
        try:
            # Create logs directory if it doesn't exist
            log_dir = os.path.dirname(self.settings.log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Fix log retention format - ensure it has proper units
            log_retention = getattr(self.settings, 'log_retention', '30 days')
            if log_retention and log_retention.isdigit():
                log_retention = f"{log_retention} days"
            
            self.async_logger = setup_async_logger(
                log_file=self.settings.log_file_path,
                level=self.settings.log_level,
                rotation=self.settings.log_rotation,
                retention=log_retention
            )
        except Exception as e:
            print(f"Error setting up async logger: {e}")
            import traceback
            traceback.print_exc()
            raise
        # Update module-level logger to use configured async logger
        global logger
        logger = self.async_logger.get_logger()
        
        # Initialize components with async logger
        self.symbol_manager = OpenAlgoSymbolManager(self.settings)
        self.zerodha_fetcher = OpenAlgoZerodhaHistoricalFetcher(self.settings)
        self.questdb_client = OptimizedQuestDBClient(self.settings, async_logger=self.async_logger)
        self.notification_manager = NotificationManager(self.settings)
        self.performance_monitor = PerformanceMonitor(
            collection_interval=60.0,
            enable_auto_collection=self.settings.enable_performance_monitoring
        )
        
<<<<<<< HEAD
        # Initialize metrics tracker for detailed reporting
        self.metrics_tracker = FetchMetricsTracker(
            email_notifier=self.notification_manager.email_notifier if hasattr(self.notification_manager, 'email_notifier') else None,
            telegram_notifier=self.notification_manager.telegram_notifier if hasattr(self.notification_manager, 'telegram_notifier') else None
        )
        
=======
>>>>>>> 98cb17d (Fix historicalfetcher)
        # Processing statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_symbols': 0,
            'processed_symbols': 0,
            'successful_symbols': 0,
            'failed_symbols': 0,
            'total_records': 0,
            'instrument_type_stats': {},
            'exchange_stats': {},
            'timeframe_stats': {},
            'errors': [],
            'warnings': []
        }
        
        # Progress tracking
        self.last_progress_notification = 0
        self.progress_notification_interval = 300  # 5 minutes
    
    async def run(self):
        """Main execution method"""
        
        try:
            logger.info("🚀 Starting OpenAlgo Historical Data Fetcher")
            await self.async_logger.log_system_metrics(await self._get_system_info())
            
            # Initialize all components
            await self._initialize_components()
            
            # Start performance monitoring
            await self.performance_monitor.start_monitoring()
            
            # Get symbols categorized by instrument type
            symbols_by_type = await self.symbol_manager.get_all_active_symbols()
            
            # Calculate total symbols and start processing stats
            total_symbols = sum(len(symbols) for symbols in symbols_by_type.values())
            self.stats['total_symbols'] = total_symbols
            self.stats['start_time'] = datetime.now()
            
<<<<<<< HEAD
            # Initialize metrics tracker
            self.metrics_tracker.start_tracking(total_symbols)
            
=======
>>>>>>> 98cb17d (Fix historicalfetcher)
            self.performance_monitor.start_processing(total_symbols)
            
            logger.info(f"📊 Found {total_symbols:,} symbols across {len(symbols_by_type)} instrument types")
            await self._log_symbol_breakdown(symbols_by_type)
            
            # Send initial notification if configured
            if self.notification_manager.is_configured():
                await self.notification_manager.send_custom_message(
                    f"🚀 OpenAlgo Historical data fetch started\n"
                    f"📊 Processing {total_symbols:,} symbols across {len(symbols_by_type)} instrument types",
                    channels=['telegram']
                )
            
<<<<<<< HEAD
            # Process each instrument type in priority order
            # Priority: INDEX CE/PE FIRST, then INDEX symbols, then equity CE/PE, then FUT, EQ
            
            # Helper function to check if an option is an index option
            def extract_underlying_from_option(symbol_info):
                """Extract underlying symbol from option symbol (e.g., "NIFTY28NOV2424000CE" -> "NIFTY")"""
                import re
                symbol_upper = symbol_info.symbol.upper()
                
                # Pattern: UNDERLYING + DATE + STRIKE + CE/PE
                match = re.match(r"^([A-Z]+)(\d{2}[A-Z]{3}\d{2}[\d.]+)(CE|PE)$", symbol_upper)
                if match:
                    return match.group(1)
                
                # Fallback: try to extract from name field if available
                if hasattr(symbol_info, 'name') and symbol_info.name:
                    return symbol_info.name.upper()
                
                return None
            
            def is_index_option(symbol_info):
                """Check if a CE/PE option is an index option based on underlying symbol"""
                underlying = extract_underlying_from_option(symbol_info)
                if not underlying:
                    return False
                
                # Check if underlying is an index
                index_underlyings = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX", 
                                   "NIFTY50", "NIFTY IT", "NIFTY BANK", "NIFTY NEXT 50", "NIFTY MIDCAP 50"]
                # Direct match
                if underlying in index_underlyings:
                    return True
                
                # Partial match (e.g., "NIFTY50" contains "NIFTY")
                for index_underlying in index_underlyings:
                    if index_underlying in underlying or underlying in index_underlying:
                        return True
                
                return False
            
            def is_allowed_underlying(symbol_info):
                """Check if option's underlying is in the allowed list"""
                # If filtering is disabled, allow all
                if not self.settings.options_filter_by_underlying:
                    return True
                
                # If no allowed underlyings specified, allow all
                if not self.settings.options_allowed_underlyings:
                    return True
                
                underlying = extract_underlying_from_option(symbol_info)
                if not underlying:
                    return False
                
                # Check if underlying is in allowed list (case-insensitive, exact match)
                underlying_upper = underlying.upper()
                for allowed in self.settings.options_allowed_underlyings:
                    if allowed.upper() == underlying_upper:
                        return True
                    # Also check for partial matches (e.g., "NIFTY50" contains "NIFTY")
                    if allowed.upper() in underlying_upper or underlying_upper in allowed.upper():
                        return True
                
                return False
            
            # Helper function to parse expiry date from symbol or expiry field
            def parse_expiry_date(symbol_info):
                """Parse expiry date from symbol_info"""
                from datetime import datetime as dt
                import re
                
                # First try to use expiry field if available
                if symbol_info.expiry:
                    try:
                        # Try DD-MMM-YY format (e.g., "28-Nov-24")
                        return dt.strptime(symbol_info.expiry, "%d-%b-%y").date()
                    except:
                        try:
                            # Try DDMMMYY format (e.g., "28NOV24")
                            if len(symbol_info.expiry) == 7 and symbol_info.expiry[2:5].isalpha():
                                day = int(symbol_info.expiry[:2])
                                month_str = symbol_info.expiry[2:5].upper()
                                year = int('20' + symbol_info.expiry[5:7])
                                month_map = {
                                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                                    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                                }
                                month = month_map.get(month_str, 1)
                                return dt(year, month, day).date()
                        except:
                            pass
                
                # Try to extract from symbol: SYMBOL + DDMMMYY + STRIKE + CE/PE
                match = re.match(r"^[A-Z]+(\d{2}[A-Z]{3}\d{2})[\d.]+(CE|PE)$", symbol_info.symbol.upper())
                if match:
                    expiry_str = match.group(1)  # e.g., "28NOV24"
                    try:
                        day = int(expiry_str[:2])
                        month_str = expiry_str[2:5].upper()
                        year = int('20' + expiry_str[5:7])
                        month_map = {
                            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                        }
                        month = month_map.get(month_str, 1)
                        return dt(year, month, day).date()
                    except:
                        pass
                
                return None
            
            # Helper function to filter and sort options by expiry
            def filter_options_by_expiry(options_list):
                """Filter options to only include nearest N expiries, sorted by expiry date"""
                from datetime import date
                
                today = date.today()
                max_expiries = self.settings.options_max_expiries
                
                # Group options by expiry date
                options_by_expiry = {}
                for option in options_list:
                    expiry_date = parse_expiry_date(option)
                    if expiry_date:
                        # Only include expiries not in the past
                        if expiry_date >= today:
                            if expiry_date not in options_by_expiry:
                                options_by_expiry[expiry_date] = []
                            options_by_expiry[expiry_date].append(option)
                    else:
                        # If we can't parse expiry, include it but log warning
                        logger.debug(f"Could not parse expiry for {option.symbol}, including anyway")
                        # Add to a special "unknown" expiry group
                        if None not in options_by_expiry:
                            options_by_expiry[None] = []
                        options_by_expiry[None].append(option)
                
                # Sort expiries by date (earliest first), None expiries go last
                sorted_expiries = sorted([exp for exp in options_by_expiry.keys() if exp is not None])
                if None in options_by_expiry:
                    sorted_expiries.append(None)
                
                # Take only the first N expiries (nearest expiries)
                selected_expiries = sorted_expiries[:max_expiries]
                
                # Flatten the list of options, maintaining order by expiry
                filtered_options = []
                for expiry_date in selected_expiries:
                    filtered_options.extend(options_by_expiry[expiry_date])
                
                # Log filtering results
                if len(options_list) != len(filtered_options):
                    logger.info(f"📅 Options expiry filtering:")
                    logger.info(f"   Total options: {len(options_list)}")
                    logger.info(f"   After filtering: {len(filtered_options)}")
                    logger.info(f"   Selected expiries: {len([e for e in selected_expiries if e is not None])} (max: {max_expiries} nearest expiries)")
                    if selected_expiries and selected_expiries[0] is not None:
                        logger.info(f"   Expiry dates: {[str(exp) for exp in selected_expiries if exp is not None]}")
                
                return filtered_options

            # ============================================================
            # PRIORITY 1: INDEX CE OPTIONS (Fetch FIRST)
            # ============================================================
            if 'CE' in symbols_by_type and symbols_by_type['CE']:
                ce_symbols = symbols_by_type['CE']
                index_ce_symbols = [s for s in ce_symbols if is_index_option(s)]
                
                # Filter by allowed underlyings if configured
                if self.settings.options_filter_by_underlying and self.settings.options_allowed_underlyings:
                    before_count = len(index_ce_symbols)
                    index_ce_symbols = [s for s in index_ce_symbols if is_allowed_underlying(s)]
                    if before_count != len(index_ce_symbols):
                        logger.info(f"📋 Filtered INDEX CE options: {before_count:,} -> {len(index_ce_symbols):,} (allowed underlyings: {', '.join(self.settings.options_allowed_underlyings)})")
                
                # Filter and sort by expiry: only nearest N expiries within M months, earliest first
                if index_ce_symbols:
                    index_ce_symbols = filter_options_by_expiry(index_ce_symbols)
                    
                    # Process INDEX CE options FIRST (highest priority)
                    if index_ce_symbols:
                        logger.info(f"🔄 [PRIORITY 1] Processing {len(index_ce_symbols):,} INDEX CE options (filtered by expiry)")
                        await self._process_instrument_type('CE', index_ce_symbols)
            
            # ============================================================
            # PRIORITY 2: INDEX PE OPTIONS (Fetch SECOND)
            # ============================================================
            if 'PE' in symbols_by_type and symbols_by_type['PE']:
                pe_symbols = symbols_by_type['PE']
                index_pe_symbols = [s for s in pe_symbols if is_index_option(s)]
                
                # Filter by allowed underlyings if configured
                if self.settings.options_filter_by_underlying and self.settings.options_allowed_underlyings:
                    before_count = len(index_pe_symbols)
                    index_pe_symbols = [s for s in index_pe_symbols if is_allowed_underlying(s)]
                    if before_count != len(index_pe_symbols):
                        logger.info(f"📋 Filtered INDEX PE options: {before_count:,} -> {len(index_pe_symbols):,} (allowed underlyings: {', '.join(self.settings.options_allowed_underlyings)})")
                
                # Filter and sort by expiry: only nearest N expiries within M months, earliest first
                if index_pe_symbols:
                    index_pe_symbols = filter_options_by_expiry(index_pe_symbols)
                    
                    # Process INDEX PE options SECOND
                    if index_pe_symbols:
                        logger.info(f"🔄 [PRIORITY 2] Processing {len(index_pe_symbols):,} INDEX PE options (filtered by expiry)")
                        await self._process_instrument_type('PE', index_pe_symbols)
            
            # ============================================================
            # PRIORITY 3: INDEX SYMBOLS (Spot/Index values like NIFTY, BANKNIFTY)
            # ============================================================
            if 'INDEX' in symbols_by_type and symbols_by_type['INDEX']:
                index_symbols = symbols_by_type['INDEX']
                
                # Separate NSE_INDEX and other INDEX symbols
                nse_index_symbols = [s for s in index_symbols if s.exchange == 'NSE_INDEX']
                other_index_symbols = [s for s in index_symbols if s.exchange != 'NSE_INDEX']
                
                # Process NSE_INDEX first (includes NIFTY, BANKNIFTY, etc.)
                if nse_index_symbols:
                    logger.info(f"🔄 [PRIORITY 3] Processing {len(nse_index_symbols):,} NSE_INDEX symbols")
                    await self._process_instrument_type('INDEX', nse_index_symbols)
                
                # Then process other INDEX symbols (BSE_INDEX, etc.)
                if other_index_symbols:
                    logger.info(f"🔄 [PRIORITY 3] Processing {len(other_index_symbols):,} other INDEX symbols")
                    await self._process_instrument_type('INDEX', other_index_symbols)
            
            # ============================================================
            # PRIORITY 4: EQUITY CE OPTIONS
            # ============================================================
            if 'CE' in symbols_by_type and symbols_by_type['CE']:
                ce_symbols = symbols_by_type['CE']
                equity_ce_symbols = [s for s in ce_symbols if not is_index_option(s)]
                
                # Filter by allowed underlyings if configured (for equity options too)
                if self.settings.options_filter_by_underlying and self.settings.options_allowed_underlyings:
                    before_count = len(equity_ce_symbols)
                    equity_ce_symbols = [s for s in equity_ce_symbols if is_allowed_underlying(s)]
                    if before_count != len(equity_ce_symbols):
                        logger.info(f"📋 Filtered EQUITY CE options: {before_count:,} -> {len(equity_ce_symbols):,} (allowed underlyings: {', '.join(self.settings.options_allowed_underlyings)})")
                
                # Filter and sort by expiry: only nearest N expiries within M months, earliest first
                if equity_ce_symbols:
                    equity_ce_symbols = filter_options_by_expiry(equity_ce_symbols)
                    
                    # Process equity CE options
                    if equity_ce_symbols:
                        logger.info(f"🔄 [PRIORITY 4] Processing {len(equity_ce_symbols):,} equity CE options (filtered by expiry)")
                        await self._process_instrument_type('CE', equity_ce_symbols)
            
            # ============================================================
            # PRIORITY 5: EQUITY PE OPTIONS
            # ============================================================
            if 'PE' in symbols_by_type and symbols_by_type['PE']:
                pe_symbols = symbols_by_type['PE']
                equity_pe_symbols = [s for s in pe_symbols if not is_index_option(s)]
                
                # Filter by allowed underlyings if configured (for equity options too)
                if self.settings.options_filter_by_underlying and self.settings.options_allowed_underlyings:
                    before_count = len(equity_pe_symbols)
                    equity_pe_symbols = [s for s in equity_pe_symbols if is_allowed_underlying(s)]
                    if before_count != len(equity_pe_symbols):
                        logger.info(f"📋 Filtered EQUITY PE options: {before_count:,} -> {len(equity_pe_symbols):,} (allowed underlyings: {', '.join(self.settings.options_allowed_underlyings)})")
                
                # Filter and sort by expiry: only nearest N expiries within M months, earliest first
                if equity_pe_symbols:
                    equity_pe_symbols = filter_options_by_expiry(equity_pe_symbols)
                    
                    # Process equity PE options
                    if equity_pe_symbols:
                        logger.info(f"🔄 [PRIORITY 5] Processing {len(equity_pe_symbols):,} equity PE options (filtered by expiry)")
                        await self._process_instrument_type('PE', equity_pe_symbols)
            
            # Process remaining instrument types (FUT, EQ)
            remaining_types = ['FUT', 'EQ']
            for instrument_type in remaining_types:
                if instrument_type in symbols_by_type and symbols_by_type[instrument_type]:
                    symbols = symbols_by_type[instrument_type]
                    logger.info(f"🔄 Processing {len(symbols):,} {instrument_type} symbols")
                    await self._process_instrument_type(instrument_type, symbols)
            
            # Process any remaining instrument types not in priority list
            processed_types = {'INDEX', 'CE', 'PE', 'FUT', 'EQ'}
            for instrument_type, symbols in symbols_by_type.items():
                if instrument_type not in processed_types:
                    if symbols:
                        logger.info(f"🔄 Processing {len(symbols):,} {instrument_type} symbols")
                        await self._process_instrument_type(instrument_type, symbols)
=======
            # Process each instrument type
            for instrument_type, symbols in symbols_by_type.items():
                if not symbols:
                    continue
                
                logger.info(f"🔄 Processing {instrument_type} symbols ({len(symbols):,} symbols)")
                await self._process_instrument_type(instrument_type, symbols)
>>>>>>> 98cb17d (Fix historicalfetcher)
            
            # Finalize processing
            await self._finalize_processing()
            
            # Send success notification
            await self.notification_manager.send_success_notification(self.stats)
            
            logger.info("✅ OpenAlgo Historical data fetch completed successfully")
            
        except KeyboardInterrupt:
            logger.info("⏹️ Process interrupted by user")
            await self._handle_interruption()
            
        except Exception as e:
            logger.exception(f"💥 Fatal error in historical data fetcher: {e}")
            
            await self._handle_fatal_error(e)
            raise
            
        finally:
            await self._cleanup_components()
    
    async def _initialize_components(self):
        """Initialize all components and verify connections"""
        
        logger.info("🔗 Initializing OpenAlgo components...")
        
        # Initialize Zerodha fetcher with async logger
        await self.zerodha_fetcher.initialize(async_logger=self.async_logger)
        
        # Connect to QuestDB (table creation is handled automatically by table manager)
        await self.questdb_client.connect()
        
        # Test notification connections
        if self.notification_manager.is_configured():
            connection_results = await self.notification_manager.test_all_connections()
            for channel, status in connection_results.items():
                if status:
                    logger.info(f"✅ {channel.title()} notification connection verified")
                else:
                    logger.warning(f"⚠️ {channel.title()} notification connection failed")
        
        logger.info("✅ All OpenAlgo components initialized successfully")
    
    async def _log_symbol_breakdown(self, symbols_by_type: Dict[str, List]):
        """Log detailed symbol breakdown"""
        
        for instrument_type, symbols in symbols_by_type.items():
            count = len(symbols)
            self.stats['instrument_type_stats'][instrument_type] = count
            
            if count > 0:
                # Get exchange breakdown for this instrument type
                exchange_breakdown = {}
                for symbol in symbols:
                    exchange = symbol.exchange
                    exchange_breakdown[exchange] = exchange_breakdown.get(exchange, 0) + 1
                
                logger.info(f"  • {instrument_type}: {count:,} symbols")
                for exchange, ex_count in exchange_breakdown.items():
                    logger.info(f"    - {exchange}: {ex_count:,}")
                    
                    # Update global exchange stats
                    self.stats['exchange_stats'][exchange] = (
                        self.stats['exchange_stats'].get(exchange, 0) + ex_count
                    )
    
    async def _process_instrument_type(self, instrument_type: str, symbols: List):
        """Process all symbols for a specific instrument type"""
        
        # Process symbols in batches with concurrency control
        batch_size = self.settings.batch_size
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(symbols) + batch_size - 1) // batch_size
            
            logger.info(
                f"📦 Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} symbols) for {instrument_type}"
            )
            
            # Process batch concurrently
            tasks = [
                self._process_symbol_with_semaphore(semaphore, symbol)
                for symbol in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze batch results
            successful_in_batch = sum(1 for r in results if not isinstance(r, Exception))
            failed_in_batch = len(batch) - successful_in_batch
            
            logger.info(
                f"✅ Batch {batch_num} completed: "
                f"{successful_in_batch}/{len(batch)} successful, "
                f"{failed_in_batch} failed"
            )
            
            # Send progress notification if enough time has passed
            await self._maybe_send_progress_notification(batch[-1].symbol if batch else 'N/A')
            
            # Check resource limits
            await self._check_resource_limits()
    
    async def _process_symbol_with_semaphore(self, semaphore, symbol_info):
        """Process single symbol with rate limiting and error handling"""
        
        async with semaphore:
            start_time = time.monotonic()
            
            try:
                records_inserted = await self._process_single_symbol(symbol_info)
                
                # Update statistics
                self.stats['successful_symbols'] += 1
                self.stats['total_records'] += records_inserted
                
                processing_time = time.monotonic() - start_time
                
                # Log symbol processing
                await self.async_logger.log_symbol_processing(
                    symbol=symbol_info.symbol,
                    instrument_type=symbol_info.instrument_type,
                    timeframe="all",
                    records_count=records_inserted,
                    processing_time=processing_time,
                    status="success"
                )
                
                logger.debug(
                    f"✅ {symbol_info.symbol} ({symbol_info.instrument_type}): "
                    f"{records_inserted:,} records in {processing_time:.2f}s"
                )
                
            except Exception as e:
                # Update statistics
                self.stats['failed_symbols'] += 1
                processing_time = time.monotonic() - start_time
                
                error_info = {
                    'symbol': symbol_info.symbol,
                    'exchange': symbol_info.exchange,
                    'instrument_type': symbol_info.instrument_type,
                    'error': str(e),
                    'processing_time': processing_time
                }
                self.stats['errors'].append(error_info)
                
                # Log error
                await self.async_logger.log_symbol_processing(
                    symbol=symbol_info.symbol,
                    instrument_type=symbol_info.instrument_type,
                    timeframe="all",
                    records_count=0,
                    processing_time=processing_time,
                    status="error"
                )
                
                logger.error(
                    f"❌ Error processing {symbol_info.symbol} "
                    f"({symbol_info.instrument_type}) after {processing_time:.2f}s: {e}"
                )
            
            finally:
                self.stats['processed_symbols'] += 1
                
                # Update performance monitor
                self.performance_monitor.update_processing_progress(
                    processed=self.stats['processed_symbols'],
                    successful=self.stats['successful_symbols'],
                    failed=self.stats['failed_symbols']
                )
    
<<<<<<< HEAD
    async def _fetch_underlying_spot_price(self, symbol_info: SymbolInfo) -> Optional[float]:
        """
        Fetch spot price (LTP) for the underlying symbol of an option
        
        Args:
            symbol_info: SymbolInfo for the option (CE/PE)
            
        Returns:
            Spot price (LTP) as float, or None if fetch fails
        """
        try:
            # Extract underlying symbol from option symbol
            import re
            underlying = symbol_info.symbol
            match = re.match(r"^([A-Z]+)(\d{2}[A-Z]{3}\d{2}[\d.]+)(CE|PE)$", symbol_info.symbol.upper())
            if match:
                underlying = match.group(1)
            else:
                logger.warning(f"Could not extract underlying from option symbol: {symbol_info.symbol}")
                return None
            
            # Determine underlying exchange
            # For NFO options, underlying is usually on NSE or NSE_INDEX
            underlying_exchange = "NSE"
            if underlying in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"]:
                underlying_exchange = "NSE_INDEX"
            
            # Fetch quote using OpenAlgo client's broker_data
            if not self.zerodha_fetcher.openalgo_client:
                await self.zerodha_fetcher.initialize()
            
            client = self.zerodha_fetcher.openalgo_client
            
            # Use broker_data.get_quotes method if available
            try:
                if hasattr(client, 'broker_data') and hasattr(client.broker_data, 'get_quotes'):
                    quote_data = client.broker_data.get_quotes(symbol=underlying, exchange=underlying_exchange)
                    if quote_data and isinstance(quote_data, dict):
                        ltp = quote_data.get('ltp')
                        if ltp and ltp > 0:
                            logger.debug(f"Fetched spot price for {underlying}: {ltp}")
                            return float(ltp)
                elif hasattr(client, 'get_quotes'):
                    # Direct get_quotes method
                    quote_data = client.get_quotes(symbol=underlying, exchange=underlying_exchange)
                    if quote_data and isinstance(quote_data, dict):
                        ltp = quote_data.get('ltp')
                        if ltp and ltp > 0:
                            logger.debug(f"Fetched spot price for {underlying}: {ltp}")
                            return float(ltp)
                else:
                    # Try using services/quotes_service
                    from services.quotes_service import get_quotes
                    success, quote_response, status_code = get_quotes(
                        symbol=underlying,
                        exchange=underlying_exchange,
                        api_key=self.settings.openalgo_api_key
                    )
                    if success and quote_response:
                        ltp = quote_response.get('data', {}).get('ltp')
                        if ltp and ltp > 0:
                            logger.debug(f"Fetched spot price for {underlying}: {ltp}")
                            return float(ltp)
            except Exception as e:
                logger.debug(f"Error fetching spot price for {underlying}: {e}")
            
            logger.warning(f"Could not fetch LTP for {underlying} on {underlying_exchange}")
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching spot price for {symbol_info.symbol} (underlying: {underlying}): {e}")
            return None
    
=======
>>>>>>> 98cb17d (Fix historicalfetcher)
    async def _process_single_symbol(self, symbol_info) -> int:
        """Process historical data for a single symbol across all timeframes"""
        
        total_records = 0
        timeframes = self.settings.get_timeframe_objects()
        
        # Process timeframes in priority order (daily first, then intraday)
        ordered_timeframes = self._get_processing_order(timeframes)
        
        for timeframe in ordered_timeframes:
<<<<<<< HEAD
            from_date = None
            to_date = None
            date_range_str = "N/A"
            
=======
>>>>>>> 98cb17d (Fix historicalfetcher)
            try:
                async with AsyncTimer(self.performance_monitor, f'fetch_{timeframe.value}'):
                    # Determine date range
                    from_date, to_date = await self._get_date_range(symbol_info, timeframe)
                    
                    if not from_date or not to_date:
<<<<<<< HEAD
                        # Track as failure - no date range
                        date_range_str = "Invalid date range"
                        self.metrics_tracker.record_symbol_failure(
                            symbol_info=symbol_info,
                            timeframe=timeframe,
                            date_range=date_range_str,
                            failure_reason=FailureReason.CONVERSION_ERROR,
                            error_message="Invalid or missing date range"
                        )
                        continue
                    
                    date_range_str = f"{from_date.date()} to {to_date.date()}"
                    
                    # ============================================================
                    # VALIDATE DATE RANGE BEFORE API CALL
                    # ============================================================
                    today_datetime = datetime.combine(datetime.now().date(), datetime.min.time())
                    validation_errors = []
                    
                    # Check if dates are in the future
                    if from_date > today_datetime:
                        validation_errors.append(f"Start date {from_date.date()} is in the future")
                    if to_date > today_datetime:
                        validation_errors.append(f"End date {to_date.date()} is in the future")
                    
                    # Check if start_date > end_date
                    if from_date > to_date:
                        validation_errors.append(f"Start date {from_date.date()} is after end date {to_date.date()}")
                    
                    # Check date range size (warn if too large)
                    days_in_range = (to_date - from_date).days
                    if days_in_range > 400:
                        validation_errors.append(f"Date range is very large: {days_in_range} days (may cause broker API issues)")
                    
                    if validation_errors:
                        error_msg = "; ".join(validation_errors)
                        logger.error(f"❌ DATE VALIDATION FAILED for {symbol_info.symbol} ({symbol_info.exchange}, {symbol_info.instrument_type}): {error_msg}")
                        logger.error(f"   Requested range: {date_range_str}")
                        logger.error(f"   Today: {today_datetime.date()}")
                        
                        self.metrics_tracker.record_symbol_failure(
                            symbol_info=symbol_info,
                            timeframe=timeframe,
                            date_range=date_range_str,
                            failure_reason=FailureReason.CONVERSION_ERROR,
                            error_message=f"Date validation failed: {error_msg}"
                        )
                        continue
                    
                    # ============================================================
                    # DETAILED LOGGING BEFORE API CALL
                    # ============================================================
                    logger.info(f"📥 Fetching {symbol_info.symbol} | Exchange: {symbol_info.exchange} | Type: {symbol_info.instrument_type} | Timeframe: {timeframe.value} | Date Range: {date_range_str} | Days: {days_in_range}")
                    logger.debug(f"   📋 Request Details:")
                    logger.debug(f"      - Symbol: {symbol_info.symbol}")
                    logger.debug(f"      - Exchange: {symbol_info.exchange}")
                    logger.debug(f"      - Instrument Type: {symbol_info.instrument_type}")
                    logger.debug(f"      - Token: {symbol_info.token}")
                    logger.debug(f"      - Timeframe: {timeframe.value}")
                    logger.debug(f"      - Start Date: {from_date.date()} ({from_date.strftime('%Y-%m-%d %H:%M:%S')})")
                    logger.debug(f"      - End Date: {to_date.date()} ({to_date.strftime('%Y-%m-%d %H:%M:%S')})")
                    logger.debug(f"      - Days in Range: {days_in_range}")
                    logger.debug(f"      - Today: {today_datetime.date()}")
                    
                    # Increment total timeframes counter
                    self.metrics_tracker.increment_total_timeframes()
                    
                    # Fetch historical data with enhanced error handling
                    try:
                        candles = await self.zerodha_fetcher.fetch_historical_data(
                            symbol_info,
                            timeframe,
                            from_date,
                            to_date
                        )
                    except Exception as fetch_error:
                        # Enhanced error logging for broker API failures
                        error_msg = str(fetch_error)
                        error_type = type(fetch_error).__name__
                        
                        logger.error(f"❌ FETCH ERROR for {symbol_info.symbol} ({symbol_info.exchange}, {symbol_info.instrument_type}):")
                        logger.error(f"   Error Type: {error_type}")
                        logger.error(f"   Error Message: {error_msg}")
                        logger.error(f"   Symbol: {symbol_info.symbol}")
                        logger.error(f"   Exchange: {symbol_info.exchange}")
                        logger.error(f"   Instrument Type: {symbol_info.instrument_type}")
                        logger.error(f"   Timeframe: {timeframe.value}")
                        logger.error(f"   Date Range: {date_range_str}")
                        logger.error(f"   Days in Range: {days_in_range}")
                        
                        # Check for specific broker API error patterns
                        if "Error for chunk" in error_msg or "ERROR in data" in error_msg:
                            logger.error(f"   🔍 BROKER API CHUNK ERROR DETECTED - This indicates broker API is failing to fetch data chunks")
                            logger.error(f"   💡 Possible causes:")
                            logger.error(f"      1. Symbol format mismatch for broker API")
                            logger.error(f"      2. Exchange not properly handled by broker")
                            logger.error(f"      3. Date range too large for broker API")
                            logger.error(f"      4. Broker API rate limiting or service issues")
                        
                        # Re-raise to be handled by outer exception handler
                        raise
                    
                    if candles:
                        # Calculate min/max timestamps from candles
                        min_timestamp = min(c.timestamp for c in candles)
                        max_timestamp = max(c.timestamp for c in candles)
                        
                        # Handle both datetime objects and Unix timestamps
                        if isinstance(min_timestamp, datetime):
                            min_time_str = min_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            min_time_str = datetime.fromtimestamp(min_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        
                        if isinstance(max_timestamp, datetime):
                            max_time_str = max_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            max_time_str = datetime.fromtimestamp(max_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Log after fetching: Records count, Min/Max timestamp with symbol
                        logger.info(f"📊 Received {symbol_info.symbol} | {len(candles):,} records | Min: {min_time_str} | Max: {max_time_str} | Timeframe: {timeframe.value}")
                        
                        # Get spot price for options (CE/PE)
                        # Note: Historical API does NOT include underlying spot price in the response
                        # The API only returns: timestamp, open, high, low, close, volume, oi
                        # We need to fetch spot price separately via quotes API
                        spot_price = None
                        if symbol_info.instrument_type in [InstrumentType.CE, InstrumentType.PE, 'CE', 'PE']:
                            logger.debug(f"Fetching underlying spot price for {symbol_info.symbol} via quotes API...")
                            spot_price = await self._fetch_underlying_spot_price(symbol_info)
                            
                            # If spot price is not available, estimate from strike + option price
                            if spot_price is None or spot_price == 0:
                                if symbol_info.strike and symbol_info.strike > 0:
                                    # Use strike price + option premium as estimated spot for CE
                                    # Use strike price - option premium as estimated spot for PE
                                    # This is a rough approximation assuming slight ITM
                                    if candles and len(candles) > 0:
                                        latest_candle = candles[-1]
                                        option_premium = latest_candle.close
                                        is_call = symbol_info.instrument_type in [InstrumentType.CE, 'CE']
                                        
                                        if is_call:
                                            # For CE: Spot ≈ Strike + Intrinsic Value
                                            # Assume option is slightly ITM, so spot ≈ strike + premium * 0.5
                                            spot_price = symbol_info.strike + (option_premium * 0.3)
                                        else:
                                            # For PE: Spot ≈ Strike - Intrinsic Value
                                            spot_price = symbol_info.strike - (option_premium * 0.3)
                                        
                                        logger.warning(f"⚠️ Could not fetch spot price for {symbol_info.symbol}, "
                                                     f"estimated spot={spot_price:.2f} from strike={symbol_info.strike} "
                                                     f"and option premium={option_premium:.2f}")
                                    else:
                                        # No candles, use strike as approximation (ATM assumption)
                                        spot_price = symbol_info.strike
                                        logger.warning(f"⚠️ Could not fetch spot price for {symbol_info.symbol}, using strike price ({spot_price}) as fallback")
                                else:
                                    logger.warning(f"⚠️ Could not fetch spot price for {symbol_info.symbol} and no strike available, Greeks calculation will be skipped")
                                    spot_price = 0.0
                            else:
                                logger.debug(f"✅ Fetched spot price for {symbol_info.symbol}: {spot_price}")
                        
                        # Get table name before storing (for logging)
                        table_name = await self.questdb_client.table_manager.get_or_create_table(symbol_info)
                        
                        # Store in QuestDB
                        async with AsyncTimer(self.performance_monitor, f'store_{timeframe.value}'):
                            records_inserted = await self.questdb_client.upsert_historical_data_with_indicators(
                                symbol_info,
                                timeframe,
                                candles,
                                spot_price=spot_price
=======
                        continue
                    
                    # Fetch historical data
                    candles = await self.zerodha_fetcher.fetch_historical_data(
                        symbol_info,
                        timeframe,
                        from_date,
                        to_date
                    )
                    
                    if candles:
                        # Store in QuestDB
                        async with AsyncTimer(self.performance_monitor, f'store_{timeframe.value}'):
                            records_inserted = await self.questdb_client.upsert_historical_data(
                                symbol_info,
                                timeframe.value,
                                candles
>>>>>>> 98cb17d (Fix historicalfetcher)
                            )
                        
                        total_records += records_inserted
                        
<<<<<<< HEAD
                        # Track success in metrics
                        self.metrics_tracker.record_symbol_success(
                            symbol_info=symbol_info,
                            timeframe=timeframe,
                            candles_count=len(candles),
                            records_inserted=records_inserted
                        )
                        
=======
>>>>>>> 98cb17d (Fix historicalfetcher)
                        # Update timeframe stats
                        if timeframe.value not in self.stats['timeframe_stats']:
                            self.stats['timeframe_stats'][timeframe.value] = 0
                        self.stats['timeframe_stats'][timeframe.value] += records_inserted
                        
                        # Update fetch status
                        await self.questdb_client.update_fetch_status(
                            symbol_info,
                            timeframe.value,
                            'success',
                            records_inserted
                        )
                        
<<<<<<< HEAD
                        # Log after saving: Success with records inserted, symbol, timeframe, date range, table name
                        logger.info(f"✅ Saved {symbol_info.symbol} | Timeframe: {timeframe.value} | Date Range: {date_range_str} | Records: {records_inserted:,} | Table: {table_name}")
                    else:
                        # Track as failure - no data
                        logger.warning(f"⚠️ No data found for {symbol_info.symbol} | Timeframe: {timeframe.value} | Date Range: {date_range_str}")
                        
                        self.metrics_tracker.record_symbol_failure(
                            symbol_info=symbol_info,
                            timeframe=timeframe,
                            date_range=date_range_str,
                            failure_reason=FailureReason.NO_DATA,
                            error_message="No data returned from API"
                        )
=======
                        logger.debug(
                            f"✅ {symbol_info.symbol} ({timeframe.value}): {records_inserted:,} records"
                        )
                    else:
                        logger.debug(f"⚠️ No data found for {symbol_info.symbol} ({timeframe.value})")
>>>>>>> 98cb17d (Fix historicalfetcher)
                        
                        await self.questdb_client.update_fetch_status(
                            symbol_info,
                            timeframe.value,
                            'no_data',
                            0
                        )
            
            except Exception as e:
<<<<<<< HEAD
                error_msg = str(e)
                error_type = type(e).__name__
                
                # Enhanced error logging with full context
                logger.exception(f"❌ Error fetching {symbol_info.symbol} ({timeframe.value}): {e}")
                logger.error(f"   📋 Error Context:")
                logger.error(f"      - Symbol: {symbol_info.symbol}")
                logger.error(f"      - Exchange: {symbol_info.exchange}")
                logger.error(f"      - Instrument Type: {symbol_info.instrument_type}")
                logger.error(f"      - Token: {symbol_info.token}")
                logger.error(f"      - Timeframe: {timeframe.value}")
                logger.error(f"      - Date Range: {date_range_str}")
                logger.error(f"      - Error Type: {error_type}")
                logger.error(f"      - Error Message: {error_msg[:500]}")
                
                # Check for broker API chunk errors specifically
                is_broker_chunk_error = "Error for chunk" in error_msg or "ERROR in data" in error_msg
                if is_broker_chunk_error:
                    logger.error(f"   🚨 BROKER API CHUNK ERROR - This is a broker API issue, not historical fetcher issue")
                    logger.error(f"   📝 The broker API (Fyers/FivePaisa) is failing to fetch data chunks")
                    logger.error(f"   🔍 Check broker API logs for detailed error messages")
                
                # Determine failure reason based on error type
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    failure_reason = FailureReason.RATE_LIMIT
                elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    failure_reason = FailureReason.NETWORK_ERROR
                elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                    failure_reason = FailureReason.NETWORK_ERROR
                elif "database" in error_msg.lower() or "questdb" in error_msg.lower() or "Invalid column" in error_msg:
                    failure_reason = FailureReason.DATABASE_ERROR
                elif is_broker_chunk_error or "api" in error_msg.lower():
                    failure_reason = FailureReason.API_ERROR
                else:
                    failure_reason = FailureReason.UNKNOWN_ERROR
                
                # Track failure with timeframe and date range
                self.metrics_tracker.record_symbol_failure(
                    symbol_info=symbol_info,
                    timeframe=timeframe,
                    date_range=date_range_str,
                    failure_reason=failure_reason,
                    error_message=error_msg[:500]  # Limit error message length
                )
=======
                logger.exception(f"❌ Error fetching {symbol_info.symbol} ({timeframe.value}): {e}")
>>>>>>> 98cb17d (Fix historicalfetcher)
                
                await self.questdb_client.update_fetch_status(
                    symbol_info,
                    timeframe.value,
                    'error',
<<<<<<< HEAD
                    0
=======
                    0,
                    str(e)
>>>>>>> 98cb17d (Fix historicalfetcher)
                )
        
        return total_records
    
    def _get_processing_order(self, timeframes: List[TimeFrame]) -> List[TimeFrame]:
        """Get timeframes in processing order (daily first, then intraday)"""
        daily_timeframes = [tf for tf in timeframes if tf == TimeFrame.DAILY]
        intraday_timeframes = [tf for tf in timeframes if tf != TimeFrame.DAILY]
        
        # Sort intraday timeframes by duration (longest first)
        intraday_timeframes.sort(key=lambda x: self._get_timeframe_minutes(x), reverse=True)
        
        return daily_timeframes + intraday_timeframes
    
    def _get_timeframe_minutes(self, timeframe: TimeFrame) -> int:
        """Get timeframe duration in minutes"""
        timeframe_minutes = {
            TimeFrame.MINUTE_1: 1,
            TimeFrame.MINUTE_3: 3,
            TimeFrame.MINUTE_5: 5,
            TimeFrame.MINUTE_15: 15,
            TimeFrame.MINUTE_30: 30,
            TimeFrame.HOUR_1: 60,
            TimeFrame.DAILY: 1440
        }
        return timeframe_minutes.get(timeframe, 1)
    
    async def _get_date_range(self, symbol_info, timeframe: TimeFrame) -> tuple:
        """Get appropriate date range for fetching historical data"""
        
<<<<<<< HEAD
        # Fix: Use today's date, not current datetime to avoid future dates
        today = datetime.now().date()
        
        # For intraday timeframes, end date should be previous trading day if after market hours
        # For daily timeframes, end date can be today
        if timeframe in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_3, TimeFrame.MINUTE_5, 
                        TimeFrame.MINUTE_15, TimeFrame.MINUTE_30, TimeFrame.HOUR_1]:
            # For intraday data, use previous trading day if current time is after market close
            current_time = datetime.now().time()
            market_close = datetime.strptime("15:30", "%H:%M").time()
            
            if current_time > market_close or today.weekday() >= 5:  # Weekend check
                # Use previous trading day (skip weekends)
                days_back = 1
                if today.weekday() == 6:  # Sunday
                    days_back = 2
                elif today.weekday() == 0 and current_time <= market_close:  # Monday before market close
                    days_back = 0  # Use today
                else:
                    days_back = 1
                
                end_date = datetime.combine(today - timedelta(days=days_back), datetime.min.time())
            else:
                # Use today
                end_date = datetime.combine(today, datetime.min.time())
        else:
            # For daily data, use today (but skip weekends)
            if today.weekday() >= 5:  # Weekend
                days_back = today.weekday() - 4  # Go back to Friday
                end_date = datetime.combine(today - timedelta(days=days_back), datetime.min.time())
            else:
                end_date = datetime.combine(today, datetime.min.time())
        
        # Check if we have existing data and should do incremental fetch
        # Only do incremental fetch if last fetch was recent (within 7 days), otherwise do full fetch
        last_fetch_date = await self.questdb_client.get_last_fetch_date(symbol_info, timeframe.value)
        
        # Calculate full date range first
        # For options (CE/PE), use shorter historical period (3 months default)
        # For other instruments, use full historical period (365 days default)
        if symbol_info.instrument_type in [InstrumentType.CE, InstrumentType.PE]:
            historical_days = self.settings.options_historical_days_limit
            logger.debug(f"Using options historical days limit: {historical_days} days for {symbol_info.symbol}")
        else:
            historical_days = self.settings.historical_days_limit
            logger.debug(f"Using standard historical days limit: {historical_days} days for {symbol_info.symbol}")
        
        if self.settings.start_date_override:
            try:
                full_start_date = datetime.strptime(self.settings.start_date_override, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Invalid start_date_override: {self.settings.start_date_override}")
                full_start_date = end_date - timedelta(days=historical_days)
        else:
            full_start_date = end_date - timedelta(days=historical_days)
        
        # Decide between incremental and full fetch
        # IMPORTANT: Only do incremental if we have recent data AND the gap is small
        # Otherwise, always do full fetch to ensure we have complete 365 days of data
        if last_fetch_date and self.settings.start_date_override is None:
            # Check if last fetch is recent (within 3 days) and covers enough data
            days_since_last_fetch = (end_date - last_fetch_date).days
            days_covered = (last_fetch_date - full_start_date).days if last_fetch_date > full_start_date else 0
            
            # Only do incremental if:
            # 1. Last fetch was recent (within 3 days)
            # 2. We already have most of the data (at least 90% of historical_days_limit)
            # Use appropriate historical_days_limit based on instrument type
            if symbol_info.instrument_type in [InstrumentType.CE, InstrumentType.PE]:
                required_days_covered = (self.settings.options_historical_days_limit * 0.9)
            else:
                required_days_covered = (self.settings.historical_days_limit * 0.9)
            
            if days_since_last_fetch <= 3 and days_covered >= required_days_covered:
                # Recent data exists and we have most data, do incremental fetch
                start_date = last_fetch_date
            else:
                # Gap is too large or don't have enough data, do full fetch
                start_date = full_start_date
        else:
            # No existing data or override specified, do full fetch
            start_date = full_start_date
        
        # Ensure start_date is not in the future and end_date is not in the future
        today_datetime = datetime.combine(datetime.now().date(), datetime.min.time())
        original_end_date = end_date
        original_start_date = start_date
        
        if end_date > today_datetime:
            # End date should never be in the future
            logger.warning(f"⚠️ End date {end_date.date()} is in the future for {symbol_info.symbol} ({symbol_info.exchange}, {symbol_info.instrument_type}, {timeframe.value}). Adjusting to today {today_datetime.date()}")
            end_date = today_datetime
        
        if start_date > end_date:
            logger.warning(f"⚠️ Start date {start_date.date()} is after end date {end_date.date()} for {symbol_info.symbol} ({symbol_info.exchange}, {symbol_info.instrument_type}, {timeframe.value}). Adjusting start date.")
            # Use appropriate historical_days_limit based on instrument type
            if symbol_info.instrument_type in [InstrumentType.CE, InstrumentType.PE]:
                start_date = end_date - timedelta(days=self.settings.options_historical_days_limit)
            else:
                start_date = end_date - timedelta(days=self.settings.historical_days_limit)
        
        # Ensure start_date is not in the future
        if start_date > today_datetime:
            # Use appropriate historical_days_limit based on instrument type
            if symbol_info.instrument_type in [InstrumentType.CE, InstrumentType.PE]:
                adjusted_days = self.settings.options_historical_days_limit
            else:
                adjusted_days = self.settings.historical_days_limit
            logger.warning(f"⚠️ Start date {start_date.date()} is in the future for {symbol_info.symbol} ({symbol_info.exchange}, {symbol_info.instrument_type}, {timeframe.value}). Adjusting to {today_datetime.date() - timedelta(days=adjusted_days)}")
            start_date = today_datetime - timedelta(days=adjusted_days)
        
        # Calculate actual days in range
        actual_days = (end_date - start_date).days
        
        # Log date range adjustments if any were made
        if original_start_date != start_date or original_end_date != end_date:
            logger.info(f"📅 Date range adjusted for {symbol_info.symbol} ({symbol_info.exchange}, {symbol_info.instrument_type}):")
            logger.info(f"   Original: {original_start_date.date()} to {original_end_date.date()}")
            logger.info(f"   Adjusted: {start_date.date()} to {end_date.date()}")
            logger.info(f"   Days: {actual_days}")
        
        # Warn if range is too small
        # Use appropriate historical_days_limit based on instrument type
        if symbol_info.instrument_type in [InstrumentType.CE, InstrumentType.PE]:
            expected_days = self.settings.options_historical_days_limit
        else:
            expected_days = self.settings.historical_days_limit
        
        if actual_days < expected_days * 0.5:
            logger.warning(f"⚠️ Date range for {symbol_info.symbol} ({symbol_info.exchange}, {symbol_info.instrument_type}, {timeframe.value}) is only {actual_days} days, expected ~{expected_days} days")
        
        # Warn if range is too large (may cause broker API issues)
        if actual_days > 400:
            logger.warning(f"⚠️ Date range for {symbol_info.symbol} ({symbol_info.exchange}, {symbol_info.instrument_type}, {timeframe.value}) is very large: {actual_days} days. Broker API may chunk this into multiple requests.")
        
        # Final validation - ensure dates are valid
        if start_date > end_date:
            logger.error(f"❌ CRITICAL: Date range validation failed for {symbol_info.symbol} ({symbol_info.exchange}, {symbol_info.instrument_type}): start_date {start_date.date()} > end_date {end_date.date()}")
            # Return None to indicate invalid range
            return None, None
        
        if start_date > today_datetime or end_date > today_datetime:
            logger.error(f"❌ CRITICAL: Date range contains future dates for {symbol_info.symbol} ({symbol_info.exchange}, {symbol_info.instrument_type}): start={start_date.date()}, end={end_date.date()}, today={today_datetime.date()}")
            # Return None to indicate invalid range
            return None, None
        
=======
        end_date = datetime.now()
        
        # Check if we have existing data and should do incremental fetch
        last_fetch_date = await self.questdb_client.get_last_fetch_date(symbol_info, timeframe.value)
        
        if last_fetch_date and self.settings.start_date_override is None:
            # Incremental fetch from last successful date
            start_date = last_fetch_date
            logger.debug(f"Incremental fetch for {symbol_info.symbol} from {start_date.date()}")
        else:
            # Full fetch based on settings
            if self.settings.start_date_override:
                try:
                    start_date = datetime.strptime(self.settings.start_date_override, "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Invalid start_date_override: {self.settings.start_date_override}")
                    start_date = end_date - timedelta(days=self.settings.historical_days_limit)
            else:
                start_date = end_date - timedelta(days=self.settings.historical_days_limit)
        
>>>>>>> 98cb17d (Fix historicalfetcher)
        return start_date, end_date
    
    async def _maybe_send_progress_notification(self, current_symbol: str):
        """Send progress notification if enough time has passed"""
        
        current_time = time.time()
        
        if (current_time - self.last_progress_notification) >= self.progress_notification_interval:
            if self.notification_manager.is_configured():
                progress_info = NotificationFormatter.create_progress_summary(
                    processed=self.stats['processed_symbols'],
                    total=self.stats['total_symbols'],
                    current_item=current_symbol
                )
                
                progress_info.update({
                    'elapsed_time': NotificationFormatter.format_duration(
                        (datetime.now() - self.stats['start_time']).total_seconds()
                    ),
                    'records_inserted': self.stats['total_records']
                })
                
                await self.notification_manager.send_progress_notification(progress_info)
            
            self.last_progress_notification = current_time
    
    async def _check_resource_limits(self):
        """Check system resource usage and warn if limits exceeded"""
        
        if not self.settings.enable_performance_monitoring:
            return
        
        resource_check = self.performance_monitor.check_resource_limits(
            max_memory_mb=self.settings.memory_limit_mb,
            max_cpu_percent=80.0  # 80% CPU threshold
        )
        
        if resource_check['status'] == 'warning':
            warning_info = {
                'warning_type': 'Resource Usage',
                'message': '; '.join(resource_check['warnings']),
                'processed_symbols': self.stats['processed_symbols'],
                'records_inserted': self.stats['total_records'],
                'action': 'Continuing with processing'
            }
            
            self.stats['warnings'].append(warning_info)
            
            # Send warning notification
            await self.notification_manager.send_warning_notification(warning_info)
    
    async def _finalize_processing(self):
        """Finalize processing and calculate final statistics"""
        
        self.stats['end_time'] = datetime.now()
        self.performance_monitor.finish_processing()
        
        # Calculate derived statistics
        if self.stats['start_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            self.stats['duration'] = duration.total_seconds() / 60  # minutes
        
        if self.stats['total_symbols'] > 0:
            self.stats['success_rate'] = (self.stats['successful_symbols'] / self.stats['total_symbols']) * 100
        else:
            self.stats['success_rate'] = 0
        
        self.stats['completed_at'] = self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S IST')
        
        # Categorize records by type for reporting
        minute_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h']
        self.stats['minute_data_records'] = sum(
            self.stats['timeframe_stats'].get(tf, 0) for tf in minute_timeframes
        )
        self.stats['daily_data_records'] = self.stats['timeframe_stats'].get('D', 0)
        
<<<<<<< HEAD
        # Finish metrics tracking
        self.metrics_tracker.finish_tracking()
        
=======
>>>>>>> 98cb17d (Fix historicalfetcher)
        # Insert fetch summary into database
        await self._insert_fetch_summary()
        
        # Log final statistics
        await self._log_final_statistics()
<<<<<<< HEAD
        
        # Generate and log detailed report with failed symbols
        await self._log_detailed_completion_report()
=======
>>>>>>> 98cb17d (Fix historicalfetcher)
    
    async def _insert_fetch_summary(self):
        """Insert daily fetch summary into database"""
        
        try:
            summary = FetchSummaryModel(
                fetch_date=datetime.now().date(),
                total_symbols=self.stats['total_symbols'],
                successful_symbols=self.stats['successful_symbols'],
                failed_symbols=self.stats['failed_symbols'],
                total_records_inserted=self.stats['total_records'],
                processing_time_minutes=int(self.stats.get('duration', 0)),
                equity_symbols=self.stats['instrument_type_stats'].get('EQ', 0),
                futures_symbols=self.stats['instrument_type_stats'].get('FUT', 0),
                options_symbols=(
                    self.stats['instrument_type_stats'].get('CE', 0) + 
                    self.stats['instrument_type_stats'].get('PE', 0)
                ),
                index_symbols=self.stats['instrument_type_stats'].get('INDEX', 0),
                minute_data_records=self.stats['minute_data_records'],
                daily_data_records=self.stats['daily_data_records'],
                created_at=datetime.now()
            )
            
            await self.questdb_client.insert_fetch_summary(summary)
            
        except Exception as e:
            logger.error(f"Error inserting fetch summary: {e}")
    
    async def _log_final_statistics(self):
        """Log comprehensive final statistics"""
        
        logger.info("📊 Final Statistics:")
        logger.info(f"  • Total Symbols: {self.stats['total_symbols']:,}")
        logger.info(f"  • Successful: {self.stats['successful_symbols']:,}")
        logger.info(f"  • Failed: {self.stats['failed_symbols']:,}")
        logger.info(f"  • Success Rate: {self.stats['success_rate']:.1f}%")
        logger.info(f"  • Total Records: {self.stats['total_records']:,}")
        logger.info(f"  • Processing Time: {self.stats.get('duration', 0):.1f} minutes")
        
        # Log performance metrics
        perf_summary = self.performance_monitor.get_metrics_summary(minutes=60)
        if perf_summary:
            logger.info("🔧 Performance Summary:")
            logger.info(f"  • Avg Memory: {perf_summary.get('avg_memory_mb', 0):.1f} MB")
            logger.info(f"  • Max Memory: {perf_summary.get('max_memory_mb', 0):.1f} MB")
            logger.info(f"  • Avg CPU: {perf_summary.get('avg_cpu_percent', 0):.1f}%")
            logger.info(f"  • Processing Rate: {perf_summary.get('processing_stats', {}).get('items_per_second', 0):.1f} symbols/sec")
    
<<<<<<< HEAD
    async def _log_detailed_completion_report(self):
        """Generate and log detailed completion report with all failed symbols"""
        
        # Generate detailed report
        detailed_report = self.metrics_tracker.generate_detailed_report()
        
        # Log the entire report
        logger.info("\n" + "=" * 80)
        logger.info("📋 DETAILED COMPLETION REPORT")
        logger.info("=" * 80)
        for line in detailed_report.split('\n'):
            logger.info(line)
        logger.info("=" * 80 + "\n")
        
        # Save failed symbols to JSON file
        try:
            log_dir = os.path.dirname(self.settings.log_file_path) if hasattr(self.settings, 'log_file_path') else 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            failed_symbols_file = os.path.join(log_dir, f"failed_symbols_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            self.metrics_tracker.save_failed_symbols_json(failed_symbols_file)
            logger.info(f"💾 Detailed failed symbols report saved to: {failed_symbols_file}")
        except Exception as e:
            logger.warning(f"Could not save failed symbols JSON file: {e}")
        
        # Log summary of failed symbols grouped by symbol
        if self.metrics_tracker.metrics.failed_symbols_list:
            logger.info("\n" + "=" * 80)
            logger.info("🚫 FAILED SYMBOLS SUMMARY (Grouped by Symbol)")
            logger.info("=" * 80)
            
            # Group failures by symbol
            from collections import defaultdict
            symbol_failures = defaultdict(list)
            for failed in self.metrics_tracker.metrics.failed_symbols_list:
                symbol_failures[failed.symbol].append(failed)
            
            # Log each symbol's failures
            for symbol, failures in sorted(symbol_failures.items()):
                logger.info(f"\n❌ {symbol} ({failures[0].exchange}, {failures[0].instrument_type}):")
                logger.info(f"   Total Failures: {len(failures)}")
                for failure in failures:
                    logger.info(f"   • Timeframe: {failure.timeframe} | Date Range: {failure.date_range}")
                    logger.info(f"     Reason: {failure.failure_reason.value} | Error: {failure.error_message[:100]}")
            
            logger.info("\n" + "=" * 80)
    
=======
>>>>>>> 98cb17d (Fix historicalfetcher)
    async def _handle_interruption(self):
        """Handle graceful shutdown on interruption"""
        
        logger.info("🛑 Gracefully shutting down...")
        
        # Send interruption notification
        if self.notification_manager.is_configured():
            await self.notification_manager.send_custom_message(
                f"⏹️ OpenAlgo Historical data fetch interrupted\n"
                f"📊 Processed {self.stats['processed_symbols']:,}/{self.stats['total_symbols']:,} symbols\n"
                f"💾 Saved {self.stats['total_records']:,} records",
                channels=['telegram']
            )
    
    async def _handle_fatal_error(self, error: Exception):
        """Handle fatal errors with comprehensive error reporting"""
        
        error_info = NotificationFormatter.create_error_summary(
            error=error,
            context={
                'timestamp': datetime.now().isoformat(),
                'processing_time': self._get_elapsed_time()
            },
            processed_items=self.stats['processed_symbols'],
            saved_records=self.stats['total_records']
        )
        
        # Send error notification
        await self.notification_manager.send_error_notification(error_info)
        
        # Log error context
        await self.async_logger.log_error_with_context(
            error=error,
            context={
                'stats': self.stats,
                'settings': {
                    'total_symbols': self.stats['total_symbols'],
                    'batch_size': self.settings.batch_size,
                    'max_concurrent': self.settings.max_concurrent_requests
                }
            },
            operation="openalgo_historical_data_fetch"
        )
        
        # Force flush logs to ensure error is written to file
        await self.async_logger.flush_logs()
        self.async_logger.force_sync_flush()
    
    async def _cleanup_components(self):
        """Cleanup all components and resources"""
        
        logger.info("🧹 Cleaning up OpenAlgo components...")
        
        try:
            # Stop performance monitoring
            await self.performance_monitor.stop_monitoring()
            
            # Cleanup fetcher
            await self.zerodha_fetcher.cleanup()
            
            # Cleanup database client
            await self.questdb_client.cleanup()
            
            logger.info("✅ OpenAlgo cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _get_elapsed_time(self) -> str:
        """Get elapsed time as formatted string"""
        
        if not self.stats['start_time']:
            return "N/A"
        
        elapsed = datetime.now() - self.stats['start_time']
        return NotificationFormatter.format_duration(elapsed.total_seconds())
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for logging"""
        
        system_info = self.performance_monitor.get_system_info()
        system_info.update({
            'settings': {
                'batch_size': self.settings.batch_size,
                'max_concurrent_requests': self.settings.max_concurrent_requests,
                'api_requests_per_second': self.settings.api_requests_per_second,
                'historical_days_limit': self.settings.historical_days_limit,
                'enabled_timeframes': self.settings.enabled_timeframes,
                'enabled_instrument_types': self.settings.enabled_instrument_types,
                'enabled_exchanges': self.settings.enabled_exchanges
            }
        })
        
        return system_info

async def main():
    """Main entry point"""
    
    try:
        fetcher = OpenAlgoHistoricalDataFetcher()
        await fetcher.run()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Process interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"💥 Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        
        # Force flush logs before exit
        try:
            import time
            time.sleep(0.2)  # Allow loguru to flush
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    # Set up event loop policy for Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())
