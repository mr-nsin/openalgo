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
from typing import Dict, Any, List
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
            
            # Process each instrument type in priority order
            # Priority: INDEX (especially NIFTY, BANKNIFTY, SENSEX) first, then EQ, FUT, options
            priority_order = ['INDEX', 'EQ', 'FUT', 'CE', 'PE']
            
            # Process INDEX symbols first (already sorted by symbol manager with priority indices first)
            if 'INDEX' in symbols_by_type and symbols_by_type['INDEX']:
                index_symbols = symbols_by_type['INDEX']
                
                # Log major indices that will be processed first
                major_indices = ['NIFTY', 'BANKNIFTY', 'SENSEX', 'NIFTY50', 'NIFTY 50', 'NIFTY BANK', 'NIFTY IT']
                major_found = [s for s in index_symbols[:10] if any(major in s.symbol.upper() for major in major_indices)]
                if major_found:
                    logger.info(f"📊 Major indices to be processed first: {', '.join([s.symbol for s in major_found])}")
                
                # Separate NSE_INDEX and other INDEX symbols
                nse_index_symbols = [s for s in index_symbols if s.exchange == 'NSE_INDEX']
                other_index_symbols = [s for s in index_symbols if s.exchange != 'NSE_INDEX']
                
                # Process NSE_INDEX first (includes NIFTY, BANKNIFTY, etc.)
                if nse_index_symbols:
                    logger.info(f"🔄 Processing NSE_INDEX symbols first ({len(nse_index_symbols):,} symbols)")
                    logger.info(f"   First 5: {', '.join([s.symbol for s in nse_index_symbols[:5]])}")
                    await self._process_instrument_type('INDEX', nse_index_symbols)
                
                # Then process other INDEX symbols (BSE_INDEX, etc.)
                if other_index_symbols:
                    logger.info(f"🔄 Processing other INDEX symbols ({len(other_index_symbols):,} symbols)")
                    await self._process_instrument_type('INDEX', other_index_symbols)
            
            # Process remaining instrument types in priority order
            for instrument_type in priority_order:
                if instrument_type == 'INDEX':  # Already processed above
                    continue
                    
                if instrument_type in symbols_by_type and symbols_by_type[instrument_type]:
                    symbols = symbols_by_type[instrument_type]
                    logger.info(f"🔄 Processing {instrument_type} symbols ({len(symbols):,} symbols)")
                    await self._process_instrument_type(instrument_type, symbols)
            
            # Process any remaining instrument types not in priority list
            for instrument_type, symbols in symbols_by_type.items():
                if instrument_type not in priority_order and instrument_type != 'INDEX':
                    if symbols:
                        logger.info(f"🔄 Processing {instrument_type} symbols ({len(symbols):,} symbols)")
                        await self._process_instrument_type(instrument_type, symbols)
            
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
    
    async def _process_single_symbol(self, symbol_info) -> int:
        """Process historical data for a single symbol across all timeframes"""
        
        total_records = 0
        timeframes = self.settings.get_timeframe_objects()
        
        # Process timeframes in priority order (daily first, then intraday)
        ordered_timeframes = self._get_processing_order(timeframes)
        
        for timeframe in ordered_timeframes:
            try:
                async with AsyncTimer(self.performance_monitor, f'fetch_{timeframe.value}'):
                    # Determine date range
                    from_date, to_date = await self._get_date_range(symbol_info, timeframe)
                    
                    if not from_date or not to_date:
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
                            records_inserted = await self.questdb_client.upsert_historical_data_with_indicators(
                                symbol_info,
                                timeframe,
                                candles
                            )
                        
                        total_records += records_inserted
                        
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
                        
                        logger.debug(
                            f"✅ {symbol_info.symbol} ({timeframe.value}): {records_inserted:,} records"
                        )
                    else:
                        logger.debug(f"⚠️ No data found for {symbol_info.symbol} ({timeframe.value})")
                        
                        await self.questdb_client.update_fetch_status(
                            symbol_info,
                            timeframe.value,
                            'no_data',
                            0
                        )
            
            except Exception as e:
                logger.exception(f"❌ Error fetching {symbol_info.symbol} ({timeframe.value}): {e}")
                
                await self.questdb_client.update_fetch_status(
                    symbol_info,
                    timeframe.value,
                    'error',
                    0
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
        
        # Ensure start_date is not in the future
        if start_date > end_date:
            start_date = end_date - timedelta(days=self.settings.historical_days_limit)
        
        # Log the date range for debugging
        logger.info(f"📅 Date range for {symbol_info.symbol} ({timeframe.value}): {start_date.date()} to {end_date.date()}")
        
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
        
        # Insert fetch summary into database
        await self._insert_fetch_summary()
        
        # Log final statistics
        await self._log_final_statistics()
    
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
