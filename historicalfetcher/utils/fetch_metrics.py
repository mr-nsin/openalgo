"""
Comprehensive metrics tracking system for historical data fetching.
Tracks success/failure rates, failed symbols, and generates detailed reports.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from loguru import logger

try:
    from historicalfetcher.models.data_models import SymbolInfo
    from historicalfetcher.config.openalgo_settings import TimeFrame
    from historicalfetcher.notifications.email_notifier import EmailNotifier
    from historicalfetcher.notifications.telegram_notifier import TelegramNotifier
except ImportError:
    from ..models.data_models import SymbolInfo
    from ..config.openalgo_settings import TimeFrame
    from ..notifications.email_notifier import EmailNotifier
    from ..notifications.telegram_notifier import TelegramNotifier


class FailureReason(Enum):
    """Enumeration of possible failure reasons"""
    RATE_LIMIT = "rate_limit"
    NO_DATA = "no_data"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    CONVERSION_ERROR = "conversion_error"
    DATABASE_ERROR = "database_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class FailedSymbolInfo:
    """Information about a failed symbol fetch"""
    symbol: str
    exchange: str
    instrument_type: str
    timeframe: str
    date_range: str
    failure_reason: FailureReason
    error_message: str
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0


@dataclass
class FetchMetrics:
    """Comprehensive metrics for fetch operations"""
    # Overall statistics
    total_symbols: int = 0
    successful_symbols: int = 0
    failed_symbols: int = 0
    total_timeframes: int = 0
    successful_timeframes: int = 0
    failed_timeframes: int = 0
    
    # Data statistics
    total_candles_fetched: int = 0
    total_records_inserted: int = 0
    total_tables_created: int = 0
    
    # Timing information
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Failed symbols tracking
    failed_symbols_list: List[FailedSymbolInfo] = field(default_factory=list)
    
    # Error breakdown
    error_breakdown: Dict[FailureReason, int] = field(default_factory=lambda: {reason: 0 for reason in FailureReason})
    
    # Exchange and instrument type breakdown
    exchange_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)  # {exchange: {success: x, failed: y}}
    instrument_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)  # {type: {success: x, failed: y}}
    timeframe_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)  # {timeframe: {success: x, failed: y}}
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate"""
        if self.total_symbols == 0:
            return 0.0
        return (self.successful_symbols / self.total_symbols) * 100
    
    def get_timeframe_success_rate(self) -> float:
        """Calculate timeframe-level success rate"""
        if self.total_timeframes == 0:
            return 0.0
        return (self.successful_timeframes / self.total_timeframes) * 100
    
    def get_duration(self) -> timedelta:
        """Get total duration of fetch operation"""
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    def get_duration_str(self) -> str:
        """Get formatted duration string"""
        duration = self.get_duration()
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


class FetchMetricsTracker:
    """Tracks and manages fetch metrics throughout the operation"""
    
    def __init__(self, email_notifier: Optional[EmailNotifier] = None, 
                 telegram_notifier: Optional[TelegramNotifier] = None):
        self.metrics = FetchMetrics()
        self.email_notifier = email_notifier
        self.telegram_notifier = telegram_notifier
        
    def start_tracking(self, total_symbols: int):
        """Initialize tracking with total symbol count"""
        self.metrics.start_time = datetime.now()
        self.metrics.total_symbols = total_symbols
        logger.info(f"📊 Started tracking metrics for {total_symbols} symbols")
    
    def record_symbol_success(self, symbol_info: SymbolInfo, timeframe: TimeFrame, 
                            candles_count: int, records_inserted: int):
        """Record a successful symbol fetch"""
        self.metrics.successful_symbols += 1
        self.metrics.successful_timeframes += 1
        self.metrics.total_candles_fetched += candles_count
        self.metrics.total_records_inserted += records_inserted
        
        # Update exchange stats
        exchange = symbol_info.exchange
        if exchange not in self.metrics.exchange_stats:
            self.metrics.exchange_stats[exchange] = {"success": 0, "failed": 0}
        self.metrics.exchange_stats[exchange]["success"] += 1
        
        # Update instrument stats
        instrument = symbol_info.instrument_type
        if instrument not in self.metrics.instrument_stats:
            self.metrics.instrument_stats[instrument] = {"success": 0, "failed": 0}
        self.metrics.instrument_stats[instrument]["success"] += 1
        
        # Update timeframe stats
        tf_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
        if tf_str not in self.metrics.timeframe_stats:
            self.metrics.timeframe_stats[tf_str] = {"success": 0, "failed": 0}
        self.metrics.timeframe_stats[tf_str]["success"] += 1
        
        logger.debug(f"✅ Recorded success for {symbol_info.symbol} ({tf_str}): {candles_count} candles, {records_inserted} records")
    
    def record_symbol_failure(self, symbol_info: SymbolInfo, timeframe: TimeFrame,
                            date_range: str, failure_reason: FailureReason, 
                            error_message: str, retry_count: int = 0):
        """Record a failed symbol fetch"""
        self.metrics.failed_symbols += 1
        self.metrics.failed_timeframes += 1
        
        # Create failed symbol info
        tf_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
        failed_info = FailedSymbolInfo(
            symbol=symbol_info.symbol,
            exchange=symbol_info.exchange,
            instrument_type=symbol_info.instrument_type,
            timeframe=tf_str,
            date_range=date_range,
            failure_reason=failure_reason,
            error_message=error_message,
            retry_count=retry_count
        )
        
        self.metrics.failed_symbols_list.append(failed_info)
        self.metrics.error_breakdown[failure_reason] += 1
        
        # Update exchange stats
        exchange = symbol_info.exchange
        if exchange not in self.metrics.exchange_stats:
            self.metrics.exchange_stats[exchange] = {"success": 0, "failed": 0}
        self.metrics.exchange_stats[exchange]["failed"] += 1
        
        # Update instrument stats
        instrument = symbol_info.instrument_type
        if instrument not in self.metrics.instrument_stats:
            self.metrics.instrument_stats[instrument] = {"success": 0, "failed": 0}
        self.metrics.instrument_stats[instrument]["failed"] += 1
        
        # Update timeframe stats
        if tf_str not in self.metrics.timeframe_stats:
            self.metrics.timeframe_stats[tf_str] = {"success": 0, "failed": 0}
        self.metrics.timeframe_stats[tf_str]["failed"] += 1
        
        logger.warning(f"❌ Recorded failure for {symbol_info.symbol} ({tf_str}): {failure_reason.value} - {error_message}")
    
    def increment_total_timeframes(self):
        """Increment total timeframes counter"""
        self.metrics.total_timeframes += 1
    
    def record_table_created(self):
        """Record that a table was created"""
        self.metrics.total_tables_created += 1
    
    def finish_tracking(self):
        """Finalize tracking and prepare for reporting"""
        self.metrics.end_time = datetime.now()
        logger.info(f"📊 Finished tracking metrics. Duration: {self.metrics.get_duration_str()}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for quick overview"""
        return {
            "total_symbols": self.metrics.total_symbols,
            "successful_symbols": self.metrics.successful_symbols,
            "failed_symbols": self.metrics.failed_symbols,
            "success_rate": round(self.metrics.get_success_rate(), 2),
            "timeframe_success_rate": round(self.metrics.get_timeframe_success_rate(), 2),
            "total_candles": self.metrics.total_candles_fetched,
            "total_records": self.metrics.total_records_inserted,
            "tables_created": self.metrics.total_tables_created,
            "duration": self.metrics.get_duration_str(),
            "top_failure_reasons": self._get_top_failure_reasons(5)
        }
    
    def _get_top_failure_reasons(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get top failure reasons sorted by count"""
        return sorted(
            [(reason.value, count) for reason, count in self.metrics.error_breakdown.items() if count > 0],
            key=lambda x: x[1],
            reverse=True
        )[:limit]
    
    def generate_detailed_report(self) -> str:
        """Generate a detailed text report"""
        report = []
        report.append("=" * 80)
        report.append("📊 HISTORICAL DATA FETCH REPORT")
        report.append("=" * 80)
        report.append(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"⏱️ Duration: {self.metrics.get_duration_str()}")
        report.append("")
        
        # Overall Statistics
        report.append("📈 OVERALL STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Symbols: {self.metrics.total_symbols}")
        report.append(f"Successful: {self.metrics.successful_symbols}")
        report.append(f"Failed: {self.metrics.failed_symbols}")
        report.append(f"Success Rate: {self.metrics.get_success_rate():.2f}%")
        report.append("")
        
        # Timeframe Statistics
        report.append("⏰ TIMEFRAME STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Timeframes: {self.metrics.total_timeframes}")
        report.append(f"Successful: {self.metrics.successful_timeframes}")
        report.append(f"Failed: {self.metrics.failed_timeframes}")
        report.append(f"Success Rate: {self.metrics.get_timeframe_success_rate():.2f}%")
        report.append("")
        
        # Data Statistics
        report.append("💾 DATA STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Candles Fetched: {self.metrics.total_candles_fetched:,}")
        report.append(f"Total Records Inserted: {self.metrics.total_records_inserted:,}")
        report.append(f"Tables Created: {self.metrics.total_tables_created}")
        report.append("")
        
        # Error Breakdown
        if self.metrics.failed_symbols > 0:
            report.append("❌ ERROR BREAKDOWN")
            report.append("-" * 40)
            for reason, count in self._get_top_failure_reasons():
                percentage = (count / self.metrics.failed_symbols) * 100
                report.append(f"{reason.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Exchange Breakdown
        if self.metrics.exchange_stats:
            report.append("🏢 EXCHANGE BREAKDOWN")
            report.append("-" * 40)
            for exchange, stats in self.metrics.exchange_stats.items():
                total = stats["success"] + stats["failed"]
                success_rate = (stats["success"] / total * 100) if total > 0 else 0
                report.append(f"{exchange}: {stats['success']}/{total} ({success_rate:.1f}% success)")
            report.append("")
        
        # Instrument Type Breakdown
        if self.metrics.instrument_stats:
            report.append("📋 INSTRUMENT TYPE BREAKDOWN")
            report.append("-" * 40)
            for instrument, stats in self.metrics.instrument_stats.items():
                total = stats["success"] + stats["failed"]
                success_rate = (stats["success"] / total * 100) if total > 0 else 0
                report.append(f"{instrument}: {stats['success']}/{total} ({success_rate:.1f}% success)")
            report.append("")
        
        # Timeframe Breakdown
        if self.metrics.timeframe_stats:
            report.append("⏱️ TIMEFRAME BREAKDOWN")
            report.append("-" * 40)
            for timeframe, stats in self.metrics.timeframe_stats.items():
                total = stats["success"] + stats["failed"]
                success_rate = (stats["success"] / total * 100) if total > 0 else 0
                report.append(f"{timeframe}: {stats['success']}/{total} ({success_rate:.1f}% success)")
            report.append("")
        
        # Failed Symbols (first 50)
        if self.metrics.failed_symbols_list:
            report.append("🚫 FAILED SYMBOLS (First 50)")
            report.append("-" * 120)
            report.append(f"{'Symbol':<15} {'Exchange':<10} {'Type':<6} {'Timeframe':<10} {'Date Range':<25} {'Reason':<15} {'Error'}")
            report.append("-" * 120)
            
            for failed in self.metrics.failed_symbols_list[:50]:
                error_short = failed.error_message[:30] + "..." if len(failed.error_message) > 30 else failed.error_message
                date_range_short = failed.date_range[:23] + "..." if len(failed.date_range) > 23 else failed.date_range
                report.append(f"{failed.symbol:<15} {failed.exchange:<10} {failed.instrument_type:<6} "
                           f"{failed.timeframe:<10} {date_range_short:<25} {failed.failure_reason.value:<15} {error_short}")
            
            if len(self.metrics.failed_symbols_list) > 50:
                report.append(f"... and {len(self.metrics.failed_symbols_list) - 50} more failed symbols")
            report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def save_failed_symbols_json(self, filepath: str = "failed_symbols.json"):
        """Save failed symbols to JSON file for further analysis"""
        failed_data = []
        for failed in self.metrics.failed_symbols_list:
            failed_data.append({
                "symbol": failed.symbol,
                "exchange": failed.exchange,
                "instrument_type": failed.instrument_type,
                "timeframe": failed.timeframe,
                "date_range": failed.date_range,
                "failure_reason": failed.failure_reason.value,
                "error_message": failed.error_message,
                "timestamp": failed.timestamp.isoformat(),
                "retry_count": failed.retry_count
            })
        
        with open(filepath, 'w') as f:
            json.dump({
                "report_generated": datetime.now().isoformat(),
                "summary": self.get_summary_stats(),
                "failed_symbols": failed_data
            }, f, indent=2)
        
        logger.info(f"💾 Saved failed symbols report to {filepath}")
    
    async def send_reports(self, include_detailed_email: bool = True, 
                          include_telegram_summary: bool = True):
        """Send reports via email and telegram"""
        try:
            # Send detailed email report
            if include_detailed_email and self.email_notifier:
                await self._send_email_report()
            
            # Send telegram summary
            if include_telegram_summary and self.telegram_notifier:
                await self._send_telegram_summary()
                
        except Exception as e:
            logger.error(f"Error sending reports: {e}")
    
    async def _send_email_report(self):
        """Send detailed email report"""
        try:
            subject = f"Historical Data Fetch Report - {datetime.now().strftime('%Y-%m-%d')}"
            body = self.generate_detailed_report()
            
            # Save failed symbols JSON as attachment
            json_file = f"failed_symbols_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.save_failed_symbols_json(json_file)
            
            success = await self.email_notifier.send_email(
                subject=subject,
                body=body,
                attachments=[json_file] if Path(json_file).exists() else None
            )
            
            if success:
                logger.info("📧 Email report sent successfully")
            else:
                logger.error("❌ Failed to send email report")
                
        except Exception as e:
            logger.error(f"Error sending email report: {e}")
    
    async def _send_telegram_summary(self):
        """Send concise telegram summary"""
        try:
            summary = self.get_summary_stats()
            
            message = f"""
📊 *Historical Data Fetch Summary*

✅ *Success Rate:* {summary['success_rate']}%
📈 *Symbols:* {summary['successful_symbols']}/{summary['total_symbols']}
⏰ *Timeframes:* {summary['successful_symbols']}/{summary['total_timeframes']} ({summary['timeframe_success_rate']}%)

💾 *Data Collected:*
• Candles: {summary['total_candles']:,}
• Records: {summary['total_records']:,}
• Tables: {summary['tables_created']}

⏱️ *Duration:* {summary['duration']}

❌ *Failed Symbols:* {summary['failed_symbols']}
"""
            
            if summary['failed_symbols'] > 0:
                message += f"\n🔍 *Top Failures:*\n"
                for reason, count in summary['top_failure_reasons'][:3]:
                    message += f"• {reason.replace('_', ' ').title()}: {count}\n"
            
            success = await self.telegram_notifier.send_message(message)
            
            if success:
                logger.info("📱 Telegram summary sent successfully")
            else:
                logger.error("❌ Failed to send telegram summary")
                
        except Exception as e:
            logger.error(f"Error sending telegram summary: {e}")


# Helper function to determine failure reason from error message
def classify_error(error_message: str) -> FailureReason:
    """Classify error message into failure reason category"""
    error_lower = error_message.lower()
    
    if "too many requests" in error_lower or "rate limit" in error_lower or "429" in error_lower:
        return FailureReason.RATE_LIMIT
    elif "no data" in error_lower or "empty" in error_lower or "no records" in error_lower:
        return FailureReason.NO_DATA
    elif "timeout" in error_lower or "connection" in error_lower or "network" in error_lower:
        return FailureReason.NETWORK_ERROR
    elif "conversion" in error_lower or "dataframe" in error_lower or "candle" in error_lower:
        return FailureReason.CONVERSION_ERROR
    elif "database" in error_lower or "questdb" in error_lower or "table" in error_lower:
        return FailureReason.DATABASE_ERROR
    elif "api" in error_lower:
        return FailureReason.API_ERROR
    else:
        return FailureReason.UNKNOWN_ERROR
