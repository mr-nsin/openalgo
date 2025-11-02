"""
Telegram Notifier

Sends notifications via Telegram Bot API with rich formatting and error handling.
"""

import asyncio
import aiohttp
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from historicalfetcher.utils.async_logger import get_async_logger

_async_logger = get_async_logger()
logger = _async_logger.get_logger()

# Logger is imported from loguru above

class TelegramNotifier:
    """Handles Telegram notifications with rich formatting"""
    
    def __init__(self, bot_token: str, chat_ids: List[str]):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token
            chat_ids: List of chat IDs to send notifications to
        """
        self.bot_token = bot_token
        self.chat_ids = chat_ids
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Statistics
        self.stats = {
            'total_sent': 0,
            'successful_sent': 0,
            'failed_sent': 0,
            'last_error': None
        }
    
    async def send_success_notification(self, stats: Dict[str, Any]):
        """Send success notification with comprehensive statistics"""
        
        message = self._format_success_message(stats)
        await self._send_message(message, parse_mode="Markdown")
    
    async def send_error_notification(self, error_info: Dict[str, Any]):
        """Send error notification with context"""
        
        message = self._format_error_message(error_info)
        await self._send_message(message, parse_mode="Markdown")
    
    async def send_progress_notification(self, progress_info: Dict[str, Any]):
        """Send progress notification (for long-running operations)"""
        
        message = self._format_progress_message(progress_info)
        await self._send_message(message, parse_mode="Markdown")
    
    async def send_warning_notification(self, warning_info: Dict[str, Any]):
        """Send warning notification"""
        
        message = self._format_warning_message(warning_info)
        await self._send_message(message, parse_mode="Markdown")
    
    async def send_custom_message(self, message: str, parse_mode: str = "Markdown"):
        """Send custom message"""
        
        await self._send_message(message, parse_mode=parse_mode)
    
    def _format_success_message(self, stats: Dict[str, Any]) -> str:
        """Format success notification message"""
        
        duration = stats.get('duration', 'N/A')
        if isinstance(duration, (int, float)):
            duration = f"{duration:.1f} minutes"
        
        # Format completion time
        completed_at = stats.get('completed_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'))
        
        message = f"""
🎉 **Historical Data Fetch Completed Successfully**

📊 **Overall Statistics:**
• Total Symbols: {stats.get('total_symbols', 0):,}
• Successful: {stats.get('successful_symbols', 0):,}
• Failed: {stats.get('failed_symbols', 0):,}
• Success Rate: {stats.get('success_rate', 0):.1f}%
• Total Records: {stats.get('total_records', 0):,}
• Processing Time: {duration}

📈 **Instrument Breakdown:**
{self._format_instrument_stats(stats.get('instrument_type_stats', {}))}

📊 **Timeframe Breakdown:**
{self._format_timeframe_stats(stats.get('timeframe_stats', {}))}

🏢 **Exchange Breakdown:**
{self._format_exchange_stats(stats.get('exchange_stats', {}))}

🕒 **Completed At:** {completed_at}

✅ All data successfully stored in QuestDB and ready for analysis.
        """.strip()
        
        return message
    
    def _format_error_message(self, error_info: Dict[str, Any]) -> str:
        """Format error notification message"""
        
        message = f"""
❌ **Historical Data Fetch Failed**

🚨 **Error Details:**
• Type: {error_info.get('error_type', 'Unknown')}
• Message: {error_info.get('message', 'No details available')}
• Failed At: {error_info.get('failed_at', 'N/A')}

📊 **Partial Results:**
• Symbols Processed: {error_info.get('processed_symbols', 0):,}
• Records Saved: {error_info.get('saved_records', 0):,}
• Processing Time: {error_info.get('processing_time', 'N/A')}

🔧 **Next Steps:**
• Check application logs for detailed error information
• Verify API credentials and network connectivity
• System will automatically retry on next scheduled run

📋 **Support:** Check logs at `logs/historical_fetcher.log`
        """.strip()
        
        return message
    
    def _format_progress_message(self, progress_info: Dict[str, Any]) -> str:
        """Format progress notification message"""
        
        processed = progress_info.get('processed', 0)
        total = progress_info.get('total', 0)
        percentage = (processed / total * 100) if total > 0 else 0
        current_symbol = progress_info.get('current_symbol', 'N/A')
        
        # Progress bar
        progress_bar = self._create_progress_bar(percentage)
        
        message = f"""
🔄 **Historical Data Fetch Progress**

{progress_bar} {percentage:.1f}%

📊 **Progress:** {processed:,}/{total:,}
🔍 **Current:** {current_symbol}
⏱️ **Elapsed:** {progress_info.get('elapsed_time', 'N/A')}
📈 **Records:** {progress_info.get('records_inserted', 0):,}

{self._get_progress_emoji(percentage)} Keep going!
        """.strip()
        
        return message
    
    def _format_warning_message(self, warning_info: Dict[str, Any]) -> str:
        """Format warning notification message"""
        
        message = f"""
⚠️ **Historical Data Fetch Warning**

🔍 **Warning Details:**
• Type: {warning_info.get('warning_type', 'Unknown')}
• Message: {warning_info.get('message', 'No details available')}
• Occurred At: {warning_info.get('occurred_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}

📊 **Current Status:**
• Symbols Processed: {warning_info.get('processed_symbols', 0):,}
• Records Inserted: {warning_info.get('records_inserted', 0):,}

ℹ️ **Action:** {warning_info.get('action', 'Continuing with processing')}
        """.strip()
        
        return message
    
    def _format_instrument_stats(self, instrument_stats: Dict[str, Any]) -> str:
        """Format instrument type statistics"""
        
        if not instrument_stats:
            return "• No instrument data available"
        
        lines = []
        for instrument_type, count in instrument_stats.items():
            emoji = self._get_instrument_emoji(instrument_type)
            lines.append(f"• {emoji} {instrument_type}: {count:,}")
        
        return "\n".join(lines)
    
    def _format_timeframe_stats(self, timeframe_stats: Dict[str, Any]) -> str:
        """Format timeframe statistics"""
        
        if not timeframe_stats:
            return "• No timeframe data available"
        
        lines = []
        for timeframe, count in timeframe_stats.items():
            emoji = self._get_timeframe_emoji(timeframe)
            lines.append(f"• {emoji} {timeframe}: {count:,} records")
        
        return "\n".join(lines)
    
    def _format_exchange_stats(self, exchange_stats: Dict[str, Any]) -> str:
        """Format exchange statistics"""
        
        if not exchange_stats:
            return "• No exchange data available"
        
        lines = []
        for exchange, count in exchange_stats.items():
            emoji = self._get_exchange_emoji(exchange)
            lines.append(f"• {emoji} {exchange}: {count:,}")
        
        return "\n".join(lines)
    
    def _create_progress_bar(self, percentage: float, length: int = 10) -> str:
        """Create a visual progress bar"""
        
        filled = int(length * percentage / 100)
        bar = "█" * filled + "░" * (length - filled)
        return f"[{bar}]"
    
    def _get_progress_emoji(self, percentage: float) -> str:
        """Get emoji based on progress percentage"""
        
        if percentage < 25:
            return "🚀"
        elif percentage < 50:
            return "⚡"
        elif percentage < 75:
            return "🔥"
        else:
            return "🏁"
    
    def _get_instrument_emoji(self, instrument_type: str) -> str:
        """Get emoji for instrument type"""
        
        emoji_map = {
            'EQ': '📈',
            'FUT': '📊',
            'CE': '📞',
            'PE': '📉',
            'INDEX': '🏛️'
        }
        return emoji_map.get(instrument_type, '📋')
    
    def _get_timeframe_emoji(self, timeframe: str) -> str:
        """Get emoji for timeframe"""
        
        if timeframe in ['1m', '3m', '5m']:
            return '⚡'
        elif timeframe in ['15m', '30m', '1h']:
            return '⏰'
        elif timeframe == 'D':
            return '📅'
        else:
            return '🕐'
    
    def _get_exchange_emoji(self, exchange: str) -> str:
        """Get emoji for exchange"""
        
        emoji_map = {
            'NSE': '🇮🇳',
            'BSE': '🏢',
            'NFO': '📈',
            'BFO': '📊',
            'NSE_INDEX': '🏛️',
            'BSE_INDEX': '🏛️',
            'MCX': '🥇'
        }
        return emoji_map.get(exchange, '🏪')
    
    async def _send_message(self, message: str, parse_mode: str = "Markdown"):
        """Send message to all configured chat IDs"""
        
        if not self.chat_ids:
            logger.warning("No Telegram chat IDs configured")
            return
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for chat_id in self.chat_ids:
                task = self._send_to_chat(session, chat_id, message, parse_mode)
                tasks.append(task)
            
            # Send to all chats concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                chat_id = self.chat_ids[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to send Telegram message to {chat_id}: {result}")
                    self.stats['failed_sent'] += 1
                    self.stats['last_error'] = str(result)
                else:
                    logger.info(f"Telegram message sent successfully to {chat_id}")
                    self.stats['successful_sent'] += 1
                
                self.stats['total_sent'] += 1
    
    async def _send_to_chat(
        self,
        session: aiohttp.ClientSession,
        chat_id: str,
        message: str,
        parse_mode: str
    ):
        """Send message to a specific chat"""
        
        url = f"{self.base_url}/sendMessage"
        
        # Split long messages if needed (Telegram limit is 4096 characters)
        if len(message) > 4000:
            messages = self._split_message(message, 4000)
        else:
            messages = [message]
        
        for msg in messages:
            data = {
                "chat_id": chat_id,
                "text": msg,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                response_data = await response.json()
                if not response_data.get('ok'):
                    raise Exception(f"Telegram API error: {response_data.get('description')}")
    
    def _split_message(self, message: str, max_length: int) -> List[str]:
        """Split long message into chunks"""
        
        if len(message) <= max_length:
            return [message]
        
        messages = []
        current_message = ""
        
        for line in message.split('\n'):
            if len(current_message + line + '\n') > max_length:
                if current_message:
                    messages.append(current_message.strip())
                    current_message = line + '\n'
                else:
                    # Line itself is too long, split it
                    while len(line) > max_length:
                        messages.append(line[:max_length])
                        line = line[max_length:]
                    current_message = line + '\n'
            else:
                current_message += line + '\n'
        
        if current_message:
            messages.append(current_message.strip())
        
        return messages
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics"""
        return self.stats.copy()
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/getMe"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('ok'):
                            bot_info = data.get('result', {})
                            logger.info(f"Telegram bot connected: {bot_info.get('username', 'Unknown')}")
                            return True
                    
                    error_text = await response.text()
                    logger.error(f"Telegram connection test failed: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Telegram connection test error: {e}")
            return False
