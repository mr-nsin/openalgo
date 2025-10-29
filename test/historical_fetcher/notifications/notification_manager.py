"""
Notification Manager

Unified interface for managing multiple notification channels (Telegram, Email).
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Optional

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from utils.logging import get_logger
from config.openalgo_settings import OpenAlgoSettings
from notifications.telegram_notifier import TelegramNotifier
from notifications.email_notifier import EmailNotifier

logger = get_logger(__name__)

class NotificationManager:
    """
    Manages multiple notification channels with unified interface
    """
    
    def __init__(self, settings: OpenAlgoSettings):
        """
        Initialize notification manager
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Initialize notifiers based on configuration
        self.telegram_notifier = None
        self.email_notifier = None
        
        # Setup Telegram notifier if configured
        if settings.telegram_bot_token and settings.telegram_chat_ids:
            self.telegram_notifier = TelegramNotifier(
                bot_token=settings.telegram_bot_token,
                chat_ids=settings.telegram_chat_ids
            )
            logger.info(f"Telegram notifier initialized with {len(settings.telegram_chat_ids)} chat(s)")
        
        # Setup Email notifier if configured
        if (settings.smtp_host and settings.smtp_username and 
            settings.smtp_password and settings.email_recipients):
            self.email_notifier = EmailNotifier(
                smtp_host=settings.smtp_host,
                smtp_port=settings.smtp_port,
                username=settings.smtp_username,
                password=settings.smtp_password,
                recipients=settings.email_recipients
            )
            logger.info(f"Email notifier initialized with {len(settings.email_recipients)} recipient(s)")
        
        # Check if any notifiers are configured
        if not self.telegram_notifier and not self.email_notifier:
            logger.warning("No notification channels configured")
    
    async def send_success_notification(self, stats: Dict[str, Any]):
        """Send success notification via all configured channels"""
        
        logger.info("Sending success notification")
        
        tasks = []
        
        if self.telegram_notifier:
            tasks.append(self._safe_notify(
                self.telegram_notifier.send_success_notification,
                stats,
                "Telegram"
            ))
        
        if self.email_notifier:
            tasks.append(self._safe_notify(
                self.email_notifier.send_success_notification,
                stats,
                "Email"
            ))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.warning("No notification channels available for success notification")
    
    async def send_error_notification(self, error_info: Dict[str, Any]):
        """Send error notification via all configured channels"""
        
        logger.info("Sending error notification")
        
        tasks = []
        
        if self.telegram_notifier:
            tasks.append(self._safe_notify(
                self.telegram_notifier.send_error_notification,
                error_info,
                "Telegram"
            ))
        
        if self.email_notifier:
            tasks.append(self._safe_notify(
                self.email_notifier.send_error_notification,
                error_info,
                "Email"
            ))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.warning("No notification channels available for error notification")
    
    async def send_progress_notification(self, progress_info: Dict[str, Any]):
        """Send progress notification (primarily via Telegram for real-time updates)"""
        
        # Only send progress notifications via Telegram to avoid email spam
        if self.telegram_notifier:
            await self._safe_notify(
                self.telegram_notifier.send_progress_notification,
                progress_info,
                "Telegram"
            )
        else:
            logger.debug("No Telegram notifier available for progress notification")
    
    async def send_warning_notification(self, warning_info: Dict[str, Any]):
        """Send warning notification via all configured channels"""
        
        logger.info("Sending warning notification")
        
        tasks = []
        
        if self.telegram_notifier:
            tasks.append(self._safe_notify(
                self.telegram_notifier.send_warning_notification,
                warning_info,
                "Telegram"
            ))
        
        if self.email_notifier:
            tasks.append(self._safe_notify(
                self.email_notifier.send_warning_notification,
                warning_info,
                "Email"
            ))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.warning("No notification channels available for warning notification")
    
    async def send_summary_report(self, stats: Dict[str, Any], detailed_data: Optional[Dict] = None):
        """Send detailed summary report (primarily via Email with attachments)"""
        
        logger.info("Sending summary report")
        
        # Send detailed report via email if available
        if self.email_notifier:
            await self._safe_notify(
                self.email_notifier.send_summary_report,
                stats,
                "Email",
                attachment_data=detailed_data
            )
        
        # Send basic summary via Telegram
        if self.telegram_notifier:
            await self._safe_notify(
                self.telegram_notifier.send_success_notification,
                stats,
                "Telegram"
            )
        
        if not self.email_notifier and not self.telegram_notifier:
            logger.warning("No notification channels available for summary report")
    
    async def send_custom_message(
        self,
        message: str,
        channels: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Send custom message via specified channels
        
        Args:
            message: Message to send
            channels: List of channels ('telegram', 'email') or None for all
            **kwargs: Additional parameters for specific notifiers
        """
        
        if channels is None:
            channels = ['telegram', 'email']
        
        tasks = []
        
        if 'telegram' in channels and self.telegram_notifier:
            tasks.append(self._safe_notify(
                self.telegram_notifier.send_custom_message,
                message,
                "Telegram",
                **kwargs
            ))
        
        if 'email' in channels and self.email_notifier:
            # For email, we need to format as both HTML and text
            subject = kwargs.get('subject', 'Custom Notification')
            html_content = f"<html><body><pre>{message}</pre></body></html>"
            
            tasks.append(self._safe_notify(
                self.email_notifier._send_email,
                subject,
                "Email",
                html_content=html_content,
                text_content=message
            ))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def test_all_connections(self) -> Dict[str, bool]:
        """Test all notification channel connections"""
        
        results = {}
        
        if self.telegram_notifier:
            try:
                results['telegram'] = await self.telegram_notifier.test_connection()
            except Exception as e:
                logger.error(f"Telegram connection test failed: {e}")
                results['telegram'] = False
        
        if self.email_notifier:
            try:
                results['email'] = await self.email_notifier.test_connection()
            except Exception as e:
                logger.error(f"Email connection test failed: {e}")
                results['email'] = False
        
        return results
    
    async def _safe_notify(self, notify_func, *args, channel_name: str, **kwargs):
        """Safely execute notification function with error handling"""
        
        try:
            if asyncio.iscoroutinefunction(notify_func):
                await notify_func(*args, **kwargs)
            else:
                notify_func(*args, **kwargs)
            
            logger.debug(f"{channel_name} notification sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending {channel_name} notification: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from all notification channels"""
        
        stats = {
            'telegram_enabled': self.telegram_notifier is not None,
            'email_enabled': self.email_notifier is not None,
            'total_channels': sum([
                1 for notifier in [self.telegram_notifier, self.email_notifier] 
                if notifier is not None
            ])
        }
        
        if self.telegram_notifier:
            stats['telegram_stats'] = self.telegram_notifier.get_statistics()
        
        if self.email_notifier:
            stats['email_stats'] = self.email_notifier.get_statistics()
        
        return stats
    
    def is_configured(self) -> bool:
        """Check if any notification channels are configured"""
        return self.telegram_notifier is not None or self.email_notifier is not None
    
    def get_configured_channels(self) -> List[str]:
        """Get list of configured notification channels"""
        channels = []
        
        if self.telegram_notifier:
            channels.append('telegram')
        
        if self.email_notifier:
            channels.append('email')
        
        return channels

# Utility functions for creating notification content
class NotificationFormatter:
    """Utility class for formatting notification content"""
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to human-readable string"""
        
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    @staticmethod
    def format_number(number: int) -> str:
        """Format number with thousands separators"""
        return f"{number:,}"
    
    @staticmethod
    def format_percentage(value: float) -> str:
        """Format percentage with one decimal place"""
        return f"{value:.1f}%"
    
    @staticmethod
    def create_progress_summary(processed: int, total: int, current_item: str) -> Dict[str, Any]:
        """Create standardized progress summary"""
        
        percentage = (processed / total * 100) if total > 0 else 0
        
        return {
            'processed': processed,
            'total': total,
            'percentage': percentage,
            'current_item': current_item,
            'remaining': total - processed
        }
    
    @staticmethod
    def create_error_summary(
        error: Exception,
        context: Dict[str, Any],
        processed_items: int = 0,
        saved_records: int = 0
    ) -> Dict[str, Any]:
        """Create standardized error summary"""
        
        return {
            'error_type': type(error).__name__,
            'message': str(error),
            'failed_at': context.get('timestamp', 'Unknown'),
            'processed_symbols': processed_items,
            'saved_records': saved_records,
            'processing_time': context.get('processing_time', 'Unknown'),
            'context': context
        }
