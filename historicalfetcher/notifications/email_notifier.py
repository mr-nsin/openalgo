"""
Email Notifier

Sends email notifications with HTML formatting and attachment support.
"""

import asyncio
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from historicalfetcher.utils.async_logger import get_async_logger

_async_logger = get_async_logger()
logger = _async_logger.get_logger()

# Logger is imported from loguru above

class EmailNotifier:
    """Handles email notifications with HTML formatting"""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        recipients: List[str]
    ):
        """
        Initialize email notifier
        
        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            recipients: List of email addresses to send notifications to
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
        
        # Statistics
        self.stats = {
            'total_sent': 0,
            'successful_sent': 0,
            'failed_sent': 0,
            'last_error': None
        }
    
    async def send_success_notification(self, stats: Dict[str, Any]):
        """Send success notification email"""
        
        subject = "✅ Historical Data Fetch Completed Successfully"
        html_content = self._format_success_html(stats)
        text_content = self._format_success_text(stats)
        
        await self._send_email(subject, html_content, text_content)
    
    async def send_error_notification(self, error_info: Dict[str, Any]):
        """Send error notification email"""
        
        subject = "❌ Historical Data Fetch Failed"
        html_content = self._format_error_html(error_info)
        text_content = self._format_error_text(error_info)
        
        await self._send_email(subject, html_content, text_content)
    
    async def send_warning_notification(self, warning_info: Dict[str, Any]):
        """Send warning notification email"""
        
        subject = "⚠️ Historical Data Fetch Warning"
        html_content = self._format_warning_html(warning_info)
        text_content = self._format_warning_text(warning_info)
        
        await self._send_email(subject, html_content, text_content)
    
    async def send_summary_report(self, stats: Dict[str, Any], attachment_data: Optional[Dict] = None):
        """Send detailed summary report with optional attachment"""
        
        subject = f"📊 Historical Data Fetch Report - {datetime.now().strftime('%Y-%m-%d')}"
        html_content = self._format_summary_html(stats)
        text_content = self._format_summary_text(stats)
        
        attachments = []
        if attachment_data:
            # Create JSON attachment with detailed stats
            json_data = json.dumps(attachment_data, indent=2, default=str)
            attachments.append({
                'filename': f'fetch_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                'content': json_data.encode('utf-8'),
                'content_type': 'application/json'
            })
        
        await self._send_email(subject, html_content, text_content, attachments)
    
    def _format_success_html(self, stats: Dict[str, Any]) -> str:
        """Format success notification as HTML"""
        
        duration = stats.get('duration', 'N/A')
        if isinstance(duration, (int, float)):
            duration = f"{duration:.1f} minutes"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .stats-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .stats-table th {{ background-color: #f2f2f2; }}
                .success {{ color: #4CAF50; font-weight: bold; }}
                .footer {{ background-color: #f9f9f9; padding: 15px; text-align: center; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>✅ Historical Data Fetch Completed Successfully</h1>
            </div>
            
            <div class="content">
                <h2>📊 Overall Statistics</h2>
                <table class="stats-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Symbols</td><td>{stats.get('total_symbols', 0):,}</td></tr>
                    <tr><td>Successful</td><td class="success">{stats.get('successful_symbols', 0):,}</td></tr>
                    <tr><td>Failed</td><td>{stats.get('failed_symbols', 0):,}</td></tr>
                    <tr><td>Success Rate</td><td class="success">{stats.get('success_rate', 0):.1f}%</td></tr>
                    <tr><td>Total Records</td><td>{stats.get('total_records', 0):,}</td></tr>
                    <tr><td>Processing Time</td><td>{duration}</td></tr>
                </table>
                
                <h2>📈 Instrument Breakdown</h2>
                {self._format_instrument_table_html(stats.get('instrument_type_stats', {}))}
                
                <h2>📊 Timeframe Breakdown</h2>
                {self._format_timeframe_table_html(stats.get('timeframe_stats', {}))}
                
                <h2>🏢 Exchange Breakdown</h2>
                {self._format_exchange_table_html(stats.get('exchange_stats', {}))}
                
                <p><strong>Completed At:</strong> {stats.get('completed_at', 'N/A')}</p>
                <p class="success">✅ All data successfully stored in QuestDB and ready for analysis.</p>
            </div>
            
            <div class="footer">
                <p>Historical Data Fetcher - OpenAlgo Platform</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_success_text(self, stats: Dict[str, Any]) -> str:
        """Format success notification as plain text"""
        
        duration = stats.get('duration', 'N/A')
        if isinstance(duration, (int, float)):
            duration = f"{duration:.1f} minutes"
        
        text = f"""
Historical Data Fetch Completed Successfully
==========================================

Overall Statistics:
- Total Symbols: {stats.get('total_symbols', 0):,}
- Successful: {stats.get('successful_symbols', 0):,}
- Failed: {stats.get('failed_symbols', 0):,}
- Success Rate: {stats.get('success_rate', 0):.1f}%
- Total Records: {stats.get('total_records', 0):,}
- Processing Time: {duration}

Instrument Breakdown:
{self._format_dict_as_text(stats.get('instrument_type_stats', {}))}

Timeframe Breakdown:
{self._format_dict_as_text(stats.get('timeframe_stats', {}))}

Exchange Breakdown:
{self._format_dict_as_text(stats.get('exchange_stats', {}))}

Completed At: {stats.get('completed_at', 'N/A')}

All data successfully stored in QuestDB and ready for analysis.
        """.strip()
        
        return text
    
    def _format_error_html(self, error_info: Dict[str, Any]) -> str:
        """Format error notification as HTML"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f44336; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .error {{ color: #f44336; font-weight: bold; }}
                .warning {{ color: #ff9800; }}
                .info-box {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0; }}
                .footer {{ background-color: #f9f9f9; padding: 15px; text-align: center; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>❌ Historical Data Fetch Failed</h1>
            </div>
            
            <div class="content">
                <h2>🚨 Error Details</h2>
                <div class="info-box">
                    <p><strong>Type:</strong> <span class="error">{error_info.get('error_type', 'Unknown')}</span></p>
                    <p><strong>Message:</strong> {error_info.get('message', 'No details available')}</p>
                    <p><strong>Failed At:</strong> {error_info.get('failed_at', 'N/A')}</p>
                </div>
                
                <h2>📊 Partial Results</h2>
                <ul>
                    <li>Symbols Processed: {error_info.get('processed_symbols', 0):,}</li>
                    <li>Records Saved: {error_info.get('saved_records', 0):,}</li>
                    <li>Processing Time: {error_info.get('processing_time', 'N/A')}</li>
                </ul>
                
                <h2>🔧 Next Steps</h2>
                <ul>
                    <li>Check application logs for detailed error information</li>
                    <li>Verify API credentials and network connectivity</li>
                    <li>System will automatically retry on next scheduled run</li>
                </ul>
                
                <div class="info-box">
                    <p><strong>Support:</strong> Check logs at <code>logs/historical_fetcher.log</code></p>
                </div>
            </div>
            
            <div class="footer">
                <p>Historical Data Fetcher - OpenAlgo Platform</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_error_text(self, error_info: Dict[str, Any]) -> str:
        """Format error notification as plain text"""
        
        text = f"""
Historical Data Fetch Failed
===========================

Error Details:
- Type: {error_info.get('error_type', 'Unknown')}
- Message: {error_info.get('message', 'No details available')}
- Failed At: {error_info.get('failed_at', 'N/A')}

Partial Results:
- Symbols Processed: {error_info.get('processed_symbols', 0):,}
- Records Saved: {error_info.get('saved_records', 0):,}
- Processing Time: {error_info.get('processing_time', 'N/A')}

Next Steps:
- Check application logs for detailed error information
- Verify API credentials and network connectivity
- System will automatically retry on next scheduled run

Support: Check logs at logs/historical_fetcher.log
        """.strip()
        
        return text
    
    def _format_warning_html(self, warning_info: Dict[str, Any]) -> str:
        """Format warning notification as HTML"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #ff9800; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .warning {{ color: #ff9800; font-weight: bold; }}
                .info-box {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ff9800; margin: 20px 0; }}
                .footer {{ background-color: #f9f9f9; padding: 15px; text-align: center; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>⚠️ Historical Data Fetch Warning</h1>
            </div>
            
            <div class="content">
                <h2>🔍 Warning Details</h2>
                <div class="info-box">
                    <p><strong>Type:</strong> <span class="warning">{warning_info.get('warning_type', 'Unknown')}</span></p>
                    <p><strong>Message:</strong> {warning_info.get('message', 'No details available')}</p>
                    <p><strong>Occurred At:</strong> {warning_info.get('occurred_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
                </div>
                
                <h2>📊 Current Status</h2>
                <ul>
                    <li>Symbols Processed: {warning_info.get('processed_symbols', 0):,}</li>
                    <li>Records Inserted: {warning_info.get('records_inserted', 0):,}</li>
                </ul>
                
                <p><strong>Action:</strong> {warning_info.get('action', 'Continuing with processing')}</p>
            </div>
            
            <div class="footer">
                <p>Historical Data Fetcher - OpenAlgo Platform</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_warning_text(self, warning_info: Dict[str, Any]) -> str:
        """Format warning notification as plain text"""
        
        text = f"""
Historical Data Fetch Warning
============================

Warning Details:
- Type: {warning_info.get('warning_type', 'Unknown')}
- Message: {warning_info.get('message', 'No details available')}
- Occurred At: {warning_info.get('occurred_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}

Current Status:
- Symbols Processed: {warning_info.get('processed_symbols', 0):,}
- Records Inserted: {warning_info.get('records_inserted', 0):,}

Action: {warning_info.get('action', 'Continuing with processing')}
        """.strip()
        
        return text
    
    def _format_summary_html(self, stats: Dict[str, Any]) -> str:
        """Format detailed summary report as HTML"""
        
        return self._format_success_html(stats)  # Reuse success format for now
    
    def _format_summary_text(self, stats: Dict[str, Any]) -> str:
        """Format detailed summary report as plain text"""
        
        return self._format_success_text(stats)  # Reuse success format for now
    
    def _format_instrument_table_html(self, instrument_stats: Dict[str, Any]) -> str:
        """Format instrument statistics as HTML table"""
        
        if not instrument_stats:
            return "<p>No instrument data available</p>"
        
        rows = []
        for instrument_type, count in instrument_stats.items():
            rows.append(f"<tr><td>{instrument_type}</td><td>{count:,}</td></tr>")
        
        return f"""
        <table class="stats-table">
            <tr><th>Instrument Type</th><th>Count</th></tr>
            {''.join(rows)}
        </table>
        """
    
    def _format_timeframe_table_html(self, timeframe_stats: Dict[str, Any]) -> str:
        """Format timeframe statistics as HTML table"""
        
        if not timeframe_stats:
            return "<p>No timeframe data available</p>"
        
        rows = []
        for timeframe, count in timeframe_stats.items():
            rows.append(f"<tr><td>{timeframe}</td><td>{count:,}</td></tr>")
        
        return f"""
        <table class="stats-table">
            <tr><th>Timeframe</th><th>Records</th></tr>
            {''.join(rows)}
        </table>
        """
    
    def _format_exchange_table_html(self, exchange_stats: Dict[str, Any]) -> str:
        """Format exchange statistics as HTML table"""
        
        if not exchange_stats:
            return "<p>No exchange data available</p>"
        
        rows = []
        for exchange, count in exchange_stats.items():
            rows.append(f"<tr><td>{exchange}</td><td>{count:,}</td></tr>")
        
        return f"""
        <table class="stats-table">
            <tr><th>Exchange</th><th>Symbols</th></tr>
            {''.join(rows)}
        </table>
        """
    
    def _format_dict_as_text(self, data: Dict[str, Any]) -> str:
        """Format dictionary as plain text list"""
        
        if not data:
            return "- No data available"
        
        lines = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                lines.append(f"- {key}: {value:,}")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    async def _send_email(
        self,
        subject: str,
        html_content: str,
        text_content: str,
        attachments: Optional[List[Dict]] = None
    ):
        """Send email to all recipients"""
        
        if not self.recipients:
            logger.warning("No email recipients configured")
            return
        
        # Run email sending in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._send_email_sync,
            subject,
            html_content,
            text_content,
            attachments
        )
    
    def _send_email_sync(
        self,
        subject: str,
        html_content: str,
        text_content: str,
        attachments: Optional[List[Dict]] = None
    ):
        """Send email synchronously"""
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.username
            msg['Subject'] = subject
            
            # Add text and HTML parts
            text_part = MIMEText(text_content, 'plain')
            html_part = MIMEText(html_content, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Add attachments if provided
            if attachments:
                for attachment in attachments:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment['content'])
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment["filename"]}'
                    )
                    msg.attach(part)
            
            # Send to all recipients
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
                
                for recipient in self.recipients:
                    msg['To'] = recipient
                    text = msg.as_string()
                    server.sendmail(self.username, recipient, text)
                    
                    logger.info(f"Email sent successfully to {recipient}")
                    self.stats['successful_sent'] += 1
                    self.stats['total_sent'] += 1
                    
                    # Remove To header for next recipient
                    del msg['To']
        
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            self.stats['failed_sent'] += len(self.recipients)
            self.stats['total_sent'] += len(self.recipients)
            self.stats['last_error'] = str(e)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics"""
        return self.stats.copy()
    
    async def test_connection(self) -> bool:
        """Test SMTP connection"""
        
        try:
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
                logger.info("SMTP connection test successful")
                return True
                
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False
