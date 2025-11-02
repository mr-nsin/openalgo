"""
Cron Scheduler for Historical Data Fetcher

Market-aware scheduler that runs historical data fetching at optimal times
based on market calendar and trading schedules.
"""

import asyncio
import sys
import os
from datetime import datetime, time, date
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
import pytz
from typing import Dict, Any, Optional, List

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from historicalfetcher.utils.async_logger import get_async_logger

_async_logger = get_async_logger()
logger = _async_logger.get_logger()
from historicalfetcher.config.openalgo_settings import OpenAlgoSettings
from historicalfetcher.scheduler.market_calendar import MarketCalendar

# Using loguru logger directly

class HistoricalDataScheduler:
    """
    Market-aware scheduler for historical data fetching
    """
    
    def __init__(self, settings: OpenAlgoSettings):
        """
        Initialize scheduler with market calendar integration
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.market_calendar = MarketCalendar()
        
        # Configure job store (use OpenAlgo database)
        jobstores = {
            'default': SQLAlchemyJobStore(url=settings.openalgo_database_url, tablename='scheduler_jobs')
        }
        
        # Configure executors
        executors = {
            'default': AsyncIOExecutor()
        }
        
        # Job defaults
        job_defaults = {
            'coalesce': False,  # Don't combine missed jobs
            'max_instances': 1,  # Only one instance of each job
            'misfire_grace_time': 300  # 5 minutes grace time for missed jobs
        }
        
        # Configure scheduler with IST timezone
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=pytz.timezone('Asia/Kolkata')
        )
        
        # Add event listeners
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
        
        # Job statistics
        self.job_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'last_run': None,
            'last_success': None,
            'last_error': None
        }
    
    def schedule_daily_fetch(self):
        """Schedule daily historical data fetch after market hours"""
        
        # Schedule for 6:30 PM IST (3 hours after market closes at 3:30 PM)
        self.scheduler.add_job(
            func=self._run_daily_fetch,
            trigger=CronTrigger(
                hour=18,
                minute=30,
                timezone=pytz.timezone('Asia/Kolkata')
            ),
            id='daily_historical_fetch',
            name='Daily Historical Data Fetch',
            replace_existing=True,
            kwargs={'fetch_type': 'daily'}
        )
        
        logger.info("📅 Scheduled daily historical data fetch at 6:30 PM IST")
    
    def schedule_weekend_full_sync(self):
        """Schedule comprehensive sync on weekends"""
        
        # Schedule for Saturday 10:00 PM IST (when markets are closed)
        self.scheduler.add_job(
            func=self._run_weekend_sync,
            trigger=CronTrigger(
                day_of_week='sat',
                hour=22,
                minute=0,
                timezone=pytz.timezone('Asia/Kolkata')
            ),
            id='weekend_full_sync',
            name='Weekend Full Historical Sync',
            replace_existing=True,
            kwargs={'fetch_type': 'weekend_full'}
        )
        
        logger.info("📅 Scheduled weekend full sync at Saturday 10:00 PM IST")
    
    def schedule_intraday_updates(self):
        """Schedule intraday updates during market hours (optional)"""
        
        # Schedule every 30 minutes during market hours (9:15 AM to 3:30 PM)
        # This is optional and can be enabled for real-time data needs
        self.scheduler.add_job(
            func=self._run_intraday_update,
            trigger=CronTrigger(
                minute='15,45',  # At 15 and 45 minutes past each hour
                hour='9-15',     # During market hours
                timezone=pytz.timezone('Asia/Kolkata')
            ),
            id='intraday_updates',
            name='Intraday Historical Updates',
            replace_existing=True,
            kwargs={'fetch_type': 'intraday'}
        )
        
        logger.info("📅 Scheduled intraday updates every 30 minutes during market hours")
    
    def schedule_monthly_cleanup(self):
        """Schedule monthly data cleanup and maintenance"""
        
        # Schedule for first Sunday of each month at 2:00 AM
        self.scheduler.add_job(
            func=self._run_monthly_cleanup,
            trigger=CronTrigger(
                day='1-7',       # First week of month
                day_of_week='sun',  # Sunday
                hour=2,
                minute=0,
                timezone=pytz.timezone('Asia/Kolkata')
            ),
            id='monthly_cleanup',
            name='Monthly Data Cleanup',
            replace_existing=True,
            kwargs={'fetch_type': 'cleanup'}
        )
        
        logger.info("📅 Scheduled monthly cleanup for first Sunday at 2:00 AM IST")
    
    def schedule_one_time_fetch(
        self,
        run_time: datetime,
        fetch_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schedule a one-time historical data fetch
        
        Args:
            run_time: When to run the fetch
            fetch_config: Custom configuration for the fetch
            
        Returns:
            Job ID
        """
        
        job_id = f"one_time_fetch_{run_time.strftime('%Y%m%d_%H%M%S')}"
        
        self.scheduler.add_job(
            func=self._run_custom_fetch,
            trigger=DateTrigger(run_date=run_time),
            id=job_id,
            name=f'One-time Fetch - {run_time.strftime("%Y-%m-%d %H:%M")}',
            replace_existing=True,
            kwargs={
                'fetch_type': 'one_time',
                'config': fetch_config or {}
            }
        )
        
        logger.info(f"📅 Scheduled one-time fetch for {run_time}")
        return job_id
    
    async def _run_daily_fetch(self, fetch_type: str = 'daily'):
        """Execute daily historical data fetch with market calendar check"""
        
        # Check if today is a trading day
        if not self.market_calendar.is_trading_day(date.today()):
            logger.info("📅 Skipping daily fetch - not a trading day")
            return
        
        # Check if it's appropriate time to fetch
        if not self.market_calendar.should_run_historical_fetch():
            logger.info("📅 Skipping daily fetch - not optimal time")
            return
        
        logger.info("🚀 Starting scheduled daily historical data fetch")
        
        try:
            # Import here to avoid circular imports
            from openalgo_main import OpenAlgoHistoricalDataFetcher
            
            # Create fetcher with default settings
            fetcher = OpenAlgoHistoricalDataFetcher()
            
            # Override settings for daily fetch
            fetcher.settings.historical_days_limit = 5  # Last 5 days for daily fetch
            
            await fetcher.run()
            
            self.job_stats['successful_runs'] += 1
            self.job_stats['last_success'] = datetime.now()
            
        except Exception as e:
            logger.error(f"💥 Error in scheduled daily fetch: {e}")
            self.job_stats['failed_runs'] += 1
            self.job_stats['last_error'] = str(e)
            raise
        
        finally:
            self.job_stats['total_runs'] += 1
            self.job_stats['last_run'] = datetime.now()
    
    async def _run_weekend_sync(self, fetch_type: str = 'weekend_full'):
        """Execute weekend full sync"""
        
        logger.info("🚀 Starting scheduled weekend full sync")
        
        try:
            from openalgo_main import OpenAlgoHistoricalDataFetcher
            
            # Create fetcher with extended settings for full sync
            fetcher = OpenAlgoHistoricalDataFetcher()
            
            # Override settings for full sync
            fetcher.settings.historical_days_limit = 365  # Full year for weekend sync
            fetcher.settings.batch_size = 100  # Larger batches for weekend processing
            
            await fetcher.run()
            
            self.job_stats['successful_runs'] += 1
            self.job_stats['last_success'] = datetime.now()
            
        except Exception as e:
            logger.error(f"💥 Error in scheduled weekend sync: {e}")
            self.job_stats['failed_runs'] += 1
            self.job_stats['last_error'] = str(e)
            raise
        
        finally:
            self.job_stats['total_runs'] += 1
            self.job_stats['last_run'] = datetime.now()
    
    async def _run_intraday_update(self, fetch_type: str = 'intraday'):
        """Execute intraday updates (only during market hours)"""
        
        # Double-check if market is open
        if not self.market_calendar.is_market_open():
            logger.debug("📅 Skipping intraday update - market is closed")
            return
        
        logger.info("🔄 Starting intraday historical data update")
        
        try:
            from openalgo_main import OpenAlgoHistoricalDataFetcher
            
            # Create fetcher with minimal settings for intraday
            fetcher = OpenAlgoHistoricalDataFetcher()
            
            # Override settings for intraday updates
            fetcher.settings.historical_days_limit = 1  # Only today
            fetcher.settings.enabled_timeframes = ['1m', '5m']  # Only minute data
            fetcher.settings.batch_size = 20  # Smaller batches for quick updates
            
            await fetcher.run()
            
            self.job_stats['successful_runs'] += 1
            self.job_stats['last_success'] = datetime.now()
            
        except Exception as e:
            logger.error(f"💥 Error in intraday update: {e}")
            self.job_stats['failed_runs'] += 1
            self.job_stats['last_error'] = str(e)
            # Don't raise for intraday updates to avoid stopping the scheduler
    
    async def _run_monthly_cleanup(self, fetch_type: str = 'cleanup'):
        """Execute monthly data cleanup and maintenance"""
        
        logger.info("🧹 Starting monthly data cleanup")
        
        try:
            # Implement cleanup logic here
            # This could include:
            # - Removing old temporary files
            # - Compacting database tables
            # - Generating monthly reports
            # - Updating market calendar
            
            # For now, just log the cleanup
            logger.info("✅ Monthly cleanup completed")
            
            self.job_stats['successful_runs'] += 1
            self.job_stats['last_success'] = datetime.now()
            
        except Exception as e:
            logger.error(f"💥 Error in monthly cleanup: {e}")
            self.job_stats['failed_runs'] += 1
            self.job_stats['last_error'] = str(e)
        
        finally:
            self.job_stats['total_runs'] += 1
            self.job_stats['last_run'] = datetime.now()
    
    async def _run_custom_fetch(self, fetch_type: str = 'custom', config: Dict[str, Any] = None):
        """Execute custom fetch with provided configuration"""
        
        logger.info(f"🎯 Starting custom historical data fetch: {config}")
        
        try:
            from openalgo_main import OpenAlgoHistoricalDataFetcher
            
            fetcher = OpenAlgoHistoricalDataFetcher()
            
            # Apply custom configuration if provided
            if config:
                for key, value in config.items():
                    if hasattr(fetcher.settings, key):
                        setattr(fetcher.settings, key, value)
                        logger.info(f"Applied custom setting: {key} = {value}")
            
            await fetcher.run()
            
            self.job_stats['successful_runs'] += 1
            self.job_stats['last_success'] = datetime.now()
            
        except Exception as e:
            logger.error(f"💥 Error in custom fetch: {e}")
            self.job_stats['failed_runs'] += 1
            self.job_stats['last_error'] = str(e)
            raise
        
        finally:
            self.job_stats['total_runs'] += 1
            self.job_stats['last_run'] = datetime.now()
    
    def _job_executed(self, event):
        """Handle job execution events"""
        logger.info(f"✅ Job '{event.job_id}' executed successfully")
    
    def _job_error(self, event):
        """Handle job error events"""
        logger.error(f"❌ Job '{event.job_id}' failed: {event.exception}")
    
    def start(self):
        """Start the scheduler"""
        try:
            self.scheduler.start()
            logger.info("⏰ Historical data scheduler started")
            
            # Log scheduled jobs
            jobs = self.scheduler.get_jobs()
            logger.info(f"📋 {len(jobs)} jobs scheduled:")
            for job in jobs:
                logger.info(f"  • {job.name} (ID: {job.id}) - Next run: {job.next_run_time}")
                
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise
    
    def shutdown(self, wait: bool = True):
        """Shutdown the scheduler"""
        try:
            self.scheduler.shutdown(wait=wait)
            logger.info("⏹️ Historical data scheduler stopped")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")
    
    def pause_job(self, job_id: str):
        """Pause a specific job"""
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"⏸️ Job '{job_id}' paused")
        except Exception as e:
            logger.error(f"Error pausing job '{job_id}': {e}")
    
    def resume_job(self, job_id: str):
        """Resume a paused job"""
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"▶️ Job '{job_id}' resumed")
        except Exception as e:
            logger.error(f"Error resuming job '{job_id}': {e}")
    
    def remove_job(self, job_id: str):
        """Remove a job from scheduler"""
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"🗑️ Job '{job_id}' removed")
        except Exception as e:
            logger.error(f"Error removing job '{job_id}': {e}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                return {
                    'id': job.id,
                    'name': job.name,
                    'next_run_time': job.next_run_time,
                    'trigger': str(job.trigger),
                    'pending': job.pending
                }
            return None
        except Exception as e:
            logger.error(f"Error getting job status for '{job_id}': {e}")
            return None
    
    def get_all_jobs_status(self) -> List[Dict[str, Any]]:
        """Get status of all scheduled jobs"""
        jobs_status = []
        
        try:
            jobs = self.scheduler.get_jobs()
            for job in jobs:
                jobs_status.append({
                    'id': job.id,
                    'name': job.name,
                    'next_run_time': job.next_run_time,
                    'trigger': str(job.trigger),
                    'pending': job.pending
                })
        except Exception as e:
            logger.error(f"Error getting jobs status: {e}")
        
        return jobs_status
    
    def get_next_run_times(self) -> Dict[str, datetime]:
        """Get next run times for all scheduled jobs"""
        
        next_runs = {}
        try:
            jobs = self.scheduler.get_jobs()
            for job in jobs:
                if job.next_run_time:
                    next_runs[job.name] = job.next_run_time
        except Exception as e:
            logger.error(f"Error getting next run times: {e}")
        
        return next_runs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        
        stats = self.job_stats.copy()
        
        # Add success rate
        if stats['total_runs'] > 0:
            stats['success_rate'] = (stats['successful_runs'] / stats['total_runs']) * 100
        else:
            stats['success_rate'] = 0
        
        # Add scheduler info
        stats['scheduler_running'] = self.scheduler.running
        stats['total_jobs'] = len(self.scheduler.get_jobs())
        
        return stats
    
    def is_running(self) -> bool:
        """Check if scheduler is running"""
        return self.scheduler.running

# Standalone scheduler runner
async def run_scheduler():
    """Run the scheduler as a standalone service"""
    
    try:
        # Load settings
        settings = Settings()
        
        # Create and configure scheduler
        scheduler = HistoricalDataScheduler(settings)
        
        # Schedule all jobs
        scheduler.schedule_daily_fetch()
        scheduler.schedule_weekend_full_sync()
        # scheduler.schedule_intraday_updates()  # Uncomment if needed
        scheduler.schedule_monthly_cleanup()
        
        # Start scheduler
        scheduler.start()
        
        logger.info("🚀 Scheduler service started. Press Ctrl+C to stop.")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                # Log status periodically
                if datetime.now().minute == 0:  # Every hour
                    stats = scheduler.get_statistics()
                    logger.info(f"📊 Scheduler stats: {stats}")
                    
        except KeyboardInterrupt:
            logger.info("⏹️ Shutdown requested")
            
    except Exception as e:
        logger.error(f"💥 Scheduler service error: {e}")
        raise
        
    finally:
        scheduler.shutdown()

if __name__ == "__main__":
    # Set up event loop policy for Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(run_scheduler())
