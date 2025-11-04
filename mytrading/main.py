#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MyTrading System - Main Entry Point
===================================

This is the main entry point for the MyTrading system. It initializes and runs
the complete trading orchestrator with all components.

Usage:
    python main.py [--config config.yaml] [--symbols symbols.yaml]
    
Example:
    python main.py --config config/production.yaml --symbols config/nifty_symbols.yaml
"""

import asyncio
import sys
import os
import argparse
import signal
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mytrading.core.orchestrator import TradingOrchestrator
from mytrading.config.settings import TradingSettings
from mytrading.utils.logging_config import setup_logging
from loguru import logger


class TradingSystemMain:
    """Main application class for the trading system"""
    
    def __init__(self):
        self.orchestrator: Optional[TradingOrchestrator] = None
        self.shutdown_event = asyncio.Event()
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown of the trading system"""
        logger.info("🛑 Initiating system shutdown...")
        self.shutdown_event.set()
        
        if self.orchestrator:
            await self.orchestrator.shutdown()
            
        logger.info("✅ System shutdown complete")
    
    async def run(self, env_file_path: str):
        """Run the complete trading system"""
        try:
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Load environment variables if file exists
            if os.path.exists(env_file_path):
                from dotenv import load_dotenv
                load_dotenv(env_file_path)
                logger.info(f"📋 Loaded environment variables from: {env_file_path}")
            else:
                logger.info("📋 Using system environment variables (no .env file found)")
            
            # Load configuration from environment variables
            logger.info("📋 Loading configuration from environment...")
            settings = TradingSettings()
            
            # Initialize orchestrator
            logger.info("🚀 Initializing Trading Orchestrator...")
            self.orchestrator = TradingOrchestrator(settings)
            
            # Start the system
            logger.info("▶️  Starting Trading System...")
            await self.orchestrator.start()
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except KeyboardInterrupt:
            logger.info("⏹️ Interrupted by user")
        except Exception as e:
            logger.error(f"💥 Fatal error: {e}")
            logger.exception("Full traceback:")
            raise
        finally:
            await self.shutdown()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MyTrading - Advanced Real-time Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Use default .env file
  python main.py --env-file production.env  # Use production environment
  python main.py --log-level DEBUG        # Enable debug logging
        """
    )
    
    parser.add_argument(
        "--env-file", "-e",
        default=".env",
        help="Path to environment file (default: .env)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Run in simulation mode (no real trades)"
    )
    
    return parser.parse_args()


async def main():
    """Main async entry point"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Display startup banner
    logger.info("=" * 80)
    logger.info("🚀 MYTRADING SYSTEM - ADVANCED REAL-TIME TRADING")
    logger.info("=" * 80)
    logger.info(f"📋 Environment File: {args.env_file}")
    logger.info(f"📝 Log Level: {args.log_level}")
    logger.info(f"🧪 Dry Run: {'Yes' if args.dry_run else 'No'}")
    logger.info("=" * 80)
    
    # Run the trading system
    app = TradingSystemMain()
    await app.run(args.env_file)


if __name__ == "__main__":
    try:
        # Set event loop policy for Windows compatibility
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Run the main application
        asyncio.run(main())
        
    except KeyboardInterrupt:
        logger.info("⏹️ Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Application failed: {e}")
        sys.exit(1)
