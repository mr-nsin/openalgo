"""
Strategy Builder Main Entry Point

Run backtests on trading strategies using OpenAlgo historical data.
All configuration is read from .env file.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from strategies.strategybuilder.config.strategy_settings import StrategySettings
from strategies.strategybuilder.services.data_service import DataService
from strategies.strategybuilder.strategies.strategy_registry import get_strategy


def setup_logging(settings: StrategySettings):
    """Setup logging configuration"""
    log_file = Path(settings.log_file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=settings.log_level
    )
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level=settings.log_level,
        rotation="100 MB",
        retention="30 days"
    )


async def fetch_historical_data(
    settings: StrategySettings
) -> tuple:
    """
    Fetch historical data from QuestDB or OpenAlgo API
    
    Args:
        settings: Strategy settings with data source configuration
    
    Returns:
        Tuple of (DataService, DataFrame)
    """
    # Initialize data service
    data_service = DataService(settings)
    await data_service.initialize()
    
    # Fetch historical data
    logger.info(f"Fetching historical data for {settings.symbol} {settings.exchange}...")
    logger.info(f"Period: {settings.start_date} to {settings.end_date}")
    logger.info(f"Interval: {settings.interval}")
    logger.info(f"Data Source: {settings.data_source.value}")
    
    df = await data_service.get_historical_data(
        symbol=settings.symbol.upper(),
        exchange=settings.exchange.upper(),
        interval=settings.interval,
        start_date=settings.start_date,
        end_date=settings.end_date
    )
    
    if df.empty:
        raise ValueError(f"No data found for {settings.symbol} {settings.exchange}")
    
    logger.info(f"Fetched {len(df)} records")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return data_service, df


def run_strategy_backtest(
    settings: StrategySettings,
    df,
    strategy
) -> dict:
    """
    Run backtest on a strategy
    
    Args:
        settings: Strategy settings
        df: Historical data DataFrame
        strategy: Strategy instance
    
    Returns:
        Dictionary with backtest results
    """
    logger.info(f"Strategy: {strategy.name}")
    logger.info("Calculating indicators and generating signals...")
    
    # Prepare data (calculate indicators and signals)
    df = strategy.prepare_data(df)
    
    logger.info("Running backtest...")
    
    # Run backtest using strategy's own backtest method
    results = strategy.run_backtest(df, settings.initial_capital)
    
    return results


async def main():
    """Main entry point - reads all configuration from .env"""
    # Load settings from .env
    settings = StrategySettings()
    setup_logging(settings)
    
    # Validate required settings
    if not settings.start_date or not settings.end_date:
        logger.error("STRATEGY_START_DATE and STRATEGY_END_DATE must be set in .env file")
        return
    
    if not settings.openalgo_api_key:
        logger.warning("OPENALGO_API_KEY not set. Some features may not work.")
    
    logger.info("=" * 60)
    logger.info("Strategy Builder - Backtest Runner")
    logger.info("=" * 60)
    logger.info(f"Symbol: {settings.symbol} | Exchange: {settings.exchange}")
    logger.info(f"Interval: {settings.interval} | Period: {settings.start_date} to {settings.end_date}")
    logger.info(f"Strategy: {settings.strategy_name}")
    logger.info(f"Initial Capital: {settings.initial_capital}")
    logger.info("=" * 60)
    
    data_service = None
    try:
        # Step 1: Fetch historical data
        data_service, df = await fetch_historical_data(settings)
        
        # Step 2: Initialize strategy
        strategy_params = {
            'short_sma_period': settings.short_sma_period,
            'mid_ema_period': settings.mid_ema_period,
            'long_sma_period': settings.long_sma_period,
            'swing_lookback': settings.swing_lookback,
            'volume_filter': settings.volume_filter,
            'macd_filter': settings.macd_filter,
        }
        
        strategy = get_strategy(settings.strategy_name, params=strategy_params)
        
        # Step 3: Run backtest
        results = run_strategy_backtest(settings, df, strategy)
        
        # Step 4: Display results (same format as original script)
        print("\n=== BACKTEST RESULTS ===")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Total P&L: {results['total_pnl']:.2f}")
        print(f"Final Capital: {results['final_capital']:.2f}")
        
    except Exception as e:
        logger.exception(f"Error during backtest: {e}")
        raise
    finally:
        if data_service:
            await data_service.close()


if __name__ == "__main__":
    asyncio.run(main())
