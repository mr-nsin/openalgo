"""
Strategy Builder Main Entry Point

Run backtests on trading strategies using OpenAlgo historical data.
All configuration is read from .env file.
"""

import asyncio
import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import List, Dict, Any

# Add current directory to path for relative imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Use relative imports from current directory
from config.strategy_settings import StrategySettings
from services.data_service import DataService
from strategies.strategy_registry import get_strategy


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
    settings: StrategySettings,
    symbol: str,
    data_service: DataService = None
) -> tuple:
    """
    Fetch historical data from QuestDB or OpenAlgo API
    
    Args:
        settings: Strategy settings with data source configuration
        symbol: Symbol to fetch data for
        data_service: Optional pre-initialized data service
    
    Returns:
        Tuple of (DataService, DataFrame)
    """
    # Initialize data service if not provided
    if data_service is None:
        data_service = DataService(settings)
        await data_service.initialize()
    
    # Fetch historical data
    logger.info(f"Fetching historical data for {symbol} {settings.exchange}...")
    logger.info(f"Period: {settings.start_date} to {settings.end_date}")
    logger.info(f"Interval: {settings.interval}")
    logger.info(f"Data Source: {settings.data_source.value}")
    
    df = await data_service.get_historical_data(
        symbol=symbol.upper(),
        exchange=settings.exchange.upper(),
        interval=settings.interval,
        start_date=settings.start_date,
        end_date=settings.end_date
    )
    
    if df.empty:
        raise ValueError(f"No data found for {symbol} {settings.exchange}")
    
    logger.info(f"Fetched {len(df)} records")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return data_service, df


def run_strategy_backtest(
    settings: StrategySettings,
    df,
    strategy,
    symbol: str
) -> dict:
    """
    Run backtest on a strategy
    
    Args:
        settings: Strategy settings
        df: Historical data DataFrame
        strategy: Strategy instance
        symbol: Symbol being backtested
    
    Returns:
        Dictionary with backtest results including symbol
    """
    logger.info(f"Strategy: {strategy.name}")
    logger.info("Calculating indicators and generating signals...")
    
    # Prepare data (calculate indicators and signals)
    df = strategy.prepare_data(df)
    
    logger.info("Running backtest...")
    
    # Run backtest using strategy's own backtest method
    results = strategy.run_backtest(df, settings.initial_capital)
    
    # Add symbol and metadata to results
    results['symbol'] = symbol
    results['exchange'] = settings.exchange
    results['start_date'] = settings.start_date
    results['end_date'] = settings.end_date
    results['interval'] = settings.interval
    results['strategy_name'] = settings.strategy_name
    results['initial_capital'] = settings.initial_capital
    results['return_pct'] = ((results['final_capital'] - settings.initial_capital) / settings.initial_capital) * 100
    
    return results


def save_results_to_csv(results_list: List[Dict[str, Any]], output_path: str = None):
    """
    Save backtest results to CSV file
    
    Args:
        results_list: List of result dictionaries
        output_path: Optional path for CSV file. If None, uses timestamp-based filename
    """
    if not results_list:
        logger.warning("No results to save to CSV")
        return
    
    # Generate output filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create results directory in strategybuilder folder
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"backtest_results_{timestamp}.csv"
    
    # Convert to DataFrame for easier CSV writing
    df_results = pd.DataFrame(results_list)
    
    # Reorder columns for better readability
    column_order = [
        'symbol', 'exchange', 'strategy_name', 'start_date', 'end_date', 'interval',
        'initial_capital', 'final_capital', 'return_pct',
        'total_trades', 'winning_trades', 'losing_trades', 'win_rate', 'total_pnl'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in column_order if col in df_results.columns]
    remaining_columns = [col for col in df_results.columns if col not in available_columns]
    final_columns = available_columns + remaining_columns
    
    df_results = df_results[final_columns]
    
    # Save to CSV
    df_results.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Total symbols processed: {len(results_list)}")
    
    return str(output_path)


async def process_single_symbol(
    settings: StrategySettings,
    symbol: str,
    data_service: DataService,
    strategy,
    is_first_symbol: bool = False
) -> Dict[str, Any]:
    """
    Process a single symbol: fetch data, run backtest, return results
    
    Args:
        settings: Strategy settings
        symbol: Symbol to process
        data_service: Initialized data service
        strategy: Strategy instance
        is_first_symbol: Whether this is the first symbol being processed
    
    Returns:
        Dictionary with backtest results
    """
    try:
        logger.info("=" * 60)
        logger.info(f"Processing: {symbol} {settings.exchange}")
        logger.info("=" * 60)
        
        # Step 1: Fetch historical data
        data_service, df = await fetch_historical_data(settings, symbol, data_service)
        
        # Display sample of fetched data for first symbol only
        if is_first_symbol:
            logger.info("=" * 60)
            logger.info("✓ Data fetched successfully from OpenAlgo!")
            logger.info("=" * 60)
            logger.info(f"\nDataFrame Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            logger.info(f"\nColumns: {', '.join(df.columns.tolist())}")
            logger.info("\nFirst 5 rows of fetched data:")
            logger.info("\n" + str(df.head()))
            logger.info("=" * 60)
        
        # Step 2: Run backtest
        results = run_strategy_backtest(settings, df, strategy, symbol)
        
        # Step 3: Display results
        logger.info(f"\n=== BACKTEST RESULTS for {symbol} ===")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Winning Trades: {results['winning_trades']}")
        logger.info(f"Losing Trades: {results['losing_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2f}%")
        logger.info(f"Total P&L: {results['total_pnl']:.2f}")
        logger.info(f"Final Capital: {results['final_capital']:.2f}")
        logger.info(f"Return: {results['return_pct']:.2f}%")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        # Return error result
        return {
            'symbol': symbol,
            'exchange': settings.exchange,
            'strategy_name': settings.strategy_name,
            'start_date': settings.start_date,
            'end_date': settings.end_date,
            'interval': settings.interval,
            'initial_capital': settings.initial_capital,
            'error': str(e),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'final_capital': settings.initial_capital,
            'return_pct': 0.0
        }


async def main():
    """Main entry point - reads all configuration from .env"""
    # Note: .env file is loaded at module level in strategy_settings.py
    # This ensures it's loaded before Pydantic reads the environment variables
    
    # Load settings from .env (will ignore extra fields not in StrategySettings)
    settings = StrategySettings()
    setup_logging(settings)
    
    # Log API key status (masked)
    if settings.openalgo_api_key:
        api_key_preview = settings.openalgo_api_key[:8] + "..." if len(settings.openalgo_api_key) > 8 else "***"
        logger.info(f"OpenAlgo API Key loaded: {api_key_preview}")
    else:
        logger.warning("OPENALGO_API_KEY is not set in .env file!")
        logger.warning("Please check your .env file in strategies/strategybuilder/ directory")
    
    # Validate required settings
    if not settings.start_date or not settings.end_date:
        logger.error("STRATEGY_START_DATE and STRATEGY_END_DATE must be set in .env file")
        return
    
    if not settings.openalgo_api_key:
        logger.warning("OPENALGO_API_KEY not set. Some features may not work.")
    
    # Parse symbols - support both STRATEGY_SYMBOLS (comma-separated) and STRATEGY_SYMBOL (single or comma-separated)
    symbols = []
    
    # First, check if STRATEGY_SYMBOLS is explicitly set and not empty
    if settings.symbols and settings.symbols.strip():
        # Multiple symbols from STRATEGY_SYMBOLS
        symbols = [s.strip().upper() for s in settings.symbols.split(',') if s.strip()]
        logger.info(f"Using STRATEGY_SYMBOLS: {settings.symbols}")
    # If STRATEGY_SYMBOLS is not set, check if STRATEGY_SYMBOL contains commas
    elif settings.symbol and ',' in settings.symbol:
        # User put comma-separated values in STRATEGY_SYMBOL instead of STRATEGY_SYMBOLS
        symbols = [s.strip().upper() for s in settings.symbol.split(',') if s.strip()]
        logger.info(f"Detected comma-separated symbols in STRATEGY_SYMBOL: {settings.symbol}")
    else:
        # Single symbol from STRATEGY_SYMBOL
        symbols = [settings.symbol.upper()] if settings.symbol else []
        logger.info(f"Using single STRATEGY_SYMBOL: {settings.symbol}")
    
    # Validate that we have at least one symbol
    if not symbols:
        logger.error("No symbols specified! Please set either STRATEGY_SYMBOL or STRATEGY_SYMBOLS in .env file")
        return
    
    logger.info("=" * 60)
    logger.info("Strategy Builder - Multi-Symbol Backtest Runner")
    logger.info("=" * 60)
    logger.info(f"Parsed {len(symbols)} symbol(s): {', '.join(symbols)}")
    logger.info(f"Exchange: {settings.exchange}")
    logger.info(f"Interval: {settings.interval} | Period: {settings.start_date} to {settings.end_date}")
    logger.info(f"Strategy: {settings.strategy_name}")
    logger.info(f"Initial Capital: {settings.initial_capital}")
    logger.info(f"Total symbols to process: {len(symbols)}")
    logger.info("=" * 60)
    
    # Initialize strategy once (shared across all symbols)
    strategy_params = {
        'short_sma_period': settings.short_sma_period,
        'mid_ema_period': settings.mid_ema_period,
        'long_sma_period': settings.long_sma_period,
        'swing_lookback': settings.swing_lookback,
        'volume_filter': settings.volume_filter,
        'macd_filter': settings.macd_filter,
    }
    
    strategy = get_strategy(settings.strategy_name, params=strategy_params)
    
    # Initialize data service once (reused for all symbols)
    data_service = None
    all_results = []
    
    try:
        # Initialize data service
        data_service = DataService(settings)
        await data_service.initialize()
        
        # Process each symbol
        for idx, symbol in enumerate(symbols, 1):
            logger.info(f"\n[{idx}/{len(symbols)}] Processing {symbol}...")
            try:
                is_first = (idx == 1)
                result = await process_single_symbol(settings, symbol, data_service, strategy, is_first_symbol=is_first)
                all_results.append(result)
            except Exception as e:
                logger.exception(f"Failed to process {symbol}: {e}")
                # Add error result
                all_results.append({
                    'symbol': symbol,
                    'exchange': settings.exchange,
                    'strategy_name': settings.strategy_name,
                    'start_date': settings.start_date,
                    'end_date': settings.end_date,
                    'interval': settings.interval,
                    'initial_capital': settings.initial_capital,
                    'error': str(e),
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'final_capital': settings.initial_capital,
                    'return_pct': 0.0
                })
        
        # Save all results to CSV
        if all_results:
            csv_path = save_results_to_csv(all_results)
            logger.info("=" * 60)
            logger.info("SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total symbols processed: {len(all_results)}")
            successful = [r for r in all_results if 'error' not in r or not r.get('error')]
            failed = [r for r in all_results if 'error' in r and r.get('error')]
            logger.info(f"Successful: {len(successful)}")
            logger.info(f"Failed: {len(failed)}")
            if successful:
                avg_return = sum(r.get('return_pct', 0) for r in successful) / len(successful)
                total_pnl = sum(r.get('total_pnl', 0) for r in successful)
                logger.info(f"Average Return: {avg_return:.2f}%")
                logger.info(f"Total P&L (all symbols): {total_pnl:.2f}")
            logger.info(f"Results saved to: {csv_path}")
            logger.info("=" * 60)
        else:
            logger.warning("No results to save")
        
    except Exception as e:
        logger.exception(f"Error during backtest: {e}")
        raise
    finally:
        if data_service:
            await data_service.close()


if __name__ == "__main__":
    asyncio.run(main())
