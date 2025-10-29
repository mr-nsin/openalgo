"""
Comprehensive Indicator Calculation Engine

Orchestrates the calculation of all technical indicators, options Greeks,
and market analytics with optimized batch processing and caching.
"""

import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import time

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from utils.logging import get_logger
from config.settings import InstrumentType, TimeFrame
from fetchers.symbol_manager import SymbolInfo
from fetchers.zerodha_fetcher import HistoricalCandle
from indicators.numba_indicators import NumbaIndicators, TimeFrameOptimizedIndicators, NumbaOptimizer
from indicators.options_greeks import OptionsGreeksCalculator, ImpliedVolatilityCalculator, AdvancedGreeks
from database.enhanced_schemas import TimeFrameCode, OptionTypeCode

logger = get_logger(__name__)

@dataclass
class IndicatorResult:
    """Container for calculated indicators"""
    symbol: str
    timeframe: int
    timestamp: datetime
    
    # Basic OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Technical Indicators
    indicators: Dict[str, float]
    
    # Options Greeks (if applicable)
    greeks: Optional[Dict[str, float]] = None
    
    # Market Depth (if available)
    market_depth: Optional[Dict[str, float]] = None
    
    # Derived Metrics
    derived_metrics: Optional[Dict[str, float]] = None

@dataclass
class CalculationConfig:
    """Configuration for indicator calculations"""
    
    # Technical Indicators
    calculate_trend_indicators: bool = True
    calculate_momentum_indicators: bool = True
    calculate_volatility_indicators: bool = True
    calculate_volume_indicators: bool = True
    
    # Options-specific
    calculate_greeks: bool = True
    calculate_iv: bool = True
    calculate_advanced_greeks: bool = False
    
    # Market microstructure
    calculate_market_depth: bool = True
    
    # Performance settings
    use_parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes

class IndicatorEngine:
    """Main engine for calculating all indicators and analytics"""
    
    def __init__(self, config: CalculationConfig = None):
        self.config = config or CalculationConfig()
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Caching
        self.cache: Dict[str, Tuple[datetime, Dict]] = {}
        self.cache_enabled = self.config.enable_caching
        
        # Statistics
        self.stats = {
            'calculations_performed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_calculation_time': 0.0,
            'average_calculation_time': 0.0
        }
        
        # Warm up Numba functions
        NumbaOptimizer.warm_up_functions()
        
        logger.info("Indicator Engine initialized with optimized Numba functions")
    
    async def calculate_equity_indicators(
        self,
        symbol_info: SymbolInfo,
        candles: List[HistoricalCandle],
        timeframe: TimeFrame,
        market_depth_data: Optional[Dict] = None
    ) -> List[IndicatorResult]:
        """Calculate all indicators for equity data"""
        
        if not candles:
            return []
        
        start_time = time.time()
        
        try:
            # Convert candles to numpy arrays
            ohlcv_data = self._candles_to_arrays(candles)
            
            # Calculate technical indicators
            indicators = await self._calculate_technical_indicators(
                ohlcv_data, timeframe, symbol_info.instrument_type
            )
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(ohlcv_data, indicators)
            
            # Create results
            results = []
            tf_code = TimeFrameCode.from_timeframe(timeframe)
            
            for i, candle in enumerate(candles):
                # Extract indicators for this timestamp
                candle_indicators = {
                    key: values[i] if not np.isnan(values[i]) else None
                    for key, values in indicators.items()
                }
                
                # Extract derived metrics
                candle_derived = {
                    key: values[i] if not np.isnan(values[i]) else None
                    for key, values in derived_metrics.items()
                }
                
                result = IndicatorResult(
                    symbol=symbol_info.symbol,
                    timeframe=int(tf_code),
                    timestamp=candle.timestamp,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    indicators=candle_indicators,
                    derived_metrics=candle_derived,
                    market_depth=market_depth_data
                )
                
                results.append(result)
            
            # Update statistics
            calculation_time = time.time() - start_time
            self._update_stats(calculation_time)
            
            logger.debug(f"Calculated indicators for {symbol_info.symbol} in {calculation_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating equity indicators for {symbol_info.symbol}: {e}")
            return []
    
    async def calculate_options_indicators(
        self,
        symbol_info: SymbolInfo,
        candles: List[HistoricalCandle],
        timeframe: TimeFrame,
        spot_price: float,
        risk_free_rate: float = 0.06,
        dividend_yield: float = 0.0,
        market_depth_data: Optional[Dict] = None
    ) -> List[IndicatorResult]:
        """Calculate indicators and Greeks for options data"""
        
        if not candles:
            return []
        
        start_time = time.time()
        
        try:
            # Convert candles to numpy arrays
            ohlcv_data = self._candles_to_arrays(candles)
            
            # Calculate technical indicators for option premium
            indicators = await self._calculate_technical_indicators(
                ohlcv_data, timeframe, symbol_info.instrument_type
            )
            
            # Calculate options Greeks
            greeks_data = await self._calculate_options_greeks(
                symbol_info, candles, spot_price, risk_free_rate, dividend_yield
            )
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(ohlcv_data, indicators)
            
            # Create results
            results = []
            tf_code = TimeFrameCode.from_timeframe(timeframe)
            
            for i, candle in enumerate(candles):
                # Extract indicators for this timestamp
                candle_indicators = {
                    key: values[i] if not np.isnan(values[i]) else None
                    for key, values in indicators.items()
                }
                
                # Extract Greeks for this timestamp
                candle_greeks = {
                    key: values[i] if not np.isnan(values[i]) else None
                    for key, values in greeks_data.items()
                } if greeks_data else None
                
                # Extract derived metrics
                candle_derived = {
                    key: values[i] if not np.isnan(values[i]) else None
                    for key, values in derived_metrics.items()
                }
                
                result = IndicatorResult(
                    symbol=symbol_info.symbol,
                    timeframe=int(tf_code),
                    timestamp=candle.timestamp,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    indicators=candle_indicators,
                    greeks=candle_greeks,
                    derived_metrics=candle_derived,
                    market_depth=market_depth_data
                )
                
                results.append(result)
            
            # Update statistics
            calculation_time = time.time() - start_time
            self._update_stats(calculation_time)
            
            logger.debug(f"Calculated options indicators for {symbol_info.symbol} in {calculation_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating options indicators for {symbol_info.symbol}: {e}")
            return []
    
    async def calculate_futures_indicators(
        self,
        symbol_info: SymbolInfo,
        candles: List[HistoricalCandle],
        timeframe: TimeFrame,
        spot_price: Optional[float] = None,
        market_depth_data: Optional[Dict] = None
    ) -> List[IndicatorResult]:
        """Calculate indicators for futures data with F&O specific metrics"""
        
        if not candles:
            return []
        
        start_time = time.time()
        
        try:
            # Convert candles to numpy arrays
            ohlcv_data = self._candles_to_arrays(candles)
            
            # Calculate technical indicators
            indicators = await self._calculate_technical_indicators(
                ohlcv_data, timeframe, symbol_info.instrument_type
            )
            
            # Calculate futures-specific metrics
            futures_metrics = self._calculate_futures_metrics(ohlcv_data, spot_price)
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(ohlcv_data, indicators)
            
            # Combine all metrics
            all_derived = {**derived_metrics, **futures_metrics}
            
            # Create results
            results = []
            tf_code = TimeFrameCode.from_timeframe(timeframe)
            
            for i, candle in enumerate(candles):
                # Extract indicators for this timestamp
                candle_indicators = {
                    key: values[i] if not np.isnan(values[i]) else None
                    for key, values in indicators.items()
                }
                
                # Extract derived metrics
                candle_derived = {
                    key: values[i] if not np.isnan(values[i]) else None
                    for key, values in all_derived.items()
                }
                
                result = IndicatorResult(
                    symbol=symbol_info.symbol,
                    timeframe=int(tf_code),
                    timestamp=candle.timestamp,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    indicators=candle_indicators,
                    derived_metrics=candle_derived,
                    market_depth=market_depth_data
                )
                
                results.append(result)
            
            # Update statistics
            calculation_time = time.time() - start_time
            self._update_stats(calculation_time)
            
            logger.debug(f"Calculated futures indicators for {symbol_info.symbol} in {calculation_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating futures indicators for {symbol_info.symbol}: {e}")
            return []
    
    def _candles_to_arrays(self, candles: List[HistoricalCandle]) -> Dict[str, np.ndarray]:
        """Convert candle data to numpy arrays for efficient processing"""
        
        n = len(candles)
        
        return {
            'open': np.array([c.open for c in candles], dtype=np.float64),
            'high': np.array([c.high for c in candles], dtype=np.float64),
            'low': np.array([c.low for c in candles], dtype=np.float64),
            'close': np.array([c.close for c in candles], dtype=np.float64),
            'volume': np.array([c.volume for c in candles], dtype=np.int64),
            'oi': np.array([c.oi for c in candles], dtype=np.int64),
            'timestamps': np.array([c.timestamp for c in candles])
        }
    
    async def _calculate_technical_indicators(
        self,
        ohlcv_data: Dict[str, np.ndarray],
        timeframe: TimeFrame,
        instrument_type: InstrumentType
    ) -> Dict[str, np.ndarray]:
        """Calculate all technical indicators based on timeframe and instrument type"""
        
        high = ohlcv_data['high']
        low = ohlcv_data['low']
        close = ohlcv_data['close']
        volume = ohlcv_data['volume']
        
        indicators = {}
        
        if self.config.calculate_trend_indicators:
            # Trend indicators
            indicators['ema_9'] = NumbaIndicators.ema(close, 9)
            indicators['ema_21'] = NumbaIndicators.ema(close, 21)
            indicators['ema_50'] = NumbaIndicators.ema(close, 50)
            indicators['ema_200'] = NumbaIndicators.ema(close, 200)
            indicators['sma_20'] = NumbaIndicators.sma(close, 20)
            indicators['sma_50'] = NumbaIndicators.sma(close, 50)
        
        if self.config.calculate_momentum_indicators:
            # Momentum indicators
            indicators['rsi_14'] = NumbaIndicators.rsi(close, 14)
            macd_line, macd_signal, macd_hist = NumbaIndicators.macd(close, 12, 26, 9)
            indicators['macd_line'] = macd_line
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist
            
            stoch_k, stoch_d = NumbaIndicators.stochastic(high, low, close, 14, 3)
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
            
            indicators['williams_r'] = NumbaIndicators.williams_r(high, low, close, 14)
            indicators['cci_20'] = NumbaIndicators.cci(high, low, close, 20)
        
        if self.config.calculate_volatility_indicators:
            # Volatility indicators
            indicators['atr_14'] = NumbaIndicators.atr(high, low, close, 14)
            bb_upper, bb_middle, bb_lower = NumbaIndicators.bollinger_bands(close, 20, 2.0)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle * 100
            indicators['bb_percent'] = (close - bb_lower) / (bb_upper - bb_lower) * 100
        
        if self.config.calculate_volume_indicators and instrument_type != InstrumentType.INDEX:
            # Volume indicators (not applicable to indices)
            indicators['volume_sma_20'] = NumbaIndicators.sma(volume.astype(np.float64), 20)
            indicators['vwap'] = NumbaIndicators.vwap(high, low, close, volume)
            indicators['obv'] = NumbaIndicators.obv(close, volume)
            indicators['mfi_14'] = NumbaIndicators.mfi(high, low, close, volume, 14)
        
        # Trend following indicators
        supertrend_7_3, st_signal_7_3 = NumbaIndicators.supertrend(high, low, close, 7, 3.0)
        indicators['supertrend_7_3'] = supertrend_7_3
        indicators['supertrend_signal_7_3'] = st_signal_7_3.astype(np.float64)
        
        supertrend_10_3, st_signal_10_3 = NumbaIndicators.supertrend(high, low, close, 10, 3.0)
        indicators['supertrend_10_3'] = supertrend_10_3
        indicators['supertrend_signal_10_3'] = st_signal_10_3.astype(np.float64)
        
        indicators['parabolic_sar'] = NumbaIndicators.parabolic_sar(high, low, close)
        
        # ADX and directional indicators
        adx, di_plus, di_minus = NumbaIndicators.adx(high, low, close, 14)
        indicators['adx_14'] = adx
        indicators['di_plus'] = di_plus
        indicators['di_minus'] = di_minus
        
        return indicators
    
    async def _calculate_options_greeks(
        self,
        symbol_info: SymbolInfo,
        candles: List[HistoricalCandle],
        spot_price: float,
        risk_free_rate: float,
        dividend_yield: float
    ) -> Dict[str, np.ndarray]:
        """Calculate options Greeks for all candles"""
        
        if not self.config.calculate_greeks:
            return {}
        
        n = len(candles)
        greeks = {
            'delta': np.empty(n),
            'gamma': np.empty(n),
            'theta': np.empty(n),
            'vega': np.empty(n),
            'rho': np.empty(n),
            'lambda_greek': np.empty(n),
            'intrinsic_value': np.empty(n),
            'time_value': np.empty(n),
            'moneyness': np.empty(n),
            'probability_itm': np.empty(n)
        }
        
        if self.config.calculate_iv:
            greeks['implied_volatility'] = np.empty(n)
        
        if self.config.calculate_advanced_greeks:
            greeks['charm'] = np.empty(n)
            greeks['vanna'] = np.empty(n)
            greeks['volga'] = np.empty(n)
        
        # Options parameters
        strike = symbol_info.strike or 0.0
        is_call = symbol_info.instrument_type == InstrumentType.CALL_OPTION
        
        # Calculate expiry time for each candle
        expiry_date = self._parse_expiry_date(symbol_info.expiry)
        
        for i, candle in enumerate(candles):
            # Calculate time to expiry in years
            time_to_expiry = self._calculate_time_to_expiry(candle.timestamp, expiry_date)
            
            if time_to_expiry <= 0:
                # Expired option
                greeks['delta'][i] = 0.0
                greeks['gamma'][i] = 0.0
                greeks['theta'][i] = 0.0
                greeks['vega'][i] = 0.0
                greeks['rho'][i] = 0.0
                greeks['lambda_greek'][i] = 0.0
                greeks['intrinsic_value'][i] = OptionsGreeksCalculator.intrinsic_value(spot_price, strike, is_call)
                greeks['time_value'][i] = 0.0
                greeks['moneyness'][i] = OptionsGreeksCalculator.moneyness(spot_price, strike, is_call)
                greeks['probability_itm'][i] = 1.0 if greeks['intrinsic_value'][i] > 0 else 0.0
                
                if self.config.calculate_iv:
                    greeks['implied_volatility'][i] = 0.0
                
                continue
            
            # Calculate implied volatility if enabled
            if self.config.calculate_iv:
                iv = ImpliedVolatilityCalculator.implied_volatility_newton_raphson(
                    candle.close, spot_price, strike, time_to_expiry, risk_free_rate, is_call, dividend_yield
                )
                greeks['implied_volatility'][i] = iv
            else:
                iv = 0.2  # Default 20% volatility
            
            # Calculate all Greeks
            (option_price, delta, gamma, theta, vega, rho, lambda_val,
             intrinsic_val, time_val, moneyness_val, prob_itm) = OptionsGreeksCalculator.calculate_all_greeks(
                spot_price, strike, time_to_expiry, risk_free_rate, iv, is_call, dividend_yield
            )
            
            greeks['delta'][i] = delta
            greeks['gamma'][i] = gamma
            greeks['theta'][i] = theta
            greeks['vega'][i] = vega
            greeks['rho'][i] = rho
            greeks['lambda_greek'][i] = lambda_val
            greeks['intrinsic_value'][i] = intrinsic_val
            greeks['time_value'][i] = time_val
            greeks['moneyness'][i] = moneyness_val
            greeks['probability_itm'][i] = prob_itm
            
            # Advanced Greeks
            if self.config.calculate_advanced_greeks:
                greeks['charm'][i] = AdvancedGreeks.charm(spot_price, strike, time_to_expiry, risk_free_rate, iv, is_call, dividend_yield)
                greeks['vanna'][i] = AdvancedGreeks.vanna(spot_price, strike, time_to_expiry, risk_free_rate, iv, dividend_yield)
                greeks['volga'][i] = AdvancedGreeks.volga(spot_price, strike, time_to_expiry, risk_free_rate, iv, dividend_yield)
        
        return greeks
    
    def _calculate_derived_metrics(
        self,
        ohlcv_data: Dict[str, np.ndarray],
        indicators: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Calculate derived metrics from OHLCV and indicators"""
        
        close = ohlcv_data['close']
        high = ohlcv_data['high']
        low = ohlcv_data['low']
        
        derived = {}
        
        # Price change metrics
        price_changes, pct_changes = self._calculate_price_changes(close)
        derived['price_change'] = price_changes
        derived['price_change_pct'] = pct_changes
        
        # High-low metrics
        derived['high_low_pct'] = ((high - low) / close) * 100
        
        # Pivot points (calculated for each day)
        derived['pivot_point'] = np.empty_like(close)
        derived['resistance_1'] = np.empty_like(close)
        derived['resistance_2'] = np.empty_like(close)
        derived['resistance_3'] = np.empty_like(close)
        derived['support_1'] = np.empty_like(close)
        derived['support_2'] = np.empty_like(close)
        derived['support_3'] = np.empty_like(close)
        
        for i in range(len(close)):
            if i == 0:
                # Use current values for first calculation
                h, l, c = high[i], low[i], close[i]
            else:
                # Use previous day's HLC for pivot calculation
                h, l, c = high[i-1], low[i-1], close[i-1]
            
            pivot, r1, r2, r3, s1, s2, s3 = NumbaIndicators.pivot_points(h, l, c)
            
            derived['pivot_point'][i] = pivot
            derived['resistance_1'][i] = r1
            derived['resistance_2'][i] = r2
            derived['resistance_3'][i] = r3
            derived['support_1'][i] = s1
            derived['support_2'][i] = s2
            derived['support_3'][i] = s3
        
        return derived
    
    def _calculate_futures_metrics(
        self,
        ohlcv_data: Dict[str, np.ndarray],
        spot_price: Optional[float]
    ) -> Dict[str, np.ndarray]:
        """Calculate futures-specific metrics"""
        
        close = ohlcv_data['close']
        volume = ohlcv_data['volume']
        oi = ohlcv_data['oi']
        
        metrics = {}
        
        # OI change metrics
        oi_changes = np.diff(oi, prepend=oi[0])
        metrics['oi_change'] = oi_changes
        
        oi_change_pct = np.where(oi[:-1] != 0, (oi_changes[1:] / oi[:-1]) * 100, 0)
        metrics['oi_change_pct'] = np.concatenate([[0], oi_change_pct])
        
        # Volume/OI ratio
        metrics['volume_oi_ratio'] = np.where(oi != 0, volume / oi, 0)
        
        # Volume change metrics
        volume_changes = np.diff(volume, prepend=volume[0])
        metrics['volume_change'] = volume_changes
        
        volume_change_pct = np.where(volume[:-1] != 0, (volume_changes[1:] / volume[:-1]) * 100, 0)
        metrics['volume_change_pct'] = np.concatenate([[0], volume_change_pct])
        
        # Basis calculations (if spot price available)
        if spot_price is not None:
            metrics['spot_price'] = np.full_like(close, spot_price)
            metrics['basis'] = close - spot_price
            metrics['basis_pct'] = ((close - spot_price) / spot_price) * 100
        
        return metrics
    
    def _calculate_price_changes(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate price changes and percentage changes"""
        
        changes = np.diff(prices, prepend=prices[0])
        changes[0] = 0.0  # First value has no change
        
        pct_changes = np.where(prices[:-1] != 0, (changes[1:] / prices[:-1]) * 100, 0)
        pct_changes = np.concatenate([[0], pct_changes])
        
        return changes, pct_changes
    
    def _parse_expiry_date(self, expiry_str: Optional[str]) -> Optional[datetime]:
        """Parse expiry date string"""
        if not expiry_str:
            return None
        
        try:
            return datetime.strptime(expiry_str, "%d-%b-%y")
        except Exception as e:
            logger.warning(f"Could not parse expiry date '{expiry_str}': {e}")
            return None
    
    def _calculate_time_to_expiry(self, current_time: datetime, expiry_date: Optional[datetime]) -> float:
        """Calculate time to expiry in years"""
        if not expiry_date:
            return 0.0
        
        time_diff = expiry_date - current_time
        return max(0.0, time_diff.total_seconds() / (365.25 * 24 * 3600))
    
    def _update_stats(self, calculation_time: float):
        """Update calculation statistics"""
        self.stats['calculations_performed'] += 1
        self.stats['total_calculation_time'] += calculation_time
        self.stats['average_calculation_time'] = (
            self.stats['total_calculation_time'] / self.stats['calculations_performed']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear indicator cache"""
        self.cache.clear()
        logger.info("Indicator cache cleared")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.clear_cache()
        logger.info("Indicator Engine cleaned up")

class BatchIndicatorProcessor:
    """Process indicators for multiple symbols in parallel"""
    
    def __init__(self, engine: IndicatorEngine):
        self.engine = engine
    
    async def process_symbols_batch(
        self,
        symbols_data: List[Tuple[SymbolInfo, List[HistoricalCandle], TimeFrame]],
        spot_prices: Optional[Dict[str, float]] = None,
        market_depth_data: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, List[IndicatorResult]]:
        """Process indicators for multiple symbols in parallel"""
        
        results = {}
        tasks = []
        
        for symbol_info, candles, timeframe in symbols_data:
            if symbol_info.instrument_type == InstrumentType.EQUITY:
                task = self.engine.calculate_equity_indicators(
                    symbol_info, candles, timeframe,
                    market_depth_data.get(symbol_info.symbol) if market_depth_data else None
                )
            elif symbol_info.instrument_type == InstrumentType.FUTURES:
                spot_price = spot_prices.get(symbol_info.symbol) if spot_prices else None
                task = self.engine.calculate_futures_indicators(
                    symbol_info, candles, timeframe, spot_price,
                    market_depth_data.get(symbol_info.symbol) if market_depth_data else None
                )
            elif symbol_info.instrument_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]:
                underlying = symbol_info.extract_underlying_symbol()
                spot_price = spot_prices.get(underlying) if spot_prices else 0.0
                task = self.engine.calculate_options_indicators(
                    symbol_info, candles, timeframe, spot_price,
                    market_depth_data=market_depth_data.get(symbol_info.symbol) if market_depth_data else None
                )
            else:
                # Index or other instrument types
                task = self.engine.calculate_equity_indicators(
                    symbol_info, candles, timeframe,
                    market_depth_data.get(symbol_info.symbol) if market_depth_data else None
                )
            
            tasks.append((symbol_info.symbol, task))
        
        # Execute all tasks in parallel
        for symbol, task in tasks:
            try:
                result = await task
                results[symbol] = result
            except Exception as e:
                logger.error(f"Error processing indicators for {symbol}: {e}")
                results[symbol] = []
        
        return results

# Utility functions for market depth processing
def process_market_depth_data(bid_ask_data: Dict) -> Dict[str, float]:
    """Process bid/ask market depth data into standardized format"""
    
    processed = {}
    
    # Process 5 levels of bid/ask data
    for i in range(1, 6):
        bid_key = f'bid_{i}'
        ask_key = f'ask_{i}'
        bid_qty_key = f'bid_qty_{i}'
        ask_qty_key = f'ask_qty_{i}'
        
        processed[bid_key] = bid_ask_data.get(bid_key, 0.0)
        processed[ask_key] = bid_ask_data.get(ask_key, 0.0)
        processed[bid_qty_key] = bid_ask_data.get(bid_qty_key, 0)
        processed[ask_qty_key] = bid_ask_data.get(ask_qty_key, 0)
    
    # Calculate derived metrics
    if processed['bid_1'] > 0 and processed['ask_1'] > 0:
        processed['bid_ask_spread'] = processed['ask_1'] - processed['bid_1']
        processed['mid_price'] = (processed['bid_1'] + processed['ask_1']) / 2.0
        processed['bid_ask_spread_pct'] = (processed['bid_ask_spread'] / processed['mid_price']) * 100
    else:
        processed['bid_ask_spread'] = 0.0
        processed['mid_price'] = 0.0
        processed['bid_ask_spread_pct'] = 0.0
    
    # Total quantities
    processed['total_bid_qty'] = sum(processed[f'bid_qty_{i}'] for i in range(1, 6))
    processed['total_ask_qty'] = sum(processed[f'ask_qty_{i}'] for i in range(1, 6))
    
    # Bid/ask ratio
    if processed['total_ask_qty'] > 0:
        processed['bid_ask_ratio'] = processed['total_bid_qty'] / processed['total_ask_qty']
    else:
        processed['bid_ask_ratio'] = 0.0
    
    return processed
