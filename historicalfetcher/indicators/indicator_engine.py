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

from historicalfetcher.utils.async_logger import get_async_logger

_async_logger = get_async_logger()
logger = _async_logger.get_logger()
from historicalfetcher.config.openalgo_settings import TimeFrame
from historicalfetcher.models.data_models import SymbolInfo, HistoricalCandle
from enum import Enum

class InstrumentType(str, Enum):
    """Instrument types for compatibility"""
    EQUITY = "EQ"
    FUTURES = "FUT"
    CALL_OPTION = "CE"
    PUT_OPTION = "PE"
    INDEX = "INDEX"

# Import NumbaIndicators with fallback
try:
    from historicalfetcher.indicators.numba_indicators import NumbaIndicators
except ImportError:
    # Use fallback from __init__.py
    from historicalfetcher.indicators import NumbaIndicators

# Initialize IV availability flag
_IV_AVAILABLE = False

try:
    from historicalfetcher.indicators.options_greeks import (
        OptionsGreeksCalculator, 
        ImpliedVolatilityCalculator, 
        AdvancedGreeks,
        _calculate_all_greeks  # Standalone function for JIT compatibility
    )
    _CALCULATE_ALL_GREEKS_AVAILABLE = True
    # Test if ImpliedVolatilityCalculator is actually callable (JIT compilation might have failed)
    try:
        # Try a simple test call to see if it's compiled (this will trigger JIT compilation)
        _test_iv = ImpliedVolatilityCalculator.implied_volatility_newton_raphson(
            100.0, 100.0, 0.1, 0.06, True, 0.0
        )
        _IV_AVAILABLE = True
        logger.debug("ImpliedVolatilityCalculator is available and working")
    except Exception as e:
        logger.warning(f"ImpliedVolatilityCalculator JIT compilation failed: {e}, will use fallback IV")
        _IV_AVAILABLE = False
        # Create fallback function
        class ImpliedVolatilityCalculator:
            @staticmethod
            def implied_volatility_newton_raphson(*args, **kwargs):
                return 0.2  # Default IV
except ImportError as e:
    logger.warning(f"Failed to import Numba-based options Greeks calculators: {e}, using pure Python fallback")
    _IV_AVAILABLE = True  # We have a pure Python fallback
    _CALCULATE_ALL_GREEKS_AVAILABLE = True  # We have a pure Python fallback
    
    # Pure Python implementation of Black-Scholes Greeks (without Numba)
    import math
    
    def _norm_cdf(x):
        """Standard normal CDF using error function"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    def _norm_pdf(x):
        """Standard normal PDF"""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    
    def _d1_d2(S, K, T, r, sigma, q=0.0):
        """Calculate d1 and d2 for Black-Scholes"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0, 0.0
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2
    
    def _calculate_all_greeks(S, K, T, r, sigma, is_call, q=0.0):
        """Pure Python Black-Scholes Greeks calculation"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            intrinsic = max(0, S - K) if is_call else max(0, K - S)
            moneyness = S / K if is_call and K > 0 else K / S if not is_call and S > 0 else 0
            return (intrinsic, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, intrinsic, 0.0, moneyness, 0.5 if abs(S - K) < 0.01 * K else (1.0 if (is_call and S > K) or (not is_call and S < K) else 0.0))
        
        d1, d2 = _d1_d2(S, K, T, r, sigma, q)
        sqrt_T = math.sqrt(T)
        exp_qT = math.exp(-q * T)
        exp_rT = math.exp(-r * T)
        
        # Option price
        if is_call:
            price = S * exp_qT * _norm_cdf(d1) - K * exp_rT * _norm_cdf(d2)
            delta = exp_qT * _norm_cdf(d1)
            theta = (-S * sigma * exp_qT * _norm_pdf(d1) / (2 * sqrt_T) 
                    - r * K * exp_rT * _norm_cdf(d2) 
                    + q * S * exp_qT * _norm_cdf(d1)) / 365
            rho = K * T * exp_rT * _norm_cdf(d2) / 100
        else:
            price = K * exp_rT * _norm_cdf(-d2) - S * exp_qT * _norm_cdf(-d1)
            delta = -exp_qT * _norm_cdf(-d1)
            theta = (-S * sigma * exp_qT * _norm_pdf(d1) / (2 * sqrt_T) 
                    + r * K * exp_rT * _norm_cdf(-d2) 
                    - q * S * exp_qT * _norm_cdf(-d1)) / 365
            rho = -K * T * exp_rT * _norm_cdf(-d2) / 100
        
        # Greeks (same for calls and puts)
        gamma = exp_qT * _norm_pdf(d1) / (S * sigma * sqrt_T)
        vega = S * exp_qT * _norm_pdf(d1) * sqrt_T / 100
        
        # Derived metrics
        lambda_val = delta * S / price if price > 0 else 0.0
        intrinsic = max(0, S - K) if is_call else max(0, K - S)
        time_val = max(0, price - intrinsic)
        moneyness = S / K if is_call else K / S
        prob_itm = _norm_cdf(d2) if is_call else _norm_cdf(-d2)
        
        return (price, delta, gamma, theta, vega, rho, lambda_val, intrinsic, time_val, moneyness, prob_itm)
    
    class ImpliedVolatilityCalculator:
        @staticmethod
        def implied_volatility_newton_raphson(market_price, S, K, T, r, is_call, q=0.0, max_iter=100, tol=1e-6):
            """Newton-Raphson IV calculation (pure Python)"""
            if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
                return 0.2
            
            sigma = 0.2  # Initial guess
            for _ in range(max_iter):
                price, _, _, _, vega, _, _, _, _, _, _ = _calculate_all_greeks(S, K, T, r, sigma, is_call, q)
                vega_actual = vega * 100  # Undo the /100 scaling
                
                if abs(vega_actual) < 1e-10:
                    break
                    
                diff = market_price - price
                if abs(diff) < tol:
                    break
                    
                sigma = sigma + diff / vega_actual
                sigma = max(0.001, min(5.0, sigma))  # Keep sigma in reasonable bounds
            
            return sigma
    
    class OptionsGreeksCalculator:
        @staticmethod
        def calculate_all_greeks(S, K, T, r, sigma, is_call, q=0.0):
            return _calculate_all_greeks(S, K, T, r, sigma, is_call, q)
        
        @staticmethod
        def intrinsic_value(S, K, is_call):
            return max(0, S - K) if is_call else max(0, K - S)
        
        @staticmethod
        def moneyness(S, K, is_call):
            return S / K if is_call and K > 0 else K / S if not is_call and S > 0 else 0
    
    class AdvancedGreeks:
        @staticmethod
        def charm(S, K, T, r, sigma, is_call, q=0.0):
            if T <= 0 or sigma <= 0:
                return 0.0
            d1, d2 = _d1_d2(S, K, T, r, sigma, q)
            exp_qT = math.exp(-q * T)
            sqrt_T = math.sqrt(T)
            return -exp_qT * _norm_pdf(d1) * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
        
        @staticmethod
        def vanna(S, K, T, r, sigma, q=0.0):
            if T <= 0 or sigma <= 0:
                return 0.0
            d1, d2 = _d1_d2(S, K, T, r, sigma, q)
            exp_qT = math.exp(-q * T)
            sqrt_T = math.sqrt(T)
            return -exp_qT * _norm_pdf(d1) * d2 / sigma
        
        @staticmethod
        def volga(S, K, T, r, sigma, q=0.0):
            if T <= 0 or sigma <= 0:
                return 0.0
            d1, d2 = _d1_d2(S, K, T, r, sigma, q)
            exp_qT = math.exp(-q * T)
            sqrt_T = math.sqrt(T)
            vega = S * exp_qT * _norm_pdf(d1) * sqrt_T / 100
            return vega * d1 * d2 / sigma
from historicalfetcher.models.data_models import TimeFrameCode, OptionTypeCode, IndicatorResult, CalculationConfig

# Logger is imported from loguru above

# IndicatorResult and CalculationConfig are now imported from historicalfetcher.models.data_models

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
        
        # Initialize Numba indicators
        self.numba_indicators = NumbaIndicators()
        
        # Initialize options calculator if needed
        if self.config.enable_greeks:
            try:
                self.options_calculator = OptionsGreeksCalculator()
            except Exception as e:
                logger.warning(f"Options calculator initialization failed: {e}")
                self.options_calculator = None
        
    
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
            tf_string = TimeFrameCode.to_string(timeframe)
            
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
                    timeframe=tf_string,
                    timestamp=candle.timestamp,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    oi=candle.oi,
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
            
            # Log Greeks calculation status
            if greeks_data:
                logger.debug(f"Greeks calculated for {symbol_info.symbol}: {list(greeks_data.keys())}")
            else:
                logger.warning(f"No Greeks calculated for {symbol_info.symbol} - check calculate_greeks config")
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(ohlcv_data, indicators)
            
            # Create results
            results = []
            tf_string = TimeFrameCode.to_string(timeframe)
            
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
                    timeframe=tf_string,
                    timestamp=candle.timestamp,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    oi=candle.oi,
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
            tf_string = TimeFrameCode.to_string(timeframe)
            
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
                    timeframe=tf_string,
                    timestamp=candle.timestamp,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    oi=candle.oi,
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
            # Bollinger Bands width - avoid division by zero
            indicators['bb_width'] = np.where(
                bb_middle != 0,
                (bb_upper - bb_lower) / bb_middle * 100,
                0.0
            )
            # Bollinger Bands percent - avoid division by zero
            bb_range = bb_upper - bb_lower
            indicators['bb_percent'] = np.where(
                bb_range != 0,
                (close - bb_lower) / bb_range * 100,
                0.0
            )
        
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
        
        # Normalize instrument_type for comparison
        inst_type_str = str(instrument_type).upper() if isinstance(instrument_type, str) else instrument_type
        is_equity = inst_type_str in [InstrumentType.EQUITY, 'EQ']
        is_futures = inst_type_str in [InstrumentType.FUTURES, 'FUT']
        is_options = inst_type_str in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION, 'CE', 'PE']
        is_index = inst_type_str in [InstrumentType.INDEX, 'INDEX']
        
        # Ichimoku Cloud (Equity, Futures, Index)
        if is_equity or is_futures or is_index:
            ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a, ichimoku_senkou_b, ichimoku_chikou, ichimoku_cloud_top, ichimoku_cloud_bottom, ichimoku_cloud_color = NumbaIndicators.ichimoku(high, low, close)
            indicators['ichimoku_tenkan_sen'] = ichimoku_tenkan
            indicators['ichimoku_kijun_sen'] = ichimoku_kijun
            indicators['ichimoku_senkou_span_a'] = ichimoku_senkou_a
            indicators['ichimoku_senkou_span_b'] = ichimoku_senkou_b
            indicators['ichimoku_chikou_span'] = ichimoku_chikou
            indicators['ichimoku_cloud_top'] = ichimoku_cloud_top
            indicators['ichimoku_cloud_bottom'] = ichimoku_cloud_bottom
            indicators['ichimoku_cloud_color'] = ichimoku_cloud_color.astype(np.float64)
            # Ichimoku signal: 1=Buy, 0=Sell, 2=Neutral
            ichimoku_signal = np.zeros_like(close, dtype=np.int8)
            for i in range(len(close)):
                if not (np.isnan(ichimoku_cloud_top[i]) or np.isnan(ichimoku_cloud_bottom[i])):
                    if close[i] > ichimoku_cloud_top[i]:
                        ichimoku_signal[i] = 1
                    elif close[i] < ichimoku_cloud_bottom[i]:
                        ichimoku_signal[i] = 0
                    else:
                        ichimoku_signal[i] = 2
            indicators['ichimoku_signal'] = ichimoku_signal.astype(np.float64)
        
        # Aroon Indicator (Equity, Futures, Index)
        if is_equity or is_futures or is_index:
            aroon_up, aroon_down, aroon_oscillator = NumbaIndicators.aroon(high, low, 25)
            indicators['aroon_up'] = aroon_up
            indicators['aroon_down'] = aroon_down
            indicators['aroon_oscillator'] = aroon_oscillator
            # Aroon signal: 1=Strong uptrend, -1=Strong downtrend, 0=Neutral
            aroon_signal = np.zeros_like(close, dtype=np.int8)
            for i in range(len(close)):
                if not (np.isnan(aroon_up[i]) or np.isnan(aroon_down[i])):
                    if aroon_up[i] > 70 and aroon_down[i] < 30:
                        aroon_signal[i] = 1
                    elif aroon_down[i] > 70 and aroon_up[i] < 30:
                        aroon_signal[i] = -1
                    else:
                        aroon_signal[i] = 0
            indicators['aroon_signal'] = aroon_signal.astype(np.float64)
        
        # Volume-based indicators (not for Index)
        if not is_index:
            # Accumulation/Distribution Line (Equity, Futures)
            if is_equity or is_futures:
                ad_line, ad_line_slope, ad_line_signal = NumbaIndicators.ad_line(high, low, close, volume)
                indicators['ad_line'] = ad_line
                indicators['ad_line_slope'] = ad_line_slope
                indicators['ad_line_signal'] = ad_line_signal.astype(np.float64)
            
            # Chaikin Money Flow (Equity, Futures)
            if is_equity or is_futures:
                cmf_20, cmf_signal = NumbaIndicators.cmf(high, low, close, volume, 20)
                indicators['cmf_20'] = cmf_20
                indicators['cmf_signal'] = cmf_signal.astype(np.float64)
            
            # Volume Profile (Equity, Futures, Options)
            if is_equity or is_futures or is_options:
                volume_profile_poc, volume_profile_vah, volume_profile_val, volume_profile_balance = NumbaIndicators.volume_profile(high, low, close, volume, 20)
                indicators['volume_profile_poc'] = volume_profile_poc
                indicators['volume_profile_vah'] = volume_profile_vah
                indicators['volume_profile_val'] = volume_profile_val
                indicators['volume_profile_balance'] = volume_profile_balance
            
            # TWAP (Equity, Futures, Options)
            if is_equity or is_futures or is_options:
                twap = NumbaIndicators.twap(high, low, close)
                indicators['twap'] = twap
            
            # Volume Divergence (Equity, Futures, Options)
            if is_equity or is_futures or is_options:
                # Get RSI and MACD for divergence calculation
                rsi = indicators.get('rsi_14', NumbaIndicators.rsi(close, 14))
                macd_line = indicators.get('macd_line', NumbaIndicators.macd(close, 12, 26, 9)[0])
                volume_price_div, volume_div_strength, volume_div_confirmed, rsi_vol_div, macd_vol_div, price_vol_div_type = NumbaIndicators.volume_divergence(close, volume, rsi, macd_line, 14)
                indicators['volume_price_divergence'] = volume_price_div.astype(np.float64)
                indicators['volume_divergence_strength'] = volume_div_strength
                indicators['volume_divergence_confirmed'] = volume_div_confirmed.astype(np.float64)
                indicators['rsi_volume_divergence'] = rsi_vol_div.astype(np.float64)
                indicators['macd_volume_divergence'] = macd_vol_div.astype(np.float64)
                indicators['price_volume_divergence_type'] = price_vol_div_type.astype(np.float64)
        
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
            logger.debug(f"Greeks calculation disabled for {symbol_info.symbol} (calculate_greeks=False)")
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
        
        # Validate strike and spot price - critical for Greeks calculation
        if strike <= 0:
            logger.warning(f"Invalid strike price ({strike}) for {symbol_info.symbol}, cannot calculate Greeks")
            # Return arrays filled with 0 for all Greeks
            for key in greeks:
                greeks[key].fill(0.0)
            return greeks
        
        if spot_price <= 0:
            logger.warning(f"Invalid spot price ({spot_price}) for {symbol_info.symbol}, cannot calculate Greeks")
            for key in greeks:
                greeks[key].fill(0.0)
            return greeks
        
        # Determine if call or put - handle both string and enum types
        inst_type = symbol_info.instrument_type
        if isinstance(inst_type, str):
            is_call = inst_type.upper() in ['CE', 'CALL', 'CALL_OPTION']
        else:
            is_call = inst_type == InstrumentType.CALL_OPTION
        
        # Calculate expiry time for each candle
        expiry_date = self._parse_expiry_date(symbol_info.expiry)
        
        # Validate expiry date
        if expiry_date is None:
            logger.warning(f"Could not parse expiry for {symbol_info.symbol} (expiry='{symbol_info.expiry}'), cannot calculate Greeks")
            for key in greeks:
                greeks[key].fill(0.0)
            return greeks
        
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
            # Initialize iv to default value first (always available for Greeks calculation)
            iv = 0.2  # Default 20% volatility
            
            if self.config.calculate_iv:
                # Check if IV calculator is available and working
                if _IV_AVAILABLE:
                    try:
                        iv = ImpliedVolatilityCalculator.implied_volatility_newton_raphson(
                            candle.close, spot_price, strike, time_to_expiry, risk_free_rate, is_call, dividend_yield
                        )
                        greeks['implied_volatility'][i] = iv
                    except Exception as e:
                        logger.warning(f"Error calculating IV for {symbol_info.symbol}: {e}, using default")
                        iv = 0.2
                        greeks['implied_volatility'][i] = iv
                else:
                    # IV calculator not available, use default
                    iv = 0.2
                    greeks['implied_volatility'][i] = iv
            else:
                # IV calculation disabled, use default
                iv = 0.2  # Default 20% volatility
            
            # Calculate all Greeks using standalone function (avoids Numba class method issues)
            if _CALCULATE_ALL_GREEKS_AVAILABLE:
                (option_price, delta, gamma, theta, vega, rho, lambda_val,
                 intrinsic_val, time_val, moneyness_val, prob_itm) = _calculate_all_greeks(
                    spot_price, strike, time_to_expiry, risk_free_rate, iv, is_call, dividend_yield
                )
            else:
                # Fallback to class method if standalone function not available
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
        
        # High-low metrics - avoid division by zero
        derived['high_low_pct'] = np.where(
            close != 0,
            ((high - low) / close) * 100,
            0.0
        )
        
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
        if spot_price is not None and spot_price != 0:
            metrics['spot_price'] = np.full_like(close, spot_price)
            metrics['basis'] = close - spot_price
            metrics['basis_pct'] = ((close - spot_price) / spot_price) * 100
        elif spot_price is not None:
            # Spot price is zero, set basis to zero
            metrics['spot_price'] = np.full_like(close, spot_price)
            metrics['basis'] = close - spot_price
            metrics['basis_pct'] = np.zeros_like(close)
        
        return metrics
    
    def _calculate_price_changes(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate price changes and percentage changes"""
        
        changes = np.diff(prices, prepend=prices[0])
        changes[0] = 0.0  # First value has no change
        
        pct_changes = np.where(prices[:-1] != 0, (changes[1:] / prices[:-1]) * 100, 0)
        pct_changes = np.concatenate([[0], pct_changes])
        
        return changes, pct_changes
    
    def _parse_expiry_date(self, expiry_str: Optional[str]) -> Optional[datetime]:
        """Parse expiry date string supporting multiple formats"""
        if not expiry_str:
            return None
        
        # Try multiple date formats
        formats_to_try = [
            "%d-%b-%y",     # 28-Nov-24
            "%d-%b-%Y",     # 28-Nov-2024
            "%Y-%m-%d",     # 2024-11-28
            "%d/%m/%Y",     # 28/11/2024
            "%d/%m/%y",     # 28/11/24
        ]
        
        for fmt in formats_to_try:
            try:
                return datetime.strptime(expiry_str, fmt)
            except ValueError:
                continue
        
        # Try DDMMMYY format (e.g., "28NOV24") - common in Indian markets
        try:
            if len(expiry_str) == 7 and expiry_str[2:5].isalpha():
                day = int(expiry_str[:2])
                month_str = expiry_str[2:5].upper()
                year = int('20' + expiry_str[5:7])
                month_map = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }
                month = month_map.get(month_str)
                if month:
                    return datetime(year, month, day)
        except (ValueError, IndexError):
            pass
        
        # Try DDMMMYYYY format (e.g., "28NOV2024")
        try:
            if len(expiry_str) == 9 and expiry_str[2:5].isalpha():
                day = int(expiry_str[:2])
                month_str = expiry_str[2:5].upper()
                year = int(expiry_str[5:9])
                month_map = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }
                month = month_map.get(month_str)
                if month:
                    return datetime(year, month, day)
        except (ValueError, IndexError):
            pass
        
        logger.warning(f"Could not parse expiry date '{expiry_str}' with any known format")
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
    
    def _candles_to_arrays(self, candles: List[HistoricalCandle]) -> Dict[str, np.ndarray]:
        """Convert candles to numpy arrays for efficient processing"""
        
        opens = np.array([c.open for c in candles], dtype=np.float64)
        highs = np.array([c.high for c in candles], dtype=np.float64)
        lows = np.array([c.low for c in candles], dtype=np.float64)
        closes = np.array([c.close for c in candles], dtype=np.float64)
        volumes = np.array([c.volume for c in candles], dtype=np.int64)
        
        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'timestamps': [c.timestamp for c in candles]
        }
    
    async def _calculate_technical_indicators(
        self, 
        ohlcv_data: Dict[str, np.ndarray], 
        timeframe: TimeFrame, 
        instrument_type: str
    ) -> Dict[str, np.ndarray]:
        """Calculate technical indicators using Numba-optimized functions"""
        
        indicators = {}
        
        try:
            closes = ohlcv_data['close']
            highs = ohlcv_data['high']
            lows = ohlcv_data['low']
            volumes = ohlcv_data['volume']
            
            # Moving Averages
            if self.config.enable_ema:
                indicators['ema_9'] = self.numba_indicators.ema(closes, 9)
                indicators['ema_21'] = self.numba_indicators.ema(closes, 21)
                indicators['ema_50'] = self.numba_indicators.ema(closes, 50)
                indicators['ema_200'] = self.numba_indicators.ema(closes, 200)
            
            if self.config.enable_sma:
                indicators['sma_20'] = self.numba_indicators.sma(closes, 20)
                indicators['sma_50'] = self.numba_indicators.sma(closes, 50)
            
            # Momentum Indicators
            if self.config.enable_rsi:
                indicators['rsi_14'] = self.numba_indicators.rsi(closes, 14)
            
            if self.config.enable_macd:
                macd_line, macd_signal, macd_histogram = self.numba_indicators.macd(closes, 12, 26, 9)
                indicators['macd_line'] = macd_line
                indicators['macd_signal'] = macd_signal
                indicators['macd_histogram'] = macd_histogram
            
            # Volatility Indicators
            if self.config.enable_atr:
                indicators['atr_14'] = self.numba_indicators.atr(highs, lows, closes, 14)
            
            if self.config.enable_bollinger:
                bb_upper, bb_middle, bb_lower = self.numba_indicators.bollinger_bands(closes, 20, 2.0)
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
                # Bollinger Bands width - avoid division by zero
                indicators['bb_width'] = np.where(
                    bb_middle != 0,
                    (bb_upper - bb_lower) / bb_middle * 100,
                    0.0
                )
                # Bollinger Bands percent - avoid division by zero
                bb_range = bb_upper - bb_lower
                indicators['bb_percent'] = np.where(
                    bb_range != 0,
                    (closes - bb_lower) / bb_range * 100,
                    0.0
                )
            
            # Stochastic
            if self.config.enable_stochastic:
                stoch_k, stoch_d = self.numba_indicators.stochastic(highs, lows, closes, 14, 3)
                indicators['stoch_k'] = stoch_k
                indicators['stoch_d'] = stoch_d
            
            # Volume indicators
            indicators['volume_sma_20'] = self.numba_indicators.sma(volumes.astype(np.float64), 20)
            indicators['vwap'] = self._calculate_vwap(ohlcv_data)
            indicators['obv'] = self._calculate_obv(closes, volumes)
            
            # Trend following indicators
            indicators['supertrend_7_3'], indicators['supertrend_signal_7_3'] = self._calculate_supertrend(
                highs, lows, closes, 7, 3.0
            )
            indicators['supertrend_10_3'], indicators['supertrend_signal_10_3'] = self._calculate_supertrend(
                highs, lows, closes, 10, 3.0
            )
            indicators['parabolic_sar'] = self._calculate_parabolic_sar(highs, lows, closes)
            
            # Pivot points (daily only)
            if timeframe == TimeFrame.DAILY:
                pivot_data = self._calculate_pivot_points(highs, lows, closes)
                indicators.update(pivot_data)
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            # Return empty indicators on error
            return {}
        
        return indicators
    
    def _calculate_derived_metrics(
        self, 
        ohlcv_data: Dict[str, np.ndarray], 
        indicators: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Calculate derived metrics from OHLCV and indicators"""
        
        derived = {}
        
        try:
            closes = ohlcv_data['close']
            highs = ohlcv_data['high']
            lows = ohlcv_data['low']
            
            # Price changes
            price_changes = np.diff(closes, prepend=closes[0])
            derived['price_change'] = price_changes
            
            # Price change percentage - avoid division by zero
            prev_closes = np.roll(closes, 1)
            prev_closes[0] = closes[0]  # First value uses current close
            derived['price_change_pct'] = np.where(
                prev_closes != 0,
                (price_changes / prev_closes) * 100,
                0.0
            )
            derived['price_change_pct'][0] = 0.0  # First value is 0
            
            # High-Low percentage - avoid division by zero
            derived['high_low_pct'] = np.where(
                closes != 0,
                ((highs - lows) / closes) * 100,
                0.0
            )
            
        except Exception as e:
            logger.error(f"Error calculating derived metrics: {e}")
        
        return derived
    
    def _calculate_vwap(self, ohlcv_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Volume Weighted Average Price"""
        
        typical_price = (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
        volume = ohlcv_data['volume'].astype(np.float64)
        
        cumulative_pv = np.cumsum(typical_price * volume)
        cumulative_volume = np.cumsum(volume)
        
        # Avoid division by zero and NaN warnings
        # Use np.divide with where parameter to suppress warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            vwap = np.divide(
                cumulative_pv, 
                cumulative_volume, 
                out=np.full_like(cumulative_pv, np.nan), 
                where=(cumulative_volume != 0)
            )
            # Fill NaN values with typical_price
            vwap = np.where(np.isnan(vwap), typical_price, vwap)
        
        return vwap
    
    def _calculate_obv(self, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate On Balance Volume"""
        
        obv = np.zeros_like(volumes, dtype=np.float64)
        obv[0] = volumes[0]
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def _calculate_supertrend(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray, 
        period: int, 
        multiplier: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Supertrend indicator"""
        
        # Calculate ATR
        atr = self.numba_indicators.atr(highs, lows, closes, period)
        
        # Calculate basic bands
        hl2 = (highs + lows) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize arrays
        supertrend = np.zeros_like(closes)
        signals = np.zeros_like(closes, dtype=np.int8)
        
        # Calculate supertrend
        for i in range(1, len(closes)):
            # Upper band calculation
            if upper_band[i] < upper_band[i-1] or closes[i-1] > upper_band[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = upper_band[i-1]
            
            # Lower band calculation
            if lower_band[i] > lower_band[i-1] or closes[i-1] < lower_band[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = lower_band[i-1]
            
            # Supertrend calculation
            if closes[i] <= lower_band[i]:
                supertrend[i] = lower_band[i]
                signals[i] = 0  # Sell signal
            elif closes[i] >= upper_band[i]:
                supertrend[i] = upper_band[i]
                signals[i] = 1  # Buy signal
            else:
                supertrend[i] = supertrend[i-1]
                signals[i] = signals[i-1]
        
        return supertrend, signals
    
    def _calculate_parabolic_sar(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.2
    ) -> np.ndarray:
        """Calculate Parabolic SAR"""
        
        sar = np.zeros_like(closes)
        trend = np.ones_like(closes, dtype=np.int8)  # 1 for uptrend, -1 for downtrend
        af = af_start
        ep = highs[0]  # Extreme point
        
        sar[0] = lows[0]
        
        for i in range(1, len(closes)):
            # Calculate SAR
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            
            # Check for trend reversal
            if trend[i-1] == 1:  # Uptrend
                if lows[i] <= sar[i]:
                    # Trend reversal to downtrend
                    trend[i] = -1
                    sar[i] = ep
                    ep = lows[i]
                    af = af_start
                else:
                    trend[i] = 1
                    if highs[i] > ep:
                        ep = highs[i]
                        af = min(af + af_increment, af_max)
            else:  # Downtrend
                if highs[i] >= sar[i]:
                    # Trend reversal to uptrend
                    trend[i] = 1
                    sar[i] = ep
                    ep = highs[i]
                    af = af_start
                else:
                    trend[i] = -1
                    if lows[i] < ep:
                        ep = lows[i]
                        af = min(af + af_increment, af_max)
        
        return sar
    
    def _calculate_pivot_points(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate daily pivot points"""
        
        # Use previous day's HLC for pivot calculation
        prev_high = np.roll(highs, 1)
        prev_low = np.roll(lows, 1)
        prev_close = np.roll(closes, 1)
        
        # Set first values
        prev_high[0] = highs[0]
        prev_low[0] = lows[0]
        prev_close[0] = closes[0]
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Calculate support and resistance levels
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        
        return {
            'pivot_point': pivot,
            'resistance_1': r1,
            'resistance_2': r2,
            'resistance_3': r3,
            'support_1': s1,
            'support_2': s2,
            'support_3': s3
        }
    
    def _calculate_futures_metrics(
        self, 
        ohlcv_data: Dict[str, np.ndarray], 
        spot_price: Optional[float]
    ) -> Dict[str, np.ndarray]:
        """Calculate futures-specific metrics"""
        
        metrics = {}
        
        if spot_price:
            closes = ohlcv_data['close']
            
            # Basis calculation
            basis = closes - spot_price
            basis_pct = (basis / spot_price) * 100
            
            metrics['basis'] = np.full_like(closes, basis[-1])  # Use latest basis
            metrics['basis_pct'] = np.full_like(closes, basis_pct[-1])
            
            # Simple cost of carry estimation (placeholder)
            metrics['cost_of_carry'] = np.full_like(closes, 0.06)  # 6% annual rate
        
        return metrics

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
            inst_type = symbol_info.instrument_type
            if inst_type in [InstrumentType.EQUITY, 'EQ']:
                task = self.engine.calculate_equity_indicators(
                    symbol_info, candles, timeframe,
                    market_depth_data.get(symbol_info.symbol) if market_depth_data else None
                )
            elif inst_type in [InstrumentType.FUTURES, 'FUT']:
                spot_price = spot_prices.get(symbol_info.symbol) if spot_prices else None
                task = self.engine.calculate_futures_indicators(
                    symbol_info, candles, timeframe, spot_price,
                    market_depth_data.get(symbol_info.symbol) if market_depth_data else None
                )
            elif inst_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION, 'CE', 'PE']:
                # Extract underlying from option symbol
                import re
                underlying = symbol_info.symbol
                match = re.match(r"^([A-Z]+)(\d{2}[A-Z]{3}\d{2}[\d.]+)(CE|PE)$", symbol_info.symbol.upper())
                if match:
                    underlying = match.group(1)
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
        # Bid-ask spread percentage - avoid division by zero
        if processed['mid_price'] != 0:
            processed['bid_ask_spread_pct'] = (processed['bid_ask_spread'] / processed['mid_price']) * 100
        else:
            processed['bid_ask_spread_pct'] = 0.0
    else:
        processed['bid_ask_spread'] = 0.0
        processed['mid_price'] = 0.0
        processed['bid_ask_spread_pct'] = 0.0
    
    # Total quantities
    processed['total_bid_qty'] = sum(processed[f'bid_qty_{i}'] for i in range(1, 6))
    processed['total_ask_qty'] = sum(processed[f'ask_qty_{i}'] for i in range(1, 6))
    
    # Bid/ask ratio - avoid division by zero
    if processed['total_ask_qty'] > 0:
        processed['bid_ask_ratio'] = processed['total_bid_qty'] / processed['total_ask_qty']
    else:
        processed['bid_ask_ratio'] = 0.0
    
    return processed
