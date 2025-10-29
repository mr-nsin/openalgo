"""
Numba-Optimized Technical Indicators

High-performance technical indicator calculations using Numba JIT compilation
for maximum speed in processing large datasets.
"""

import numpy as np
import numba
from numba import jit, prange
import math
from typing import Tuple, Optional
import sys
import os

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from utils.logging import get_logger

logger = get_logger(__name__)

# Numba optimization settings for maximum performance
NUMBA_CONFIG = {
    'nopython': True,           # Pure machine code compilation
    'nogil': True,              # Release Python GIL
    'cache': True,              # Cache compiled functions
    'fastmath': True,           # Enable fast math optimizations
    'boundscheck': False,       # Disable bounds checking
    'wraparound': False,        # Disable negative indexing
}

class NumbaIndicators:
    """High-performance technical indicators using Numba JIT compilation"""
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Exponential Moving Average with Numba optimization
        
        Args:
            prices: Array of prices
            period: EMA period
            
        Returns:
            Array of EMA values
        """
        alpha = 2.0 / (period + 1.0)
        ema_values = np.empty_like(prices)
        ema_values[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema_values[i] = alpha * prices[i] + (1.0 - alpha) * ema_values[i-1]
        
        return ema_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        sma_values = np.empty_like(prices)
        sma_values[:period-1] = np.nan
        
        for i in range(period-1, len(prices)):
            sma_values[i] = np.mean(prices[i-period+1:i+1])
        
        return sma_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Relative Strength Index with Numba optimization
        """
        rsi_values = np.empty_like(prices)
        rsi_values[:period] = np.nan
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            rsi_values[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))
        
        # Calculate RSI for remaining periods using Wilder's smoothing
        alpha = 1.0 / period
        
        for i in range(period + 1, len(prices)):
            gain = gains[i-1]
            loss = losses[i-1]
            
            avg_gain = alpha * gain + (1.0 - alpha) * avg_gain
            avg_loss = alpha * loss + (1.0 - alpha) * avg_loss
            
            if avg_loss == 0:
                rsi_values[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD (Moving Average Convergence Divergence)
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = NumbaIndicators.ema(prices, fast)
        ema_slow = NumbaIndicators.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = NumbaIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Average True Range with Numba optimization
        """
        atr_values = np.empty_like(close)
        atr_values[0] = high[0] - low[0]
        
        # Calculate True Range
        tr = np.empty_like(close)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        # Calculate ATR using Wilder's smoothing
        atr_values[0] = tr[0]
        alpha = 1.0 / period
        
        for i in range(1, len(close)):
            if i < period:
                atr_values[i] = np.mean(tr[:i+1])
            else:
                atr_values[i] = alpha * tr[i] + (1.0 - alpha) * atr_values[i-1]
        
        return atr_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle_band = NumbaIndicators.sma(prices, period)
        
        # Calculate standard deviation
        std_values = np.empty_like(prices)
        std_values[:period-1] = np.nan
        
        for i in range(period-1, len(prices)):
            window = prices[i-period+1:i+1]
            std_values[i] = np.std(window)
        
        upper_band = middle_band + (std_dev * std_values)
        lower_band = middle_band - (std_dev * std_values)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Supertrend Indicator
        
        Returns:
            Tuple of (supertrend_values, signals) where signals: 1=Buy, 0=Sell
        """
        atr_values = NumbaIndicators.atr(high, low, close, period)
        
        # Calculate basic bands
        hl2 = (high + low) / 2.0
        upper_band = hl2 + (multiplier * atr_values)
        lower_band = hl2 - (multiplier * atr_values)
        
        # Initialize arrays
        final_upper = np.empty_like(close)
        final_lower = np.empty_like(close)
        supertrend = np.empty_like(close)
        signals = np.empty_like(close, dtype=np.int8)
        
        # First values
        final_upper[0] = upper_band[0]
        final_lower[0] = lower_band[0]
        supertrend[0] = final_upper[0]
        signals[0] = 0
        
        for i in range(1, len(close)):
            # Calculate final bands
            if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = final_upper[i-1]
            
            if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = final_lower[i-1]
            
            # Determine supertrend value and signal
            if supertrend[i-1] == final_upper[i-1] and close[i] < final_upper[i]:
                supertrend[i] = final_upper[i]
                signals[i] = 0  # Sell signal
            elif supertrend[i-1] == final_upper[i-1] and close[i] >= final_upper[i]:
                supertrend[i] = final_lower[i]
                signals[i] = 1  # Buy signal
            elif supertrend[i-1] == final_lower[i-1] and close[i] > final_lower[i]:
                supertrend[i] = final_lower[i]
                signals[i] = 1  # Buy signal
            else:
                supertrend[i] = final_upper[i]
                signals[i] = 0  # Sell signal
        
        return supertrend, signals
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Oscillator
        
        Returns:
            Tuple of (%K, %D)
        """
        k_values = np.empty_like(close)
        k_values[:k_period-1] = np.nan
        
        # Calculate %K
        for i in range(k_period-1, len(close)):
            highest_high = np.max(high[i-k_period+1:i+1])
            lowest_low = np.min(low[i-k_period+1:i+1])
            
            if highest_high == lowest_low:
                k_values[i] = 50.0
            else:
                k_values[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100.0
        
        # Calculate %D (SMA of %K)
        d_values = NumbaIndicators.sma(k_values, d_period)
        
        return k_values, d_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        Volume Weighted Average Price
        """
        typical_price = (high + low + close) / 3.0
        vwap_values = np.empty_like(close)
        
        cumulative_volume = 0.0
        cumulative_pv = 0.0
        
        for i in range(len(close)):
            cumulative_volume += volume[i]
            cumulative_pv += typical_price[i] * volume[i]
            
            if cumulative_volume > 0:
                vwap_values[i] = cumulative_pv / cumulative_volume
            else:
                vwap_values[i] = typical_price[i]
        
        return vwap_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On Balance Volume"""
        obv_values = np.empty_like(close)
        obv_values[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv_values[i] = obv_values[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv_values[i] = obv_values[i-1] - volume[i]
            else:
                obv_values[i] = obv_values[i-1]
        
        return obv_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def pivot_points(high: float, low: float, close: float) -> Tuple[float, float, float, float, float, float, float]:
        """
        Calculate pivot points and support/resistance levels
        
        Returns:
            Tuple of (pivot, r1, r2, r3, s1, s2, s3)
        """
        pivot = (high + low + close) / 3.0
        
        r1 = 2.0 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2.0 * (pivot - low)
        
        s1 = 2.0 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2.0 * (high - pivot)
        
        return pivot, r1, r2, r3, s1, s2, s3
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Average Directional Index (ADX) with DI+ and DI-
        
        Returns:
            Tuple of (adx, di_plus, di_minus)
        """
        # Calculate True Range
        tr = np.empty_like(close)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        # Calculate Directional Movement
        dm_plus = np.empty_like(close)
        dm_minus = np.empty_like(close)
        dm_plus[0] = 0.0
        dm_minus[0] = 0.0
        
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            else:
                dm_plus[i] = 0.0
            
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
            else:
                dm_minus[i] = 0.0
        
        # Smooth the values
        atr_smooth = NumbaIndicators.ema(tr, period)
        dm_plus_smooth = NumbaIndicators.ema(dm_plus, period)
        dm_minus_smooth = NumbaIndicators.ema(dm_minus, period)
        
        # Calculate DI+ and DI-
        di_plus = 100.0 * dm_plus_smooth / atr_smooth
        di_minus = 100.0 * dm_minus_smooth / atr_smooth
        
        # Calculate ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100.0
        adx = NumbaIndicators.ema(dx, period)
        
        return adx, di_plus, di_minus
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Williams %R"""
        wr_values = np.empty_like(close)
        wr_values[:period-1] = np.nan
        
        for i in range(period-1, len(close)):
            highest_high = np.max(high[i-period+1:i+1])
            lowest_low = np.min(low[i-period+1:i+1])
            
            if highest_high == lowest_low:
                wr_values[i] = -50.0
            else:
                wr_values[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100.0
        
        return wr_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3.0
        sma_tp = NumbaIndicators.sma(typical_price, period)
        
        cci_values = np.empty_like(close)
        cci_values[:period-1] = np.nan
        
        for i in range(period-1, len(close)):
            # Calculate mean deviation
            window = typical_price[i-period+1:i+1]
            mean_dev = np.mean(np.abs(window - sma_tp[i]))
            
            if mean_dev == 0:
                cci_values[i] = 0.0
            else:
                cci_values[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mean_dev)
        
        return cci_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3.0
        money_flow = typical_price * volume
        
        mfi_values = np.empty_like(close)
        mfi_values[:period] = np.nan
        
        for i in range(period, len(close)):
            positive_flow = 0.0
            negative_flow = 0.0
            
            for j in range(i-period+1, i+1):
                if j > 0:
                    if typical_price[j] > typical_price[j-1]:
                        positive_flow += money_flow[j]
                    elif typical_price[j] < typical_price[j-1]:
                        negative_flow += money_flow[j]
            
            if negative_flow == 0:
                mfi_values[i] = 100.0
            else:
                money_ratio = positive_flow / negative_flow
                mfi_values[i] = 100.0 - (100.0 / (1.0 + money_ratio))
        
        return mfi_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def parabolic_sar(high: np.ndarray, low: np.ndarray, close: np.ndarray, af_start: float = 0.02, af_max: float = 0.2) -> np.ndarray:
        """Parabolic SAR"""
        sar_values = np.empty_like(close)
        
        # Initialize
        trend = 1  # 1 for uptrend, -1 for downtrend
        af = af_start
        ep = high[0]  # Extreme point
        sar_values[0] = low[0]
        
        for i in range(1, len(close)):
            # Calculate SAR
            sar_values[i] = sar_values[i-1] + af * (ep - sar_values[i-1])
            
            # Check for trend reversal
            if trend == 1:  # Uptrend
                if low[i] <= sar_values[i]:
                    # Trend reversal to downtrend
                    trend = -1
                    sar_values[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    # Continue uptrend
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_start, af_max)
                    
                    # Adjust SAR
                    sar_values[i] = min(sar_values[i], low[i-1])
                    if i > 1:
                        sar_values[i] = min(sar_values[i], low[i-2])
            
            else:  # Downtrend
                if high[i] >= sar_values[i]:
                    # Trend reversal to uptrend
                    trend = 1
                    sar_values[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    # Continue downtrend
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_start, af_max)
                    
                    # Adjust SAR
                    sar_values[i] = max(sar_values[i], high[i-1])
                    if i > 1:
                        sar_values[i] = max(sar_values[i], high[i-2])
        
        return sar_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def calculate_all_indicators(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray
    ) -> Dict:
        """
        Calculate all technical indicators in one optimized pass
        
        This function is designed for maximum performance by calculating
        all indicators together to minimize array traversals.
        """
        # This will be implemented as a comprehensive calculation engine
        # that computes all indicators efficiently in a single pass
        pass

# Specialized indicator calculations for different timeframes
class TimeFrameOptimizedIndicators:
    """Indicators optimized for specific timeframes"""
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def intraday_indicators(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray):
        """Indicators optimized for intraday timeframes (1m, 5m, 15m)"""
        
        # Fast-moving indicators for intraday
        ema_9 = NumbaIndicators.ema(close, 9)
        ema_21 = NumbaIndicators.ema(close, 21)
        rsi_14 = NumbaIndicators.rsi(close, 14)
        atr_14 = NumbaIndicators.atr(high, low, close, 14)
        supertrend, st_signals = NumbaIndicators.supertrend(high, low, close, 7, 3.0)
        vwap = NumbaIndicators.vwap(high, low, close, volume)
        
        return {
            'ema_9': ema_9,
            'ema_21': ema_21,
            'rsi_14': rsi_14,
            'atr_14': atr_14,
            'supertrend_7_3': supertrend,
            'supertrend_signal_7_3': st_signals,
            'vwap': vwap
        }
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def daily_indicators(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray):
        """Indicators optimized for daily timeframe"""
        
        # Longer-term indicators for daily analysis
        ema_50 = NumbaIndicators.ema(close, 50)
        ema_200 = NumbaIndicators.ema(close, 200)
        sma_20 = NumbaIndicators.sma(close, 20)
        sma_50 = NumbaIndicators.sma(close, 50)
        rsi_14 = NumbaIndicators.rsi(close, 14)
        macd_line, macd_signal, macd_hist = NumbaIndicators.macd(close, 12, 26, 9)
        bb_upper, bb_middle, bb_lower = NumbaIndicators.bollinger_bands(close, 20, 2.0)
        atr_14 = NumbaIndicators.atr(high, low, close, 14)
        adx, di_plus, di_minus = NumbaIndicators.adx(high, low, close, 14)
        
        return {
            'ema_50': ema_50,
            'ema_200': ema_200,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi_14': rsi_14,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_hist,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'atr_14': atr_14,
            'adx_14': adx,
            'di_plus': di_plus,
            'di_minus': di_minus
        }

# Performance optimization utilities
class NumbaOptimizer:
    """Utilities for Numba performance optimization"""
    
    @staticmethod
    def warm_up_functions():
        """Warm up Numba functions for optimal performance"""
        
        logger.info("Warming up Numba-compiled functions...")
        
        # Create sample data
        sample_size = 1000
        sample_prices = np.random.random(sample_size) * 100 + 1000
        sample_high = sample_prices + np.random.random(sample_size) * 5
        sample_low = sample_prices - np.random.random(sample_size) * 5
        sample_volume = np.random.randint(1000, 100000, sample_size)
        
        # Warm up all functions
        try:
            NumbaIndicators.ema(sample_prices, 20)
            NumbaIndicators.sma(sample_prices, 20)
            NumbaIndicators.rsi(sample_prices, 14)
            NumbaIndicators.macd(sample_prices, 12, 26, 9)
            NumbaIndicators.atr(sample_high, sample_low, sample_prices, 14)
            NumbaIndicators.bollinger_bands(sample_prices, 20, 2.0)
            NumbaIndicators.supertrend(sample_high, sample_low, sample_prices, 10, 3.0)
            NumbaIndicators.stochastic(sample_high, sample_low, sample_prices, 14, 3)
            NumbaIndicators.vwap(sample_high, sample_low, sample_prices, sample_volume)
            NumbaIndicators.obv(sample_prices, sample_volume)
            NumbaIndicators.adx(sample_high, sample_low, sample_prices, 14)
            
            logger.info("✅ Numba functions warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Error warming up Numba functions: {e}")
    
    @staticmethod
    def get_numba_config() -> dict:
        """Get optimal Numba configuration"""
        return NUMBA_CONFIG.copy()
    
    @staticmethod
    def estimate_calculation_time(data_points: int, indicators_count: int) -> float:
        """Estimate calculation time for given data volume"""
        
        # Rough estimates based on benchmarks (seconds)
        base_time_per_indicator = 0.001  # 1ms per 1000 data points per indicator
        
        estimated_time = (data_points / 1000.0) * indicators_count * base_time_per_indicator
        
        return estimated_time

# Batch calculation utilities
@jit(**NUMBA_CONFIG)
def calculate_price_changes(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate price changes and percentage changes"""
    
    changes = np.empty_like(prices)
    pct_changes = np.empty_like(prices)
    
    changes[0] = 0.0
    pct_changes[0] = 0.0
    
    for i in range(1, len(prices)):
        changes[i] = prices[i] - prices[i-1]
        if prices[i-1] != 0:
            pct_changes[i] = (changes[i] / prices[i-1]) * 100.0
        else:
            pct_changes[i] = 0.0
    
    return changes, pct_changes

@jit(**NUMBA_CONFIG)
def calculate_high_low_metrics(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Calculate high-low percentage metrics"""
    
    hl_pct = np.empty_like(close)
    
    for i in range(len(close)):
        if close[i] != 0:
            hl_pct[i] = ((high[i] - low[i]) / close[i]) * 100.0
        else:
            hl_pct[i] = 0.0
    
    return hl_pct

@jit(**NUMBA_CONFIG)
def calculate_bid_ask_metrics(bid: np.ndarray, ask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate bid-ask spread metrics"""
    
    spread = ask - bid
    mid_price = (bid + ask) / 2.0
    spread_pct = np.where(mid_price != 0, (spread / mid_price) * 100.0, 0.0)
    
    return spread, mid_price, spread_pct
