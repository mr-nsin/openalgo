"""
Numba-Optimized Technical Indicators

High-performance technical indicator calculations using Numba JIT compilation
for maximum speed in processing large datasets.
"""

import numpy as np
import numba
from numba import jit, prange
import math
from typing import Tuple, Optional, Dict
import sys
import os

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from historicalfetcher.utils.async_logger import get_async_logger

_async_logger = get_async_logger()
logger = _async_logger.get_logger()

# Logger is imported from loguru above

# Numba optimization settings for maximum performance
NUMBA_CONFIG = {
    'nopython': True,           # Pure machine code compilation
    'nogil': True,              # Release Python GIL
    'cache': True,              # Cache compiled functions
    'fastmath': True,           # Enable fast math optimizations
    'boundscheck': False,       # Disable bounds checking for speed
    'parallel': False           # Disable parallel by default (enable per function)
}

# Configuration for parallel processing (for large datasets)
NUMBA_PARALLEL_CONFIG = {
    'nopython': True,
    'nogil': True,
    'cache': True,
    'fastmath': True,
    'boundscheck': False,
    'parallel': True            # Enable parallel processing
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
        # Calculate EMAs inline to avoid class method calls in Numba
        # EMA Fast
        alpha_fast = 2.0 / (fast + 1.0)
        ema_fast = np.empty_like(prices)
        ema_fast[0] = prices[0]
        for i in range(1, len(prices)):
            ema_fast[i] = alpha_fast * prices[i] + (1.0 - alpha_fast) * ema_fast[i-1]
        
        # EMA Slow
        alpha_slow = 2.0 / (slow + 1.0)
        ema_slow = np.empty_like(prices)
        ema_slow[0] = prices[0]
        for i in range(1, len(prices)):
            ema_slow[i] = alpha_slow * prices[i] + (1.0 - alpha_slow) * ema_slow[i-1]
        
        # MACD Line
        macd_line = ema_fast - ema_slow
        
        # Signal Line (EMA of MACD)
        alpha_signal = 2.0 / (signal + 1.0)
        signal_line = np.empty_like(macd_line)
        signal_line[0] = macd_line[0]
        for i in range(1, len(macd_line)):
            signal_line[i] = alpha_signal * macd_line[i] + (1.0 - alpha_signal) * signal_line[i-1]
        
        # Histogram
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
        # Calculate SMA inline
        middle_band = np.empty_like(prices)
        middle_band[:period-1] = np.nan
        
        for i in range(period-1, len(prices)):
            middle_band[i] = np.mean(prices[i-period+1:i+1])
        
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
        # Calculate ATR inline to avoid class method calls in Numba
        tr = np.empty_like(close)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Calculate ATR using EMA
        alpha = 2.0 / (period + 1.0)
        atr_values = np.empty_like(tr)
        atr_values[0] = tr[0]
        
        for i in range(1, len(tr)):
            atr_values[i] = alpha * tr[i] + (1.0 - alpha) * atr_values[i-1]
        
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
        
        # Calculate %D (SMA of %K) inline
        d_values = np.empty_like(k_values)
        d_values[:d_period-1] = np.nan
        
        for i in range(d_period-1, len(k_values)):
            d_values[i] = np.mean(k_values[i-d_period+1:i+1])
        
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
        
        # Smooth the values using inline EMA
        alpha = 2.0 / (period + 1.0)
        
        # ATR smoothed
        atr_smooth = np.empty_like(tr)
        atr_smooth[0] = tr[0]
        for i in range(1, len(tr)):
            atr_smooth[i] = alpha * tr[i] + (1.0 - alpha) * atr_smooth[i-1]
        
        # DM+ smoothed
        dm_plus_smooth = np.empty_like(dm_plus)
        dm_plus_smooth[0] = dm_plus[0]
        for i in range(1, len(dm_plus)):
            dm_plus_smooth[i] = alpha * dm_plus[i] + (1.0 - alpha) * dm_plus_smooth[i-1]
        
        # DM- smoothed
        dm_minus_smooth = np.empty_like(dm_minus)
        dm_minus_smooth[0] = dm_minus[0]
        for i in range(1, len(dm_minus)):
            dm_minus_smooth[i] = alpha * dm_minus[i] + (1.0 - alpha) * dm_minus_smooth[i-1]
        
        # Calculate DI+ and DI-
        di_plus = 100.0 * dm_plus_smooth / atr_smooth
        di_minus = 100.0 * dm_minus_smooth / atr_smooth
        
        # Calculate ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100.0
        
        # ADX smoothed
        adx = np.empty_like(dx)
        adx[0] = dx[0]
        for i in range(1, len(dx)):
            adx[i] = alpha * dx[i] + (1.0 - alpha) * adx[i-1]
        
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
        
        # Calculate SMA of typical price inline
        sma_tp = np.empty_like(typical_price)
        sma_tp[:period-1] = np.nan
        
        for i in range(period-1, len(typical_price)):
            sma_tp[i] = np.mean(typical_price[i-period+1:i+1])
        
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
    def ichimoku(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Ichimoku Cloud Indicator
        
        Returns:
            Tuple of (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span, cloud_top, cloud_bottom, cloud_color)
            cloud_color: 1=Green (bullish), 0=Red (bearish)
        """
        n = len(close)
        tenkan_sen = np.empty_like(close)
        kijun_sen = np.empty_like(close)
        senkou_span_a = np.empty_like(close)
        senkou_span_b = np.empty_like(close)
        chikou_span = np.empty_like(close)
        cloud_top = np.empty_like(close)
        cloud_bottom = np.empty_like(close)
        cloud_color = np.empty_like(close, dtype=np.int8)
        
        # Tenkan-sen (Conversion Line) - 9-period
        period_tenkan = 9
        for i in range(period_tenkan - 1, n):
            high_window = np.max(high[i-period_tenkan+1:i+1])
            low_window = np.min(low[i-period_tenkan+1:i+1])
            tenkan_sen[i] = (high_window + low_window) / 2.0
        tenkan_sen[:period_tenkan-1] = np.nan
        
        # Kijun-sen (Base Line) - 26-period
        period_kijun = 26
        for i in range(period_kijun - 1, n):
            high_window = np.max(high[i-period_kijun+1:i+1])
            low_window = np.min(low[i-period_kijun+1:i+1])
            kijun_sen[i] = (high_window + low_window) / 2.0
        kijun_sen[:period_kijun-1] = np.nan
        
        # Senkou Span A (Leading Span A) - shifted forward by 26 periods
        for i in range(n):
            if i >= period_kijun - 1:
                senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2.0
            else:
                senkou_span_a[i] = np.nan
        
        # Senkou Span B (Leading Span B) - 52-period, shifted forward by 26 periods
        period_senkou_b = 52
        senkou_span_b_temp = np.empty_like(close)
        for i in range(period_senkou_b - 1, n):
            high_window = np.max(high[i-period_senkou_b+1:i+1])
            low_window = np.min(low[i-period_senkou_b+1:i+1])
            senkou_span_b_temp[i] = (high_window + low_window) / 2.0
        senkou_span_b_temp[:period_senkou_b-1] = np.nan
        
        # Shift Senkou Span B forward by 26 periods
        for i in range(n):
            if i >= period_kijun - 1 and i < n - period_kijun + 1:
                senkou_span_b[i] = senkou_span_b_temp[i - period_kijun + 1] if i - period_kijun + 1 >= period_senkou_b - 1 else np.nan
            else:
                senkou_span_b[i] = np.nan
        
        # Chikou Span (Lagging Span) - close shifted backward by 26 periods
        for i in range(n):
            if i < n - period_kijun + 1:
                chikou_span[i] = close[i + period_kijun - 1]
            else:
                chikou_span[i] = np.nan
        
        # Cloud top and bottom
        for i in range(n):
            if not (np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i])):
                cloud_top[i] = max(senkou_span_a[i], senkou_span_b[i])
                cloud_bottom[i] = min(senkou_span_a[i], senkou_span_b[i])
                cloud_color[i] = 1 if senkou_span_a[i] > senkou_span_b[i] else 0
            else:
                cloud_top[i] = np.nan
                cloud_bottom[i] = np.nan
                cloud_color[i] = 0
        
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span, cloud_top, cloud_bottom, cloud_color
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def aroon(high: np.ndarray, low: np.ndarray, period: int = 25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aroon Indicator
        
        Returns:
            Tuple of (aroon_up, aroon_down, aroon_oscillator)
        """
        n = len(high)
        aroon_up = np.empty_like(high)
        aroon_down = np.empty_like(high)
        aroon_oscillator = np.empty_like(high)
        
        aroon_up[:period-1] = np.nan
        aroon_down[:period-1] = np.nan
        aroon_oscillator[:period-1] = np.nan
        
        for i in range(period - 1, n):
            # Find highest high and lowest low in period
            highest_high_idx = i
            lowest_low_idx = i
            
            for j in range(i - period + 1, i + 1):
                if high[j] > high[highest_high_idx]:
                    highest_high_idx = j
                if low[j] < low[lowest_low_idx]:
                    lowest_low_idx = j
            
            # Calculate Aroon Up
            periods_since_high = i - highest_high_idx
            aroon_up[i] = ((period - periods_since_high) / period) * 100.0
            
            # Calculate Aroon Down
            periods_since_low = i - lowest_low_idx
            aroon_down[i] = ((period - periods_since_low) / period) * 100.0
            
            # Aroon Oscillator
            aroon_oscillator[i] = aroon_up[i] - aroon_down[i]
        
        return aroon_up, aroon_down, aroon_oscillator
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def volume_profile(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, num_bins: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Volume Profile - Point of Control (POC), Value Area High (VAH), Value Area Low (VAL)
        
        Returns:
            Tuple of (poc, vah, val, volume_profile_balance)
        """
        n = len(close)
        poc = np.empty_like(close)
        vah = np.empty_like(close)
        val = np.empty_like(close)
        volume_profile_balance = np.empty_like(close)
        
        # Use a rolling window approach
        window = min(100, n)  # Use last 100 periods or all if less
        
        for i in range(n):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < 10:  # Need at least 10 periods
                poc[i] = close[i]
                vah[i] = close[i]
                val[i] = close[i]
                volume_profile_balance[i] = close[i]
                continue
            
            # Get price range for this window
            price_min = np.min(low[start_idx:end_idx])
            price_max = np.max(high[start_idx:end_idx])
            price_range = price_max - price_min
            
            if price_range == 0:
                poc[i] = close[i]
                vah[i] = close[i]
                val[i] = close[i]
                volume_profile_balance[i] = close[i]
                continue
            
            # Create bins
            bin_size = price_range / num_bins
            bin_volumes = np.zeros(num_bins)
            bin_prices = np.zeros(num_bins)
            
            for j in range(start_idx, end_idx):
                # Determine which bin this candle belongs to
                typical_price = (high[j] + low[j] + close[j]) / 3.0
                bin_idx = min(int((typical_price - price_min) / bin_size), num_bins - 1)
                bin_volumes[bin_idx] += volume[j]
                bin_prices[bin_idx] = price_min + (bin_idx + 0.5) * bin_size
            
            # Find POC (bin with highest volume)
            poc_bin = np.argmax(bin_volumes)
            poc[i] = bin_prices[poc_bin] if bin_volumes[poc_bin] > 0 else close[i]
            
            # Calculate Value Area (70% of volume)
            total_volume = np.sum(bin_volumes)
            if total_volume == 0:
                vah[i] = close[i]
                val[i] = close[i]
            else:
                target_volume = total_volume * 0.7
                cumulative_volume = 0.0
                vah_idx = poc_bin
                val_idx = poc_bin
                
                # Expand from POC to find 70% volume area
                left_idx = poc_bin
                right_idx = poc_bin
                
                while cumulative_volume < target_volume and (left_idx > 0 or right_idx < num_bins - 1):
                    left_vol = bin_volumes[left_idx - 1] if left_idx > 0 else 0.0
                    right_vol = bin_volumes[right_idx + 1] if right_idx < num_bins - 1 else 0.0
                    
                    if left_vol > right_vol and left_idx > 0:
                        cumulative_volume += left_vol
                        left_idx -= 1
                        val_idx = left_idx
                    elif right_idx < num_bins - 1:
                        cumulative_volume += right_vol
                        right_idx += 1
                        vah_idx = right_idx
                    else:
                        break
                
                vah[i] = bin_prices[vah_idx] if vah_idx < num_bins else close[i]
                val[i] = bin_prices[val_idx] if val_idx >= 0 else close[i]
            
            # Volume-weighted balance point
            total_pv = np.sum(bin_prices * bin_volumes)
            volume_profile_balance[i] = total_pv / total_volume if total_volume > 0 else close[i]
        
        return poc, vah, val, volume_profile_balance
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def ad_line(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Accumulation/Distribution Line
        
        Returns:
            Tuple of (ad_line, ad_line_slope, ad_line_signal)
            signal: 1=Accumulation, -1=Distribution, 0=Neutral
        """
        n = len(close)
        ad_line = np.empty_like(close)
        ad_line_slope = np.empty_like(close)
        ad_line_signal = np.empty_like(close, dtype=np.int8)
        
        # Money Flow Multiplier
        mfm = np.empty_like(close)
        for i in range(n):
            high_low_range = high[i] - low[i]
            if high_low_range == 0:
                mfm[i] = 0.0
            else:
                mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / high_low_range
        
        # Money Flow Volume
        mfv = mfm * volume.astype(np.float64)
        
        # A/D Line (cumulative)
        ad_line[0] = mfv[0]
        for i in range(1, n):
            ad_line[i] = ad_line[i-1] + mfv[i]
        
        # Calculate slope (change over last 5 periods)
        period_slope = 5
        ad_line_slope[0] = 0.0
        for i in range(1, n):
            if i >= period_slope:
                slope = (ad_line[i] - ad_line[i - period_slope]) / period_slope
                ad_line_slope[i] = slope
            else:
                ad_line_slope[i] = ad_line[i] - ad_line[0]
        
        # Signal: 1=Accumulation (slope > 0), -1=Distribution (slope < 0), 0=Neutral
        for i in range(n):
            if ad_line_slope[i] > 0.01:  # Small threshold to avoid noise
                ad_line_signal[i] = 1
            elif ad_line_slope[i] < -0.01:
                ad_line_signal[i] = -1
            else:
                ad_line_signal[i] = 0
        
        return ad_line, ad_line_slope, ad_line_signal
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def cmf(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chaikin Money Flow (CMF)
        
        Returns:
            Tuple of (cmf, cmf_signal)
            signal: 1=Buy, -1=Sell, 0=Neutral
        """
        n = len(close)
        cmf = np.empty_like(close)
        cmf_signal = np.empty_like(close, dtype=np.int8)
        
        cmf[:period-1] = np.nan
        cmf_signal[:period-1] = 0
        
        # Money Flow Multiplier
        mfm = np.empty_like(close)
        for i in range(n):
            high_low_range = high[i] - low[i]
            if high_low_range == 0:
                mfm[i] = 0.0
            else:
                mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / high_low_range
        
        # Money Flow Volume
        mfv = mfm * volume.astype(np.float64)
        
        # Calculate CMF
        for i in range(period - 1, n):
            sum_mfv = np.sum(mfv[i-period+1:i+1])
            sum_volume = np.sum(volume[i-period+1:i+1])
            
            if sum_volume == 0:
                cmf[i] = 0.0
            else:
                cmf[i] = sum_mfv / sum_volume
            
            # Signal: 1=Buy (CMF > 0.1), -1=Sell (CMF < -0.1), 0=Neutral
            if cmf[i] > 0.1:
                cmf_signal[i] = 1
            elif cmf[i] < -0.1:
                cmf_signal[i] = -1
            else:
                cmf_signal[i] = 0
        
        return cmf, cmf_signal
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def twap(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Time Weighted Average Price (TWAP)
        
        Uses equal time weighting per period (simplified TWAP).
        For true time-weighted calculation, actual time differences between periods would be needed.
        
        Args:
            high, low, close: Price arrays
        
        Returns:
            TWAP values
        """
        n = len(close)
        twap_values = np.empty_like(close)
        
        # Calculate typical price
        typical_price = (high + low + close) / 3.0
        
        # Simplified TWAP: equal time weighting per period
        # This is equivalent to a cumulative average of typical price
        cumulative_price = 0.0
        
        for i in range(n):
            cumulative_price += typical_price[i]
            twap_values[i] = cumulative_price / (i + 1.0)
        
        return twap_values
    
    @staticmethod
    @jit(**NUMBA_CONFIG)
    def volume_divergence(close: np.ndarray, volume: np.ndarray, rsi: np.ndarray, macd: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Volume Divergence Indicators
        
        Returns:
            Tuple of (volume_price_divergence, volume_divergence_strength, volume_divergence_confirmed,
                     rsi_volume_divergence, macd_volume_divergence, price_volume_divergence_type)
            volume_price_divergence: 1=Bullish, 2=Bearish, 0=No divergence
            rsi_volume_divergence: 1=Bullish, 2=Bearish, 0=No divergence
            macd_volume_divergence: 1=Bullish, 2=Bearish, 0=No divergence
        """
        n = len(close)
        volume_price_divergence = np.zeros(n, dtype=np.int8)
        volume_divergence_strength = np.zeros(n)
        volume_divergence_confirmed = np.zeros(n, dtype=np.int8)
        rsi_volume_divergence = np.zeros(n, dtype=np.int8)
        macd_volume_divergence = np.zeros(n, dtype=np.int8)
        price_volume_divergence_type = np.zeros(n, dtype=np.int8)
        
        # Calculate volume moving average
        volume_ma = np.empty_like(volume, dtype=np.float64)
        for i in range(period - 1, n):
            volume_ma[i] = np.mean(volume[i-period+1:i+1])
        volume_ma[:period-1] = volume[:period-1].astype(np.float64)
        
        for i in range(period, n):
            # Price-Volume Divergence
            price_change = close[i] - close[i-period]
            volume_change = volume[i] - volume_ma[i]
            prev_volume_change = volume[i-1] - volume_ma[i-1] if i > period else 0
            
            # Bullish divergence: price down, volume up
            if price_change < 0 and volume_change > 0 and volume[i] > volume_ma[i] * 1.2:
                volume_price_divergence[i] = 1
                volume_divergence_strength[i] = min(100.0, abs(volume_change / volume_ma[i]) * 100.0)
                price_volume_divergence_type[i] = 1
            # Bearish divergence: price up, volume down
            elif price_change > 0 and volume_change < 0 and volume[i] < volume_ma[i] * 0.8:
                volume_price_divergence[i] = 2
                volume_divergence_strength[i] = min(100.0, abs(volume_change / volume_ma[i]) * 100.0)
                price_volume_divergence_type[i] = 2
            
            # RSI-Volume Divergence
            if not np.isnan(rsi[i]) and not np.isnan(rsi[i-period]):
                rsi_change = rsi[i] - rsi[i-period]
                if rsi_change < -10 and volume_change > 0:  # RSI oversold, volume increasing
                    rsi_volume_divergence[i] = 1
                elif rsi_change > 10 and volume_change < 0:  # RSI overbought, volume decreasing
                    rsi_volume_divergence[i] = 2
            
            # MACD-Volume Divergence
            if not np.isnan(macd[i]) and not np.isnan(macd[i-period]):
                macd_change = macd[i] - macd[i-period]
                if macd_change < 0 and volume_change > 0:  # MACD down, volume up
                    macd_volume_divergence[i] = 1
                elif macd_change > 0 and volume_change < 0:  # MACD up, volume down
                    macd_volume_divergence[i] = 2
            
            # Confirmation: divergence persists for multiple periods
            if i >= period + 2:
                if volume_price_divergence[i] == volume_price_divergence[i-1] == volume_price_divergence[i-2]:
                    volume_divergence_confirmed[i] = 1
        
        return volume_price_divergence, volume_divergence_strength, volume_divergence_confirmed, rsi_volume_divergence, macd_volume_divergence, price_volume_divergence_type
    
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
