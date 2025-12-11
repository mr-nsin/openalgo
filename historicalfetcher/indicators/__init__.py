"""Technical Indicators and Analytics Engine"""

# Handle missing numba gracefully
try:
    from .numba_indicators import NumbaIndicators
except ImportError:
    # Create a fallback NumbaIndicators class that uses numpy
    import numpy as np
    
    class NumbaIndicators:
        """Fallback NumbaIndicators using pure numpy when numba is not available"""
        
        @staticmethod
        def sma(data, period):
            """Simple Moving Average"""
            result = np.full(len(data), np.nan)
            for i in range(period - 1, len(data)):
                result[i] = np.mean(data[i - period + 1:i + 1])
            return result
        
        @staticmethod
        def ema(data, period):
            """Exponential Moving Average"""
            result = np.full(len(data), np.nan)
            alpha = 2 / (period + 1)
            result[period - 1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            return result
        
        @staticmethod
        def rsi(close, period=14):
            """Relative Strength Index"""
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            result = np.full(len(close), np.nan)
            avg_gain = np.mean(gain[:period])
            avg_loss = np.mean(loss[:period])
            
            if avg_loss == 0:
                result[period] = 100
            else:
                rs = avg_gain / avg_loss
                result[period] = 100 - (100 / (1 + rs))
            
            for i in range(period + 1, len(close)):
                avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
                avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
                if avg_loss == 0:
                    result[i] = 100
                else:
                    rs = avg_gain / avg_loss
                    result[i] = 100 - (100 / (1 + rs))
            
            return result
        
        @staticmethod
        def macd(close, fast_period=12, slow_period=26, signal_period=9):
            """MACD calculation"""
            ema_fast = NumbaIndicators.ema(close, fast_period)
            ema_slow = NumbaIndicators.ema(close, slow_period)
            macd_line = ema_fast - ema_slow
            signal = NumbaIndicators.ema(macd_line, signal_period)
            histogram = macd_line - signal
            return macd_line, signal, histogram
        
        @staticmethod
        def bollinger_bands(close, period=20, std_dev=2.0):
            """Bollinger Bands"""
            sma = NumbaIndicators.sma(close, period)
            std = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                std[i] = np.std(close[i - period + 1:i + 1])
            upper = sma + std_dev * std
            lower = sma - std_dev * std
            return upper, sma, lower
        
        @staticmethod
        def atr(high, low, close, period=14):
            """Average True Range"""
            tr = np.maximum(high - low, np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            ))
            tr[0] = high[0] - low[0]
            return NumbaIndicators.ema(tr, period)
        
        @staticmethod
        def stochastic(high, low, close, k_period=14, d_period=3):
            """Stochastic Oscillator"""
            k = np.full(len(close), np.nan)
            for i in range(k_period - 1, len(close)):
                highest = np.max(high[i - k_period + 1:i + 1])
                lowest = np.min(low[i - k_period + 1:i + 1])
                if highest - lowest != 0:
                    k[i] = 100 * (close[i] - lowest) / (highest - lowest)
                else:
                    k[i] = 50
            d = NumbaIndicators.sma(k, d_period)
            return k, d

try:
    from .options_greeks import OptionsGreeksCalculator
except ImportError:
    # Fallback is handled in indicator_engine.py
    OptionsGreeksCalculator = None

from .indicator_engine import IndicatorEngine

__all__ = ['NumbaIndicators', 'OptionsGreeksCalculator', 'IndicatorEngine']
