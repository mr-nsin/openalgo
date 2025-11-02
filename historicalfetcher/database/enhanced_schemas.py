"""
Enhanced QuestDB Schemas with Technical Indicators and Analytics

Comprehensive schemas that include pre-calculated technical indicators,
options Greeks, bid/ask data, and advanced analytics columns.
"""

from typing import Dict, List, Optional
from enum import IntEnum
import sys
import os

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from historicalfetcher.utils.async_logger import get_async_logger

_async_logger = get_async_logger()
logger = _async_logger.get_logger()

# Logger is imported from loguru above

class IndicatorType(IntEnum):
    """Numeric codes for different indicator types"""
    # Trend Indicators
    EMA_9 = 1
    EMA_21 = 2
    EMA_50 = 3
    EMA_200 = 4
    SMA_20 = 5
    SMA_50 = 6
    
    # Momentum Indicators
    RSI_14 = 10
    MACD_SIGNAL = 11
    MACD_HISTOGRAM = 12
    STOCHASTIC_K = 13
    STOCHASTIC_D = 14
    
    # Volatility Indicators
    ATR_14 = 20
    BOLLINGER_UPPER = 21
    BOLLINGER_LOWER = 22
    BOLLINGER_WIDTH = 23
    
    # Trend Following
    SUPERTREND_7_3 = 30
    SUPERTREND_10_3 = 31
    PARABOLIC_SAR = 32
    
    # Volume Indicators
    VOLUME_SMA_20 = 40
    VOLUME_WEIGHTED_PRICE = 41
    ON_BALANCE_VOLUME = 42

class GreeksType(IntEnum):
    """Numeric codes for options Greeks"""
    DELTA = 1
    GAMMA = 2
    THETA = 3
    VEGA = 4
    RHO = 5
    IMPLIED_VOLATILITY = 6
    INTRINSIC_VALUE = 7
    TIME_VALUE = 8
    MONEYNESS = 9

class EnhancedTableSchemas:
    """Enhanced table schemas with comprehensive analytics"""
    
    @staticmethod
    def get_enhanced_equity_schema(table_name: str) -> str:
        """Enhanced equity schema with technical indicators"""
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            -- Basic OHLCV
            tf BYTE,                    -- Timeframe code
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            
            -- Price-based Technical Indicators
            ema_9 DOUBLE,               -- 9-period EMA
            ema_21 DOUBLE,              -- 21-period EMA  
            ema_50 DOUBLE,              -- 50-period EMA
            ema_200 DOUBLE,             -- 200-period EMA
            sma_20 DOUBLE,              -- 20-period SMA
            sma_50 DOUBLE,              -- 50-period SMA
            
            -- Momentum Indicators
            rsi_14 DOUBLE,              -- 14-period RSI
            macd_line DOUBLE,           -- MACD line
            macd_signal DOUBLE,         -- MACD signal line
            macd_histogram DOUBLE,      -- MACD histogram
            stoch_k DOUBLE,             -- Stochastic %K
            stoch_d DOUBLE,             -- Stochastic %D
            
            -- Volatility Indicators
            atr_14 DOUBLE,              -- 14-period ATR
            bb_upper DOUBLE,            -- Bollinger Band Upper
            bb_middle DOUBLE,           -- Bollinger Band Middle (SMA 20)
            bb_lower DOUBLE,            -- Bollinger Band Lower
            bb_width DOUBLE,            -- Bollinger Band Width
            bb_percent DOUBLE,          -- Bollinger Band %B
            
            -- Trend Following Indicators
            supertrend_7_3 DOUBLE,      -- Supertrend (7,3)
            supertrend_signal_7_3 BYTE, -- 1=Buy, 0=Sell
            supertrend_10_3 DOUBLE,     -- Supertrend (10,3)
            supertrend_signal_10_3 BYTE, -- 1=Buy, 0=Sell
            parabolic_sar DOUBLE,       -- Parabolic SAR
            
            -- Volume Indicators
            volume_sma_20 DOUBLE,       -- 20-period Volume SMA
            vwap DOUBLE,                -- Volume Weighted Average Price
            obv LONG,                   -- On Balance Volume
            
            -- Price Action Indicators
            pivot_point DOUBLE,         -- Daily Pivot Point
            resistance_1 DOUBLE,        -- R1
            resistance_2 DOUBLE,        -- R2
            resistance_3 DOUBLE,        -- R3
            support_1 DOUBLE,           -- S1
            support_2 DOUBLE,           -- S2
            support_3 DOUBLE,           -- S3
            
            -- Market Microstructure (Live Data)
            bid_1 DOUBLE,               -- Best Bid
            bid_qty_1 LONG,             -- Best Bid Quantity
            bid_2 DOUBLE,
            bid_qty_2 LONG,
            bid_3 DOUBLE,
            bid_qty_3 LONG,
            bid_4 DOUBLE,
            bid_qty_4 LONG,
            bid_5 DOUBLE,
            bid_qty_5 LONG,
            
            ask_1 DOUBLE,               -- Best Ask
            ask_qty_1 LONG,             -- Best Ask Quantity
            ask_2 DOUBLE,
            ask_qty_2 LONG,
            ask_3 DOUBLE,
            ask_qty_3 LONG,
            ask_4 DOUBLE,
            ask_qty_4 LONG,
            ask_5 DOUBLE,
            ask_qty_5 LONG,
            
            -- Derived Market Data
            bid_ask_spread DOUBLE,      -- Ask1 - Bid1
            bid_ask_spread_pct DOUBLE,  -- Spread as % of mid price
            mid_price DOUBLE,           -- (Bid1 + Ask1) / 2
            total_bid_qty LONG,         -- Sum of all bid quantities
            total_ask_qty LONG,         -- Sum of all ask quantities
            
            -- Additional Analytics
            price_change DOUBLE,        -- Change from previous close
            price_change_pct DOUBLE,    -- % change from previous close
            high_low_pct DOUBLE,        -- (High - Low) / Close * 100
            
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        """
    
    @staticmethod
    def get_enhanced_futures_schema(table_name: str) -> str:
        """Enhanced futures schema with F&O specific indicators"""
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            -- Contract Information
            contract_token SYMBOL CAPACITY 1000 CACHE,
            expiry_date DATE,
            tf BYTE,
            
            -- Basic OHLCV + OI
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            oi LONG,                    -- Open Interest
            
            -- Technical Indicators (same as equity)
            ema_9 DOUBLE, ema_21 DOUBLE, ema_50 DOUBLE, ema_200 DOUBLE,
            sma_20 DOUBLE, sma_50 DOUBLE,
            rsi_14 DOUBLE,
            macd_line DOUBLE, macd_signal DOUBLE, macd_histogram DOUBLE,
            stoch_k DOUBLE, stoch_d DOUBLE,
            atr_14 DOUBLE,
            bb_upper DOUBLE, bb_middle DOUBLE, bb_lower DOUBLE, bb_width DOUBLE,
            supertrend_7_3 DOUBLE, supertrend_signal_7_3 BYTE,
            supertrend_10_3 DOUBLE, supertrend_signal_10_3 BYTE,
            vwap DOUBLE, obv LONG,
            
            -- Futures-Specific Indicators
            oi_change LONG,             -- Change in Open Interest
            oi_change_pct DOUBLE,       -- % change in OI
            volume_oi_ratio DOUBLE,     -- Volume / OI ratio
            price_oi_correlation DOUBLE, -- Price-OI correlation
            
            -- Basis and Spread Analysis
            spot_price DOUBLE,          -- Underlying spot price
            basis DOUBLE,               -- Futures - Spot
            basis_pct DOUBLE,           -- Basis as % of spot
            cost_of_carry DOUBLE,       -- Implied cost of carry
            
            -- Market Microstructure
            bid_1 DOUBLE, bid_qty_1 LONG, bid_2 DOUBLE, bid_qty_2 LONG,
            bid_3 DOUBLE, bid_qty_3 LONG, bid_4 DOUBLE, bid_qty_4 LONG,
            bid_5 DOUBLE, bid_qty_5 LONG,
            ask_1 DOUBLE, ask_qty_1 LONG, ask_2 DOUBLE, ask_qty_2 LONG,
            ask_3 DOUBLE, ask_qty_3 LONG, ask_4 DOUBLE, ask_qty_4 LONG,
            ask_5 DOUBLE, ask_qty_5 LONG,
            
            bid_ask_spread DOUBLE, bid_ask_spread_pct DOUBLE,
            mid_price DOUBLE, total_bid_qty LONG, total_ask_qty LONG,
            
            -- Additional Analytics
            price_change DOUBLE, price_change_pct DOUBLE,
            volume_change LONG, volume_change_pct DOUBLE,
            
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        """
    
    @staticmethod
    def get_enhanced_options_schema(table_name: str) -> str:
        """Enhanced options schema with comprehensive Greeks and analytics"""
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            -- Contract Information
            contract_token SYMBOL CAPACITY 2000 CACHE,
            option_type BYTE,           -- 1=CE, 2=PE
            strike INT,                 -- Strike * 100
            tf BYTE,
            
            -- Basic OHLCV + OI
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            oi LONG,
            
            -- Options Greeks (Primary)
            delta DOUBLE,               -- Price sensitivity
            gamma DOUBLE,               -- Delta sensitivity
            theta DOUBLE,               -- Time decay
            vega DOUBLE,                -- Volatility sensitivity
            rho DOUBLE,                 -- Interest rate sensitivity
            
            -- Volatility Metrics
            implied_volatility DOUBLE,  -- IV
            historical_volatility DOUBLE, -- HV (20-day)
            iv_rank DOUBLE,             -- IV Rank (0-100)
            iv_percentile DOUBLE,       -- IV Percentile
            
            -- Value Components
            intrinsic_value DOUBLE,     -- Max(0, S-K) for calls, Max(0, K-S) for puts
            time_value DOUBLE,          -- Premium - Intrinsic Value
            moneyness DOUBLE,           -- S/K for calls, K/S for puts
            
            -- Advanced Greeks
            lambda_greek DOUBLE,        -- Leverage (Delta * S / Premium)
            epsilon DOUBLE,             -- Dividend sensitivity
            vera DOUBLE,                -- Volatility elasticity
            
            -- Risk Metrics
            probability_itm DOUBLE,     -- Probability of finishing ITM
            probability_profit DOUBLE,  -- Probability of profit
            max_pain DOUBLE,            -- Max pain level
            
            -- Technical Indicators (for options premium)
            rsi_14 DOUBLE,
            ema_9 DOUBLE, ema_21 DOUBLE,
            atr_14 DOUBLE,
            bb_upper DOUBLE, bb_lower DOUBLE,
            
            -- Market Microstructure (5 levels)
            bid_1 DOUBLE, bid_qty_1 LONG,
            bid_2 DOUBLE, bid_qty_2 LONG,
            bid_3 DOUBLE, bid_qty_3 LONG,
            bid_4 DOUBLE, bid_qty_4 LONG,
            bid_5 DOUBLE, bid_qty_5 LONG,
            
            ask_1 DOUBLE, ask_qty_1 LONG,
            ask_2 DOUBLE, ask_qty_2 LONG,
            ask_3 DOUBLE, ask_qty_3 LONG,
            ask_4 DOUBLE, ask_qty_4 LONG,
            ask_5 DOUBLE, ask_qty_5 LONG,
            
            -- Derived Market Data
            bid_ask_spread DOUBLE,
            bid_ask_spread_pct DOUBLE,
            mid_price DOUBLE,
            total_bid_qty LONG,
            total_ask_qty LONG,
            
            -- Options-Specific Market Data
            put_call_ratio DOUBLE,      -- Put/Call volume ratio
            max_pain_distance DOUBLE,   -- Distance from max pain
            skew DOUBLE,                -- Volatility skew
            
            -- Change Metrics
            price_change DOUBLE,
            price_change_pct DOUBLE,
            oi_change LONG,
            oi_change_pct DOUBLE,
            iv_change DOUBLE,
            delta_change DOUBLE,
            
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        """
    
    @staticmethod
    def get_enhanced_index_schema(table_name: str) -> str:
        """Enhanced index schema with comprehensive indicators"""
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            tf BYTE,
            
            -- Basic OHLC (no volume for indices)
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            
            -- Technical Indicators
            ema_9 DOUBLE, ema_21 DOUBLE, ema_50 DOUBLE, ema_200 DOUBLE,
            sma_20 DOUBLE, sma_50 DOUBLE, sma_100 DOUBLE,
            
            -- Momentum Indicators
            rsi_14 DOUBLE,
            macd_line DOUBLE, macd_signal DOUBLE, macd_histogram DOUBLE,
            stoch_k DOUBLE, stoch_d DOUBLE,
            
            -- Volatility Indicators
            atr_14 DOUBLE,
            bb_upper DOUBLE, bb_middle DOUBLE, bb_lower DOUBLE,
            bb_width DOUBLE, bb_percent DOUBLE,
            
            -- Trend Indicators
            supertrend_7_3 DOUBLE, supertrend_signal_7_3 BYTE,
            supertrend_10_3 DOUBLE, supertrend_signal_10_3 BYTE,
            parabolic_sar DOUBLE,
            
            -- Index-Specific Indicators
            advance_decline_ratio DOUBLE, -- A/D ratio
            high_low_index DOUBLE,      -- New highs vs new lows
            mcclellan_oscillator DOUBLE, -- Market breadth
            
            -- Support/Resistance Levels
            pivot_point DOUBLE,
            resistance_1 DOUBLE, resistance_2 DOUBLE, resistance_3 DOUBLE,
            support_1 DOUBLE, support_2 DOUBLE, support_3 DOUBLE,
            
            -- Volatility Measures
            realized_volatility DOUBLE, -- 20-day realized volatility
            garch_volatility DOUBLE,    -- GARCH volatility forecast
            
            -- Change Metrics
            price_change DOUBLE,
            price_change_pct DOUBLE,
            high_low_pct DOUBLE,
            
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        """
    
    @staticmethod
    def get_options_analytics_schema(table_name: str) -> str:
        """Specialized options analytics table for chain-level metrics"""
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name}_analytics (
            tf BYTE,
            
            -- Chain-Level Metrics
            total_call_volume LONG,
            total_put_volume LONG,
            put_call_ratio DOUBLE,
            
            total_call_oi LONG,
            total_put_oi LONG,
            put_call_oi_ratio DOUBLE,
            
            -- Volatility Surface
            atm_iv DOUBLE,              -- ATM Implied Volatility
            iv_skew DOUBLE,             -- Volatility skew
            iv_term_structure DOUBLE,   -- Term structure slope
            
            -- Max Pain Analysis
            max_pain_strike INT,        -- Strike * 100
            max_pain_value DOUBLE,
            
            -- Greeks Aggregation
            total_delta DOUBLE,         -- Net delta of all positions
            total_gamma DOUBLE,         -- Net gamma
            total_theta DOUBLE,         -- Net theta
            total_vega DOUBLE,          -- Net vega
            
            -- Support/Resistance from Options
            call_resistance_1 INT,      -- Strike with max call OI
            call_resistance_2 INT,
            put_support_1 INT,          -- Strike with max put OI
            put_support_2 INT,
            
            -- Sentiment Indicators
            fear_greed_index DOUBLE,    -- Options-based fear/greed
            volatility_risk_premium DOUBLE, -- IV - HV
            
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        """

class IndicatorColumnDefinitions:
    """Comprehensive column definitions for all indicators"""
    
    # Technical Indicators Columns
    TREND_INDICATORS = {
        'ema_9': 'DOUBLE',          # 9-period Exponential Moving Average
        'ema_21': 'DOUBLE',         # 21-period EMA
        'ema_50': 'DOUBLE',         # 50-period EMA
        'ema_200': 'DOUBLE',        # 200-period EMA
        'sma_20': 'DOUBLE',         # 20-period Simple Moving Average
        'sma_50': 'DOUBLE',         # 50-period SMA
        'sma_100': 'DOUBLE',        # 100-period SMA
        'wma_20': 'DOUBLE',         # 20-period Weighted Moving Average
        'hull_ma_20': 'DOUBLE',     # Hull Moving Average
        'tema_21': 'DOUBLE',        # Triple EMA
    }
    
    MOMENTUM_INDICATORS = {
        'rsi_14': 'DOUBLE',         # 14-period RSI
        'rsi_21': 'DOUBLE',         # 21-period RSI
        'macd_line': 'DOUBLE',      # MACD Line (12,26)
        'macd_signal': 'DOUBLE',    # MACD Signal (9)
        'macd_histogram': 'DOUBLE', # MACD Histogram
        'stoch_k': 'DOUBLE',        # Stochastic %K (14,3,3)
        'stoch_d': 'DOUBLE',        # Stochastic %D
        'williams_r': 'DOUBLE',     # Williams %R
        'roc_10': 'DOUBLE',         # 10-period Rate of Change
        'momentum_10': 'DOUBLE',    # 10-period Momentum
        'cci_20': 'DOUBLE',         # 20-period Commodity Channel Index
    }
    
    VOLATILITY_INDICATORS = {
        'atr_14': 'DOUBLE',         # 14-period Average True Range
        'atr_21': 'DOUBLE',         # 21-period ATR
        'bb_upper': 'DOUBLE',       # Bollinger Band Upper (20,2)
        'bb_middle': 'DOUBLE',      # Bollinger Band Middle
        'bb_lower': 'DOUBLE',       # Bollinger Band Lower
        'bb_width': 'DOUBLE',       # Bollinger Band Width
        'bb_percent': 'DOUBLE',     # Bollinger %B
        'keltner_upper': 'DOUBLE',  # Keltner Channel Upper
        'keltner_lower': 'DOUBLE',  # Keltner Channel Lower
        'donchian_upper': 'DOUBLE', # Donchian Channel Upper (20)
        'donchian_lower': 'DOUBLE', # Donchian Channel Lower
    }
    
    TREND_FOLLOWING = {
        'supertrend_7_3': 'DOUBLE',      # Supertrend (7,3)
        'supertrend_signal_7_3': 'BYTE', # 1=Buy, 0=Sell
        'supertrend_10_3': 'DOUBLE',     # Supertrend (10,3)
        'supertrend_signal_10_3': 'BYTE',
        'parabolic_sar': 'DOUBLE',       # Parabolic SAR
        'adx_14': 'DOUBLE',              # Average Directional Index
        'di_plus': 'DOUBLE',             # Directional Indicator +
        'di_minus': 'DOUBLE',            # Directional Indicator -
        'aroon_up': 'DOUBLE',            # Aroon Up (25)
        'aroon_down': 'DOUBLE',          # Aroon Down
        'aroon_oscillator': 'DOUBLE',    # Aroon Oscillator
    }
    
    VOLUME_INDICATORS = {
        'volume_sma_20': 'DOUBLE',       # 20-period Volume SMA
        'volume_ema_20': 'DOUBLE',       # 20-period Volume EMA
        'vwap': 'DOUBLE',                # Volume Weighted Average Price
        'twap': 'DOUBLE',                # Time Weighted Average Price
        'obv': 'LONG',                   # On Balance Volume
        'ad_line': 'DOUBLE',             # Accumulation/Distribution Line
        'cmf_20': 'DOUBLE',              # 20-period Chaikin Money Flow
        'mfi_14': 'DOUBLE',              # 14-period Money Flow Index
        'volume_rate': 'DOUBLE',         # Volume Rate (current/average)
    }
    
    # Options-Specific Greeks and Analytics
    OPTIONS_GREEKS = {
        'delta': 'DOUBLE',               # Price sensitivity
        'gamma': 'DOUBLE',               # Delta sensitivity  
        'theta': 'DOUBLE',               # Time decay
        'vega': 'DOUBLE',                # Volatility sensitivity
        'rho': 'DOUBLE',                 # Interest rate sensitivity
        'lambda_greek': 'DOUBLE',        # Leverage
        'epsilon': 'DOUBLE',             # Dividend sensitivity
        'vera': 'DOUBLE',                # Volatility elasticity
        'charm': 'DOUBLE',               # Delta decay
        'vanna': 'DOUBLE',               # Vega/Delta cross-sensitivity
        'volga': 'DOUBLE',               # Vega convexity
    }
    
    OPTIONS_ANALYTICS = {
        'implied_volatility': 'DOUBLE',   # IV
        'historical_volatility': 'DOUBLE', # 20-day HV
        'iv_rank': 'DOUBLE',             # IV Rank (0-100)
        'iv_percentile': 'DOUBLE',       # IV Percentile
        'intrinsic_value': 'DOUBLE',     # Intrinsic value
        'time_value': 'DOUBLE',          # Time value
        'moneyness': 'DOUBLE',           # S/K ratio
        'probability_itm': 'DOUBLE',     # Probability ITM
        'probability_profit': 'DOUBLE',  # Probability of profit
        'break_even': 'DOUBLE',          # Break-even price
        'max_profit': 'DOUBLE',          # Maximum profit potential
        'max_loss': 'DOUBLE',            # Maximum loss potential
    }
    
    # Market Microstructure (5-level depth)
    MARKET_DEPTH = {
        'bid_1': 'DOUBLE', 'bid_qty_1': 'LONG',
        'bid_2': 'DOUBLE', 'bid_qty_2': 'LONG',
        'bid_3': 'DOUBLE', 'bid_qty_3': 'LONG',
        'bid_4': 'DOUBLE', 'bid_qty_4': 'LONG',
        'bid_5': 'DOUBLE', 'bid_qty_5': 'LONG',
        'ask_1': 'DOUBLE', 'ask_qty_1': 'LONG',
        'ask_2': 'DOUBLE', 'ask_qty_2': 'LONG',
        'ask_3': 'DOUBLE', 'ask_qty_3': 'LONG',
        'ask_4': 'DOUBLE', 'ask_qty_4': 'LONG',
        'ask_5': 'DOUBLE', 'ask_qty_5': 'LONG',
    }
    
    DERIVED_MARKET_DATA = {
        'bid_ask_spread': 'DOUBLE',      # Ask1 - Bid1
        'bid_ask_spread_pct': 'DOUBLE',  # Spread as % of mid
        'mid_price': 'DOUBLE',           # (Bid1 + Ask1) / 2
        'weighted_mid_price': 'DOUBLE',  # Quantity-weighted mid
        'total_bid_qty': 'LONG',         # Sum of bid quantities
        'total_ask_qty': 'LONG',         # Sum of ask quantities
        'bid_ask_ratio': 'DOUBLE',       # Bid qty / Ask qty
        'market_impact': 'DOUBLE',       # Estimated market impact
    }

# Numba optimization settings
NUMBA_SETTINGS = {
    'nopython': True,           # Compile to machine code
    'nogil': True,              # Release GIL for parallel execution
    'cache': True,              # Cache compiled functions
    'fastmath': True,           # Enable fast math optimizations
    'parallel': True,           # Enable parallel execution where possible
    'boundscheck': False,       # Disable bounds checking for speed
    'wraparound': False,        # Disable negative indexing
    'cdivision': True,          # Enable C-style division
}

# Column groups for efficient updates
COLUMN_GROUPS = {
    'basic_ohlcv': ['open', 'high', 'low', 'close', 'volume'],
    'trend_indicators': ['ema_9', 'ema_21', 'ema_50', 'ema_200', 'sma_20', 'sma_50'],
    'momentum_indicators': ['rsi_14', 'macd_line', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d'],
    'volatility_indicators': ['atr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width'],
    'trend_following': ['supertrend_7_3', 'supertrend_signal_7_3', 'supertrend_10_3', 'supertrend_signal_10_3'],
    'options_greeks': ['delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility'],
    'market_depth': ['bid_1', 'bid_qty_1', 'ask_1', 'ask_qty_1', 'bid_2', 'bid_qty_2', 'ask_2', 'ask_qty_2'],
    'derived_metrics': ['bid_ask_spread', 'mid_price', 'price_change', 'price_change_pct']
}

# Index creation templates for performance
INDEX_TEMPLATES = {
    'timeframe_index': "CREATE INDEX IF NOT EXISTS idx_{table_name}_tf ON {table_name}(tf);",
    'timestamp_tf_index': "CREATE INDEX IF NOT EXISTS idx_{table_name}_ts_tf ON {table_name}(timestamp, tf);",
    'strike_index': "CREATE INDEX IF NOT EXISTS idx_{table_name}_strike ON {table_name}(strike);",
    'option_type_index': "CREATE INDEX IF NOT EXISTS idx_{table_name}_opt_type ON {table_name}(option_type);",
    'moneyness_index': "CREATE INDEX IF NOT EXISTS idx_{table_name}_moneyness ON {table_name}(moneyness);",
}

# Materialized view templates for common analytics
MATERIALIZED_VIEWS = {
    'daily_summary': """
        CREATE MATERIALIZED VIEW IF NOT EXISTS {table_name}_daily_summary AS
        SELECT 
            DATE(timestamp) as trade_date,
            FIRST(open) as day_open,
            MAX(high) as day_high,
            MIN(low) as day_low,
            LAST(close) as day_close,
            SUM(volume) as day_volume,
            LAST(oi) as day_end_oi,
            AVG(rsi_14) as avg_rsi,
            LAST(ema_50) as day_end_ema50,
            LAST(supertrend_7_3) as day_end_supertrend
        FROM {table_name}
        WHERE tf <= 60  -- Intraday data only
        GROUP BY DATE(timestamp)
        ORDER BY trade_date DESC;
    """,
    
    'options_chain_summary': """
        CREATE MATERIALIZED VIEW IF NOT EXISTS {table_name}_chain_summary AS
        SELECT 
            DATE(timestamp) as trade_date,
            strike / 100.0 as strike_price,
            SUM(CASE WHEN option_type = 1 THEN volume ELSE 0 END) as call_volume,
            SUM(CASE WHEN option_type = 2 THEN volume ELSE 0 END) as put_volume,
            LAST(CASE WHEN option_type = 1 THEN oi ELSE 0 END) as call_oi,
            LAST(CASE WHEN option_type = 2 THEN oi ELSE 0 END) as put_oi,
            AVG(CASE WHEN option_type = 1 THEN implied_volatility ELSE NULL END) as call_iv,
            AVG(CASE WHEN option_type = 2 THEN implied_volatility ELSE NULL END) as put_iv
        FROM {table_name}
        WHERE tf = 5  -- 5-minute data
        GROUP BY DATE(timestamp), strike
        ORDER BY trade_date DESC, strike_price;
    """
}

class SchemaValidator:
    """Validates and optimizes schema definitions"""
    
    @staticmethod
    def validate_column_types(schema: str) -> List[str]:
        """Validate column types for QuestDB compatibility"""
        
        warnings = []
        
        # Check for unsupported types
        unsupported_patterns = [
            'VARCHAR',  # Use STRING instead
            'INTEGER',  # Use INT instead
            'DECIMAL',  # Use DOUBLE instead
        ]
        
        for pattern in unsupported_patterns:
            if pattern in schema.upper():
                warnings.append(f"Consider replacing {pattern} with QuestDB-optimized type")
        
        return warnings
    
    @staticmethod
    def estimate_storage_size(
        columns: Dict[str, str],
        estimated_rows_per_day: int,
        days: int = 365
    ) -> Dict[str, float]:
        """Estimate storage requirements"""
        
        # Approximate sizes in bytes
        type_sizes = {
            'BYTE': 1,
            'INT': 4,
            'LONG': 8,
            'DOUBLE': 8,
            'SYMBOL': 4,  # Average
            'STRING': 20,  # Average
            'DATE': 4,
            'TIMESTAMP': 8
        }
        
        total_row_size = 0
        for col_type in columns.values():
            total_row_size += type_sizes.get(col_type, 8)
        
        total_rows = estimated_rows_per_day * days
        total_size_bytes = total_rows * total_row_size
        
        return {
            'row_size_bytes': total_row_size,
            'total_rows': total_rows,
            'total_size_mb': total_size_bytes / (1024 * 1024),
            'total_size_gb': total_size_bytes / (1024 * 1024 * 1024)
        }
