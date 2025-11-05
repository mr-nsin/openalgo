"""
Optimized QuestDB Schema Analysis for Historical Data

This file analyzes different approaches for timeframe storage and indexing
to optimize data retrieval performance.
"""

from enum import IntEnum
from typing import Dict, List

class TimeFrameEnum(IntEnum):
    """Optimized timeframe enum for compact storage"""
    MINUTE_1 = 1    # 1m
    MINUTE_3 = 2    # 3m  
    MINUTE_5 = 3    # 5m
    MINUTE_15 = 4   # 15m
    MINUTE_30 = 5   # 30m
    HOUR_1 = 6      # 1h
    DAILY = 7       # D
    
    @classmethod
    def from_string(cls, timeframe_str: str):
        """Convert string timeframe to enum"""
        mapping = {
            '1m': cls.MINUTE_1,
            '3m': cls.MINUTE_3,
            '5m': cls.MINUTE_5,
            '15m': cls.MINUTE_15,
            '30m': cls.MINUTE_30,
            '1h': cls.HOUR_1,
            'D': cls.DAILY
        }
        return mapping.get(timeframe_str, cls.MINUTE_1)
    
    def to_string(self) -> str:
        """Convert enum to string"""
        mapping = {
            self.MINUTE_1: '1m',
            self.MINUTE_3: '3m',
            self.MINUTE_5: '5m',
            self.MINUTE_15: '15m',
            self.MINUTE_30: '30m',
            self.HOUR_1: '1h',
            self.DAILY: 'D'
        }
        return mapping.get(self, '1m')

# Performance Analysis for Different Storage Types
STORAGE_ANALYSIS = {
    'SYMBOL': {
        'storage_bytes': 4,  # Reference to symbol table
        'query_performance': 'Excellent (optimized for repeated strings)',
        'human_readable': True,
        'range_queries': 'Good (string comparison)',
        'memory_usage': 'Low (deduplicated)',
        'recommended_for': 'Most use cases - best balance'
    },
    'BYTE': {
        'storage_bytes': 1,
        'query_performance': 'Excellent (fastest)',
        'human_readable': False,
        'range_queries': 'Excellent (numeric)',
        'memory_usage': 'Lowest',
        'recommended_for': 'High-frequency data, storage-critical'
    },
    'SHORT': {
        'storage_bytes': 2,
        'query_performance': 'Excellent',
        'human_readable': False,
        'range_queries': 'Excellent (numeric)',
        'memory_usage': 'Low',
        'recommended_for': 'When BYTE range is insufficient'
    }
}

# Optimized Schema Definitions
OPTIMIZED_SCHEMAS = {
    'equity_v2': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            tf SYMBOL CAPACITY 10 CACHE,    -- Timeframe: '1m','3m','5m','15m','30m','1h','D'
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        
        -- Optimized indexes for fast retrieval
        CREATE INDEX IF NOT EXISTS idx_{table_name}_tf_ts ON {table_name}(tf, timestamp);
    """,
    
    'equity_compact': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            tf BYTE,                         -- Timeframe enum: 1-7
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        
        -- Optimized indexes
        CREATE INDEX IF NOT EXISTS idx_{table_name}_tf_ts ON {table_name}(tf, timestamp);
    """
}

# Query Performance Patterns
QUERY_PATTERNS = {
    'latest_by_timeframe': {
        'description': 'Get latest data for specific timeframe',
        'symbol_version': "SELECT * FROM {table} WHERE tf = '1h' ORDER BY timestamp DESC LIMIT 100",
        'byte_version': "SELECT * FROM {table} WHERE tf = 6 ORDER BY timestamp DESC LIMIT 100",
        'performance': 'Both excellent, SYMBOL slightly more readable'
    },
    
    'time_range_query': {
        'description': 'Get data for time range and timeframe',
        'symbol_version': """
            SELECT * FROM {table} 
            WHERE tf = 'D' 
            AND timestamp BETWEEN '2024-01-01' AND '2024-12-31'
            ORDER BY timestamp
        """,
        'byte_version': """
            SELECT * FROM {table} 
            WHERE tf = 7 
            AND timestamp BETWEEN '2024-01-01' AND '2024-12-31'
            ORDER BY timestamp
        """,
        'performance': 'BYTE version slightly faster due to numeric comparison'
    },
    
    'multi_timeframe_query': {
        'description': 'Get data for multiple timeframes',
        'symbol_version': "SELECT * FROM {table} WHERE tf IN ('1m','5m','1h')",
        'byte_version': "SELECT * FROM {table} WHERE tf IN (1,3,6)",
        'performance': 'BYTE version faster for IN clauses'
    }
}

# Partitioning Strategy Analysis
PARTITIONING_STRATEGIES = {
    'current_daily': {
        'strategy': 'PARTITION BY DAY',
        'pros': ['Good for daily queries', 'Reasonable partition size'],
        'cons': ['May create many small partitions for intraday data'],
        'best_for': 'Mixed timeframe queries'
    },
    
    'monthly': {
        'strategy': 'PARTITION BY MONTH',
        'pros': ['Fewer partitions', 'Good for historical analysis'],
        'cons': ['Larger partition size', 'Slower for recent data queries'],
        'best_for': 'Long-term historical analysis'
    },
    
    'weekly': {
        'strategy': 'PARTITION BY WEEK',
        'pros': ['Balance between size and count', 'Good for weekly analysis'],
        'cons': ['Week boundaries may not align with trading patterns'],
        'best_for': 'Weekly trading strategies'
    }
}

def get_recommended_schema(use_case: str) -> str:
    """Get recommended schema based on use case"""
    
    recommendations = {
        'general': 'equity_v2',  # SYMBOL type for readability
        'high_frequency': 'equity_compact',  # BYTE type for performance
        'storage_critical': 'equity_compact',
        'human_readable': 'equity_v2'
    }
    
    return OPTIMIZED_SCHEMAS.get(recommendations.get(use_case, 'equity_v2'))

if __name__ == "__main__":
    print("=== QuestDB Schema Optimization Analysis ===")
    print("\n1. Storage Type Comparison:")
    for storage_type, analysis in STORAGE_ANALYSIS.items():
        print(f"\n{storage_type}:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
    
    print("\n2. Recommended Schema (General Use):")
    print(get_recommended_schema('general'))
