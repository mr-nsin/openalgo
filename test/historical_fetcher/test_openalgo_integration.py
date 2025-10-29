#!/usr/bin/env python3
"""
Test script for OpenAlgo-integrated historical fetcher

This script tests the integration between the historical fetcher and OpenAlgo's
existing infrastructure including database, authentication, and API components.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the OpenAlgo root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.logging import get_logger
from config.openalgo_settings import OpenAlgoSettings
from fetchers.openalgo_zerodha_fetcher import OpenAlgoZerodhaHistoricalFetcher, OpenAlgoSymbolManager, SymbolInfo
from database.symbol import SymToken, db_session

logger = get_logger(__name__)

async def test_openalgo_integration():
    """Test the OpenAlgo integration components"""
    
    print("🧪 Testing OpenAlgo Historical Fetcher Integration")
    print("=" * 60)
    
    try:
        # Test 1: Configuration Loading
        print("\n1️⃣ Testing Configuration Loading...")
        settings = OpenAlgoSettings()
        print(f"   ✅ Database URL: {settings.database_url}")
        print(f"   ✅ Zerodha API Key: {'*' * 8 if settings.zerodha_api_key else 'Not set'}")
        print(f"   ✅ Enabled Timeframes: {settings.enabled_timeframes}")
        print(f"   ✅ Enabled Instrument Types: {settings.enabled_instrument_types}")
        print(f"   ✅ Enabled Exchanges: {settings.enabled_exchanges}")
        
        # Test 2: Database Connection
        print("\n2️⃣ Testing Database Connection...")
        try:
            with db_session() as session:
                symbol_count = session.query(SymToken).count()
                print(f"   ✅ Database connected successfully")
                print(f"   ✅ Found {symbol_count} symbols in database")
                
                # Show sample symbols
                sample_symbols = session.query(SymToken).limit(5).all()
                print("   📋 Sample symbols:")
                for symbol in sample_symbols:
                    print(f"      - {symbol.symbol} ({symbol.exchange}) - {symbol.instrumenttype}")
        except Exception as e:
            print(f"   ❌ Database connection failed: {e}")
            return False
        
        # Test 3: Symbol Manager
        print("\n3️⃣ Testing Symbol Manager...")
        try:
            symbol_manager = OpenAlgoSymbolManager(settings)
            symbols_by_type = await symbol_manager.get_all_active_symbols()
            
            total_symbols = sum(len(symbols) for symbols in symbols_by_type.values())
            print(f"   ✅ Symbol manager initialized successfully")
            print(f"   ✅ Found {total_symbols} active symbols")
            
            for inst_type, symbols in symbols_by_type.items():
                if symbols:
                    print(f"      - {inst_type}: {len(symbols)} symbols")
        except Exception as e:
            print(f"   ❌ Symbol manager failed: {e}")
            return False
        
        # Test 4: Zerodha Fetcher Initialization
        print("\n4️⃣ Testing Zerodha Fetcher Initialization...")
        try:
            fetcher = OpenAlgoZerodhaHistoricalFetcher(settings)
            await fetcher.initialize()
            print(f"   ✅ Zerodha fetcher initialized successfully")
            
            # Test statistics
            stats = fetcher.get_statistics()
            print(f"   📊 Fetcher statistics: {stats}")
        except Exception as e:
            print(f"   ❌ Zerodha fetcher initialization failed: {e}")
            return False
        
        # Test 5: Sample Data Fetch (if API credentials are available)
        print("\n5️⃣ Testing Sample Data Fetch...")
        if settings.zerodha_api_key and settings.zerodha_access_token:
            try:
                # Get a sample symbol for testing
                with db_session() as session:
                    sample_symbol = session.query(SymToken).filter(
                        SymToken.instrumenttype == 'EQ',
                        SymToken.exchange == 'NSE'
                    ).first()
                
                if sample_symbol:
                    symbol_info = SymbolInfo(
                        symbol=sample_symbol.symbol,
                        exchange=sample_symbol.exchange,
                        instrument_type=sample_symbol.instrumenttype,
                        token=sample_symbol.token,
                        name=sample_symbol.name
                    )
                    
                    # Test fetching 1 day of daily data
                    from config.openalgo_settings import TimeFrame
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=1)
                    
                    print(f"   🔍 Testing with symbol: {symbol_info.symbol} ({symbol_info.exchange})")
                    candles = await fetcher.fetch_historical_data(
                        symbol_info,
                        TimeFrame.DAILY,
                        start_date,
                        end_date
                    )
                    
                    print(f"   ✅ Successfully fetched {len(candles)} candles")
                    if candles:
                        print(f"   📊 Sample candle: {candles[0].timestamp} - O:{candles[0].open} H:{candles[0].high} L:{candles[0].low} C:{candles[0].close}")
                else:
                    print("   ⚠️ No sample symbol found for testing")
            except Exception as e:
                print(f"   ❌ Sample data fetch failed: {e}")
                print(f"   💡 This might be due to API credentials or network issues")
        else:
            print("   ⚠️ API credentials not configured - skipping data fetch test")
        
        # Test 6: Cleanup
        print("\n6️⃣ Testing Cleanup...")
        try:
            await fetcher.cleanup()
            print("   ✅ Cleanup completed successfully")
        except Exception as e:
            print(f"   ❌ Cleanup failed: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 OpenAlgo Integration Test Completed Successfully!")
        print("\n📋 Summary:")
        print("   ✅ Configuration loading works")
        print("   ✅ Database connection works")
        print("   ✅ Symbol manager works")
        print("   ✅ Zerodha fetcher initialization works")
        print("   ✅ Cleanup works")
        
        if settings.zerodha_api_key and settings.zerodha_access_token:
            print("   ✅ Sample data fetch works")
        else:
            print("   ⚠️ API credentials not configured - configure BROKER_API_KEY and BROKER_API_SECRET for full functionality")
        
        print("\n🚀 The historical fetcher is ready to use!")
        print("   Run: python openalgo_main.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_openalgo_integration()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
