#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIFTY Option Symbol Checker
Utility script to check if NIFTY28OCT2525000CE exists in the database and get its mapping

Usage:
    python check_nifty_symbol.py
"""

import os
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.token_db import get_br_symbol, get_token, get_oa_symbol
from database.symbol import SymToken, db_session
from utils.logging import get_logger

logger = get_logger(__name__)

def check_symbol_in_database(symbol: str, exchange: str):
    """Check if symbol exists in database and show details"""
    print(f"🔍 Checking symbol: {symbol} on {exchange}")
    print("-" * 50)
    
    try:
        # Method 1: Using token_db functions
        print("Method 1: Using token_db functions")
        br_symbol = get_br_symbol(symbol, exchange)
        token = get_token(symbol, exchange)
        
        if br_symbol:
            print(f"✅ Broker Symbol: {br_symbol}")
        else:
            print(f"❌ No broker symbol found")
        
        if token:
            print(f"✅ Token: {token}")
        else:
            print(f"❌ No token found")
        
        print()
        
        # Method 2: Direct database query
        print("Method 2: Direct database query")
        
        # Query by OpenAlgo symbol
        oa_result = db_session.query(SymToken).filter(
            SymToken.symbol == symbol,
            SymToken.exchange == exchange
        ).first()
        
        if oa_result:
            print(f"✅ Found by OpenAlgo symbol:")
            print(f"   ID: {oa_result.id}")
            print(f"   Symbol: {oa_result.symbol}")
            print(f"   Broker Symbol: {oa_result.brsymbol}")
            print(f"   Exchange: {oa_result.exchange}")
            print(f"   Broker Exchange: {oa_result.brexchange}")
            print(f"   Token: {oa_result.token}")
            print(f"   Name: {oa_result.name}")
            print(f"   Expiry: {oa_result.expiry}")
            print(f"   Strike: {oa_result.strike}")
            print(f"   Lot Size: {oa_result.lotsize}")
            print(f"   Instrument Type: {oa_result.instrumenttype}")
        else:
            print(f"❌ Not found by OpenAlgo symbol")
        
        print()
        
        # Method 3: Search for similar symbols
        print("Method 3: Searching for similar NIFTY options")
        
        similar_symbols = db_session.query(SymToken).filter(
            SymToken.symbol.like(f"NIFTY%OCT25%CE"),
            SymToken.exchange == exchange
        ).limit(10).all()
        
        if similar_symbols:
            print(f"✅ Found {len(similar_symbols)} similar symbols:")
            for sym in similar_symbols:
                print(f"   {sym.symbol} -> {sym.brsymbol} (Strike: {sym.strike})")
        else:
            print(f"❌ No similar symbols found")
        
        print()
        
        # Method 4: Search by broker symbol pattern
        print("Method 4: Searching by broker symbol pattern")
        
        # Try different broker symbol patterns
        patterns = [
            "NIFTY25OCT28%",  # Zerodha format might be different
            "NIFTY28OCT25%",
            "%28OCT25%25000CE%",
            "%NIFTY%25000CE%"
        ]
        
        for pattern in patterns:
            broker_results = db_session.query(SymToken).filter(
                SymToken.brsymbol.like(pattern),
                SymToken.exchange == exchange
            ).limit(5).all()
            
            if broker_results:
                print(f"✅ Pattern '{pattern}' found {len(broker_results)} matches:")
                for result in broker_results:
                    print(f"   {result.brsymbol} -> {result.symbol}")
                break
        else:
            print(f"❌ No matches found for any pattern")
        
    except Exception as e:
        print(f"❌ Error checking symbol: {e}")
        logger.exception("Symbol check error")
    finally:
        db_session.close()

def search_nifty_options(exchange: str = "NFO", limit: int = 20):
    """Search for NIFTY options in the database"""
    print(f"\n🔍 Searching for NIFTY options on {exchange} (limit: {limit})")
    print("-" * 60)
    
    try:
        # Search for NIFTY options
        nifty_options = db_session.query(SymToken).filter(
            SymToken.symbol.like("NIFTY%"),
            SymToken.exchange == exchange,
            SymToken.instrumenttype.in_(["CE", "PE"])
        ).order_by(SymToken.expiry, SymToken.strike).limit(limit).all()
        
        if nifty_options:
            print(f"✅ Found {len(nifty_options)} NIFTY options:")
            print(f"{'OpenAlgo Symbol':<25} {'Broker Symbol':<25} {'Strike':<8} {'Expiry':<12} {'Type':<4}")
            print("-" * 80)
            
            for option in nifty_options:
                print(f"{option.symbol:<25} {option.brsymbol:<25} {option.strike:<8} {option.expiry:<12} {option.instrumenttype:<4}")
        else:
            print(f"❌ No NIFTY options found on {exchange}")
    
    except Exception as e:
        print(f"❌ Error searching NIFTY options: {e}")
        logger.exception("NIFTY options search error")
    finally:
        db_session.close()

def check_database_status():
    
    """Check database connection and symbol count"""
    print("🔍 Checking database status")
    print("-" * 30)
    
    try:
        # Check total symbol count
        total_symbols = db_session.query(SymToken).count()
        print(f"✅ Total symbols in database: {total_symbols}")
        
        # Check NFO symbols
        nfo_symbols = db_session.query(SymToken).filter(SymToken.exchange == "NFO").count()
        print(f"✅ NFO symbols: {nfo_symbols}")
        
        # Check NIFTY symbols
        nifty_symbols = db_session.query(SymToken).filter(
            SymToken.symbol.like("NIFTY%"),
            SymToken.exchange == "NFO"
        ).count()
        print(f"✅ NIFTY symbols on NFO: {nifty_symbols}")
        
        # Check option types
        ce_count = db_session.query(SymToken).filter(
            SymToken.instrumenttype == "CE",
            SymToken.exchange == "NFO"
        ).count()
        
        pe_count = db_session.query(SymToken).filter(
            SymToken.instrumenttype == "PE",
            SymToken.exchange == "NFO"
        ).count()
        
        print(f"✅ Call Options (CE) on NFO: {ce_count}")
        print(f"✅ Put Options (PE) on NFO: {pe_count}")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        logger.exception("Database check error")
    finally:
        db_session.close()

def main():
    """Main function"""
    print("NIFTY Option Symbol Checker")
    print("=" * 50)
    
    # Check database status
    check_database_status()
    
    # Check specific symbol
    symbol = "NIFTY28OCT2525000CE"
    exchange = "NFO"
    check_symbol_in_database(symbol, exchange)
    
    # Search for NIFTY options
    search_nifty_options(exchange)
    
    print("\n" + "=" * 50)
    print("Symbol check completed!")

if __name__ == "__main__":
    main()
