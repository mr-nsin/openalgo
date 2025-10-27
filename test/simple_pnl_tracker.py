#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple P&L Tracker - Basic Position Monitoring

A simplified version that fetches positions and displays P&L without WebSocket.
Good for basic monitoring and testing.

Usage:
    python simple_pnl_tracker.py
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.positionbook_service import get_positionbook
from utils.logging import get_logger

logger = get_logger(__name__)

# Configuration
API_KEY = os.getenv("OPENALGO_API_KEY", "4729b0f39e8d894b6667b95ab31994b06a8ce1f2fd40e8e4d1f9a74051d7c75f")
REFRESH_INTERVAL = 5.0  # seconds between updates

# Color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def colorize(text, color):
    """Add color to text"""
    return f"{color}{text}{Colors.END}"

def format_currency(amount):
    """Format currency with proper alignment"""
    return f"₹{amount:,.2f}"

def format_percentage(pct):
    """Format percentage with color"""
    if pct > 0:
        return colorize(f"+{pct:.2f}%", Colors.GREEN)
    elif pct < 0:
        return colorize(f"{pct:.2f}%", Colors.RED)
    else:
        return f"{pct:.2f}%"

def format_pnl(pnl):
    """Format P&L with color"""
    if pnl > 0:
        return colorize(f"+₹{pnl:,.2f}", Colors.GREEN)
    elif pnl < 0:
        return colorize(f"₹{pnl:,.2f}", Colors.RED)
    else:
        return f"₹{pnl:,.2f}"

def fetch_positions():
    """Fetch all positions from broker"""
    try:
        print("📊 Fetching positions...")
        success, response_data, status_code = get_positionbook(api_key=API_KEY)
        
        if not success:
            print(f"❌ Failed to fetch positions: {response_data.get('message', 'Unknown error')}")
            return []

        positions = response_data.get('data', [])
        # Filter out zero quantity positions
        active_positions = [pos for pos in positions if pos.get('quantity', 0) != 0]
        
        print(f"✅ Found {len(active_positions)} active positions")
        return active_positions

    except Exception as e:
        print(f"❌ Error fetching positions: {e}")
        return []

def display_positions(positions):
    """Display positions in a table format"""
    if not positions:
        print("ℹ️  No active positions found")
        return

    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Header
    now = datetime.now().strftime('%H:%M:%S')
    print("=" * 100)
    print(f"{Colors.BOLD}💰 SIMPLE P&L TRACKER{Colors.END}  |  ⏰ {now}  |  📊 {len(positions)} Positions")
    print("=" * 100)

    # Table headers
    print(f"{'Symbol':<15} {'Exchange':<8} {'Product':<8} {'Qty':<8} {'Avg Price':<12} {'LTP':<12} {'P&L':<15} {'P&L %':<10}")
    print("-" * 100)

    total_pnl = 0.0
    total_investment = 0.0
    profit_count = 0
    loss_count = 0

    for pos in positions:
        symbol = pos.get('symbol', 'N/A')[:15]
        exchange = pos.get('exchange', 'N/A')[:8]
        product = pos.get('product', 'N/A')[:8]
        quantity = pos.get('quantity', 0)
        avg_price = pos.get('average_price', 0.0)
        ltp = pos.get('ltp', 0.0)
        pnl = pos.get('pnl', 0.0)
        pnl_pct = pos.get('pnl_percentage', 0.0)

        # Calculate totals
        total_pnl += pnl
        if quantity != 0:
            total_investment += abs(quantity) * avg_price

        # Count profit/loss positions
        if pnl > 0:
            profit_count += 1
        elif pnl < 0:
            loss_count += 1

        # Format values
        qty_str = f"{quantity:>8}"
        avg_str = f"{format_currency(avg_price):>12}"
        ltp_str = f"{format_currency(ltp):>12}"
        pnl_str = f"{format_pnl(pnl):>15}"
        pnl_pct_str = f"{format_percentage(pnl_pct):>10}"

        print(f"{symbol:<15} {exchange:<8} {product:<8} {qty_str} {avg_str} {ltp_str} {pnl_str} {pnl_pct_str}")

    # Summary
    print("-" * 100)
    total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0.0
    
    print(f"\n📈 PORTFOLIO SUMMARY:")
    print(f"Total Investment: {format_currency(total_investment)}")
    print(f"Total P&L: {format_pnl(total_pnl)} ({format_percentage(total_pnl_pct)})")
    print(f"Profit Positions: {colorize(str(profit_count), Colors.GREEN)}")
    print(f"Loss Positions: {colorize(str(loss_count), Colors.RED)}")
    print(f"Neutral Positions: {colorize(str(len(positions) - profit_count - loss_count), Colors.YELLOW)}")
    
    print("=" * 100)
    print("🔄 Auto-refresh every 5 seconds | Ctrl+C to stop")

def main():
    """Main monitoring loop"""
    print("🚀 Starting Simple P&L Tracker...")
    
    try:
        while True:
            positions = fetch_positions()
            display_positions(positions)
            time.sleep(REFRESH_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Error in simple P&L tracker")

if __name__ == "__main__":
    main()

