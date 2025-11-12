#!/bin/bash

# Strategy Builder - Backtest Runner Script
# This script makes it easy to run backtests with common configurations

# Default values
SYMBOL="RELIANCE"
EXCHANGE="NSE"
INTERVAL="D"
START_DATE=$(date -d "1 year ago" +%Y-%m-%d 2>/dev/null || date -v-1y +%Y-%m-%d)
END_DATE=$(date +%Y-%m-%d)
INITIAL_CAPITAL=100000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --exchange)
            EXCHANGE="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --initial-capital)
            INITIAL_CAPITAL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --symbol SYMBOL          Trading symbol (default: RELIANCE)"
            echo "  --exchange EXCHANGE       Exchange (default: NSE)"
            echo "  --interval INTERVAL      Time interval (default: D)"
            echo "  --start-date DATE        Start date YYYY-MM-DD (default: 1 year ago)"
            echo "  --end-date DATE          End date YYYY-MM-DD (default: today)"
            echo "  --initial-capital AMOUNT Initial capital (default: 100000)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --symbol TCS --start-date 2023-01-01 --end-date 2024-01-01"
            echo "  $0 --symbol RELIANCE --interval 5m --start-date 2024-01-01"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to script directory
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Creating from template..."
    if [ -f env_template.txt ]; then
        cp env_template.txt .env
        echo "Please edit .env file with your OpenAlgo API key and other settings"
        echo "Then run this script again."
        exit 1
    else
        echo "Error: env_template.txt not found. Cannot create .env file."
        exit 1
    fi
fi

# Run the backtest
echo "Running backtest..."
echo "Symbol: $SYMBOL | Exchange: $EXCHANGE | Interval: $INTERVAL"
echo "Period: $START_DATE to $END_DATE"
echo "Initial Capital: $INITIAL_CAPITAL"
echo ""

python main.py \
    --symbol "$SYMBOL" \
    --exchange "$EXCHANGE" \
    --interval "$INTERVAL" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --initial-capital "$INITIAL_CAPITAL"

