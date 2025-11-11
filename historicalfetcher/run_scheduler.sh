#!/bin/bash

# OpenAlgo Historical Data Fetcher - Scheduler Service
# This script runs the scheduler service for daily automated data fetching

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check Python version (3.8+ required)
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION="3.8"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        print_error "Python 3.8+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python version: $PYTHON_VERSION"
}

# Function to activate virtual environment
activate_venv() {
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated (Linux/Mac)"
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        print_success "Virtual environment activated (Windows)"
    else
        print_warning "Virtual environment not found, using system Python"
    fi
}

# Function to check if .env file exists
check_env_file() {
    if [ -f ".env" ]; then
        print_success ".env file found"
        return 0
    elif [ -f "../.env" ]; then
        print_success "OpenAlgo .env file found in parent directory"
        return 0
    else
        print_warning "No .env file found"
        print_status "Please create a .env file with your configuration"
        return 1
    fi
}

# Function to run the scheduler
run_scheduler() {
    print_status "Starting OpenAlgo Historical Data Scheduler..."
    print_status "The scheduler will run:"
    print_status "  • Daily fetch at 6:30 PM IST (after market close)"
    print_status "  • Weekend full sync on Saturday 10:00 PM IST"
    print_status "  • Monthly cleanup on first Sunday at 2:00 AM IST"
    print_status ""
    print_status "Press Ctrl+C to stop"
    echo
    
    # Run the scheduler
    $PYTHON_CMD -m historicalfetcher.scheduler.cron_scheduler
}

# Main execution
main() {
    print_status "OpenAlgo Historical Data Scheduler - Starting..."
    echo
    
    # Check Python version
    check_python_version
    
    # Activate virtual environment if it exists
    activate_venv
    
    # Check for .env file
    check_env_file
    
    # Run the scheduler
    run_scheduler
}

# Run main function
main "$@"

