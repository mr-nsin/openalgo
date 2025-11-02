#!/bin/bash

# OpenAlgo Historical Data Fetcher - Run Script
# This script runs the historical data fetcher with proper environment setup

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

# Function to check if virtual environment exists
check_venv() {
    if [ -d "venv" ]; then
        print_status "Virtual environment found"
        return 0
    else
        print_warning "Virtual environment not found"
        return 1
    fi
}

# Function to create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
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
        print_error "Cannot find virtual environment activation script"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed from requirements.txt"
    else
        print_warning "requirements.txt not found, installing basic dependencies..."
        pip install loguru pydantic python-dotenv asyncio aiohttp asyncpg pandas
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
        print_status "You can copy env_template.txt to .env and fill in your values"
        return 1
    fi
}

# Function to run the fetcher
run_fetcher() {
    print_status "Starting OpenAlgo Historical Data Fetcher..."
    print_status "Press Ctrl+C to stop"
    echo
    
    # Run the main script
    $PYTHON_CMD openalgo_main.py "$@"
}

# Function to show help
show_help() {
    echo "OpenAlgo Historical Data Fetcher - Run Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --no-venv      Skip virtual environment setup"
    echo "  --no-deps      Skip dependency installation"
    echo "  --check-only   Only check environment, don't run fetcher"
    echo
    echo "Environment Variables:"
    echo "  HIST_FETCHER_LOG_LEVEL    Set log level (DEBUG, INFO, WARNING, ERROR)"
    echo "  HIST_FETCHER_QUESTDB_HOST Set QuestDB host (default: localhost)"
    echo "  HIST_FETCHER_QUESTDB_PORT Set QuestDB port (default: 9000)"
    echo
    echo "Examples:"
    echo "  $0                                    # Run with default settings"
    echo "  $0 --check-only                      # Check environment only"
    echo "  HIST_FETCHER_LOG_LEVEL=DEBUG $0      # Run with debug logging"
}

# Parse command line arguments
SKIP_VENV=false
SKIP_DEPS=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --no-venv)
            SKIP_VENV=true
            shift
            ;;
        --no-deps)
            SKIP_DEPS=true
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        *)
            # Pass remaining arguments to the Python script
            break
            ;;
    esac
done

# Main execution
main() {
    print_status "OpenAlgo Historical Data Fetcher - Starting..."
    echo
    
    # Check Python version
    check_python_version
    
    # Check if we should skip virtual environment setup
    if [ "$SKIP_VENV" = false ]; then
        # Check or create virtual environment
        if ! check_venv; then
            create_venv
        fi
        
        # Activate virtual environment
        activate_venv
    else
        print_warning "Skipping virtual environment setup"
    fi
    
    # Check if we should skip dependency installation
    if [ "$SKIP_DEPS" = false ]; then
        # Install dependencies
        install_dependencies
    else
        print_warning "Skipping dependency installation"
    fi
    
    # Check for .env file
    check_env_file
    
    # If check-only mode, exit here
    if [ "$CHECK_ONLY" = true ]; then
        print_success "Environment check completed"
        exit 0
    fi
    
    # Run the fetcher
    run_fetcher "$@"
}

# Run main function
main "$@"
