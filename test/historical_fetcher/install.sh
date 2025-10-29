#!/bin/bash

# Historical Data Fetcher Installation Script
# This script sets up the complete environment for the Historical Data Fetcher

set -e  # Exit on any error

echo "🚀 Setting up Historical Data Fetcher..."
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons."
   exit 1
fi

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    DISTRO=$(lsb_release -si 2>/dev/null || echo "Unknown")
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    OS="unknown"
fi

print_status "Detected OS: $OS"

# Step 1: Create directory structure
print_step "Creating directory structure..."
mkdir -p logs
mkdir -p tmp
mkdir -p data/questdb

# Create __init__.py files for Python modules
find . -type d -name "__pycache__" -prune -o -type d -print | while read dir; do
    if [[ "$dir" != "." && "$dir" != "./logs" && "$dir" != "./tmp" && "$dir" != "./data" ]]; then
        touch "$dir/__init__.py" 2>/dev/null || true
    fi
done

print_status "Directory structure created"

# Step 2: Check Python version
print_step "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
        print_status "Python $PYTHON_VERSION found (compatible)"
        PYTHON_CMD="python3"
    else
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Step 3: Create virtual environment (optional but recommended)
print_step "Setting up Python virtual environment..."
if [[ ! -d "venv" ]]; then
    $PYTHON_CMD -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
if [[ "$OS" == "windows" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

print_status "Virtual environment activated"

# Step 4: Upgrade pip and install dependencies
print_step "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

print_status "Python dependencies installed"

# Step 5: Install and setup QuestDB
print_step "Setting up QuestDB..."

if command -v questdb &> /dev/null; then
    print_status "QuestDB already installed"
else
    print_status "Installing QuestDB..."
    
    if [[ "$OS" == "linux" ]]; then
        # Linux installation
        if command -v curl &> /dev/null; then
            curl -L https://github.com/questdb/questdb/releases/latest/download/questdb-7.3.10-rt-linux-amd64.tar.gz | tar xz
            sudo mv questdb-7.3.10-rt-linux-amd64 /opt/questdb
            sudo ln -sf /opt/questdb/bin/questdb.sh /usr/local/bin/questdb
            print_status "QuestDB installed to /opt/questdb"
        else
            print_error "curl not found. Please install curl and run again."
            exit 1
        fi
    elif [[ "$OS" == "macos" ]]; then
        # macOS installation
        if command -v brew &> /dev/null; then
            brew install questdb
            print_status "QuestDB installed via Homebrew"
        else
            print_warning "Homebrew not found. Please install QuestDB manually from https://questdb.io/get-questdb/"
        fi
    else
        print_warning "Please install QuestDB manually from https://questdb.io/get-questdb/"
    fi
fi

# Step 6: Start QuestDB service
print_step "Starting QuestDB service..."
if command -v questdb &> /dev/null; then
    # Check if QuestDB is already running
    if pgrep -f "questdb" > /dev/null; then
        print_status "QuestDB is already running"
    else
        # Start QuestDB in background
        nohup questdb start -d data/questdb > logs/questdb.log 2>&1 &
        sleep 5
        
        if pgrep -f "questdb" > /dev/null; then
            print_status "QuestDB started successfully"
            print_status "QuestDB Web Console: http://localhost:9000"
        else
            print_warning "QuestDB may not have started properly. Check logs/questdb.log"
        fi
    fi
else
    print_warning "QuestDB not found in PATH. Please start it manually."
fi

# Step 7: Create configuration file
print_step "Setting up configuration..."
if [[ ! -f ".env" ]]; then
    cp env_template.txt .env
    print_status "Configuration template copied to .env"
    print_warning "Please edit .env file with your actual configuration values"
else
    print_status "Configuration file .env already exists"
fi

# Step 8: Test database connections
print_step "Testing database connections..."

# Test QuestDB connection
$PYTHON_CMD -c "
import asyncio
import asyncpg
import sys

async def test_questdb():
    try:
        conn = await asyncpg.connect(host='localhost', port=9000, database='qdb')
        await conn.execute('SELECT 1')
        await conn.close()
        print('✅ QuestDB connection successful')
        return True
    except Exception as e:
        print(f'❌ QuestDB connection failed: {e}')
        return False

if asyncio.run(test_questdb()):
    sys.exit(0)
else:
    sys.exit(1)
" && QUESTDB_OK=1 || QUESTDB_OK=0

# Test OpenAlgo database connection
if [[ -f "../../db/openalgo.db" ]]; then
    print_status "OpenAlgo database found"
    OPENALGO_OK=1
else
    print_warning "OpenAlgo database not found at ../../db/openalgo.db"
    print_warning "Make sure OpenAlgo is properly installed and master contract is downloaded"
    OPENALGO_OK=0
fi

# Step 9: Create systemd service (Linux only)
if [[ "$OS" == "linux" && $QUESTDB_OK -eq 1 ]]; then
    print_step "Setting up systemd service..."
    
    SERVICE_FILE="/etc/systemd/system/historical-fetcher.service"
    TIMER_FILE="/etc/systemd/system/historical-fetcher.timer"
    CURRENT_USER=$(whoami)
    CURRENT_DIR=$(pwd)
    
    # Create service file
    sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Historical Data Fetcher
After=network.target

[Service]
Type=oneshot
User=$CURRENT_USER
WorkingDirectory=$CURRENT_DIR
Environment=PATH=$CURRENT_DIR/venv/bin
ExecStart=$CURRENT_DIR/venv/bin/python openalgo_main.py
StandardOutput=append:$CURRENT_DIR/logs/service.log
StandardError=append:$CURRENT_DIR/logs/service.log

[Install]
WantedBy=multi-user.target
EOF

    # Create timer file
    sudo tee $TIMER_FILE > /dev/null <<EOF
[Unit]
Description=Run Historical Data Fetcher daily
Requires=historical-fetcher.service

[Timer]
OnCalendar=*-*-* 18:30:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Enable and start timer
    sudo systemctl daemon-reload
    sudo systemctl enable historical-fetcher.timer
    sudo systemctl start historical-fetcher.timer
    
    print_status "Systemd service and timer created"
    print_status "Service will run daily at 6:30 PM IST"
fi

# Step 10: Create startup scripts
print_step "Creating startup scripts..."

# Create run script
cat > run.sh << 'EOF'
#!/bin/bash
# Historical Data Fetcher Runner

# Activate virtual environment
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
elif [[ -f "venv/Scripts/activate" ]]; then
    source venv/Scripts/activate
fi

# Run the fetcher
python openalgo_main.py "$@"
EOF

chmod +x run.sh

# Create scheduler script
cat > run_scheduler.sh << 'EOF'
#!/bin/bash
# Historical Data Fetcher Scheduler Runner

# Activate virtual environment
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
elif [[ -f "venv/Scripts/activate" ]]; then
    source venv/Scripts/activate
fi

# Run the scheduler
python scheduler/cron_scheduler.py "$@"
EOF

chmod +x run_scheduler.sh

print_status "Startup scripts created"

# Step 11: Final summary
echo ""
echo "🎉 Installation completed!"
echo "========================="
echo ""
print_status "Summary:"
echo "  • Python virtual environment: $(if [[ -d "venv" ]]; then echo "✅ Created"; else echo "❌ Failed"; fi)"
echo "  • Dependencies: ✅ Installed"
echo "  • QuestDB: $(if [[ $QUESTDB_OK -eq 1 ]]; then echo "✅ Running"; else echo "❌ Not running"; fi)"
echo "  • OpenAlgo DB: $(if [[ $OPENALGO_OK -eq 1 ]]; then echo "✅ Found"; else echo "⚠️ Not found"; fi)"
echo "  • Configuration: $(if [[ -f ".env" ]]; then echo "✅ Template created"; else echo "❌ Failed"; fi)"
echo ""

print_status "Next steps:"
echo "  1. Edit .env file with your API credentials and settings"
echo "  2. Ensure OpenAlgo is running and master contract is downloaded"
echo "  3. Test the setup: ./run.sh"
echo "  4. Start scheduler service: ./run_scheduler.sh"
echo ""

print_status "Useful commands:"
echo "  • Manual run: ./run.sh"
echo "  • Run scheduler: ./run_scheduler.sh"
echo "  • View logs: tail -f logs/historical_fetcher.log"
echo "  • QuestDB console: http://localhost:9000"
echo ""

if [[ $QUESTDB_OK -eq 0 ]]; then
    print_warning "QuestDB is not running. Please start it manually before running the fetcher."
fi

if [[ $OPENALGO_OK -eq 0 ]]; then
    print_warning "OpenAlgo database not found. Please ensure OpenAlgo is properly installed."
fi

print_status "Installation script completed!"
