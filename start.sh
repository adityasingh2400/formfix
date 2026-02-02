#!/bin/bash

# FormFix Backend Startup Script
# This script sets up the virtual environment and starts the FastAPI server

set -e  # Exit on any error

echo "🏀 Starting FormFix Backend..."
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3.11 or Python 3 not found. Please install Python 3.11+"
        exit 1
    else
        PYTHON_CMD="python3"
        print_warning "Using python3 instead of python3.11"
    fi
else
    PYTHON_CMD="python3.11"
fi

print_info "Using Python: $PYTHON_CMD"

# Check if we're in the right directory
if [[ ! -f "backend/requirements.txt" ]]; then
    print_error "Please run this script from the formfix root directory (where backend/ folder is)"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [[ ! -d ".venv" ]]; then
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate

# Check if ffmpeg is installed (needed for video processing)
if ! command -v ffmpeg &> /dev/null; then
    print_warning "ffmpeg not found. Install it for better video compatibility:"
    print_info "  macOS: brew install ffmpeg"
    print_info "  Ubuntu/Debian: sudo apt install ffmpeg"
fi

# Install/update dependencies
print_info "Installing/updating Python dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt
print_status "Dependencies installed"

# Verify MediaPipe installation
print_info "Verifying MediaPipe installation..."
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__); assert hasattr(mp, 'solutions'), 'MediaPipe solutions not available'" || {
    print_error "MediaPipe verification failed. Trying to reinstall..."
    pip install --force-reinstall mediapipe==0.10.9
}
print_status "MediaPipe verified"

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port 8000 is already in use. Trying port 8001..."
    PORT=8001
else
    PORT=8000
fi

print_status "All setup complete!"
echo ""
echo "================================"
print_info "Starting FormFix Backend Server on port $PORT..."
print_info "API will be available at: http://127.0.0.1:$PORT"
print_info "Health check: http://127.0.0.1:$PORT/health"
print_info "API docs: http://127.0.0.1:$PORT/docs"
echo ""
print_info "Press Ctrl+C to stop the server"
echo "================================"
echo ""

# Start the FastAPI server
python -m uvicorn backend.src.main:app --reload --port $PORT --host 127.0.0.1

# Cleanup message
echo ""
print_info "FormFix Backend stopped. Have a great day! 🏀"