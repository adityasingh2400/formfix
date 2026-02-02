#!/bin/bash

# FormFix Frontend Startup Script
# This script starts the frontend development server

set -e  # Exit on any error

echo "🏀 Starting FormFix Frontend..."
echo "================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "frontend/index.html" ]]; then
    echo "Please run this script from the formfix root directory (where frontend/ folder is)"
    exit 1
fi

# Check if port 3000 is already in use
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_info "Port 3000 is already in use. Trying port 3001..."
    PORT=3001
else
    PORT=3000
fi

print_status "Starting FormFix Frontend on port $PORT..."
print_info "Frontend available at: http://localhost:$PORT"
print_info "Make sure the backend is running on port 8000 or 8001"
echo ""
print_info "Press Ctrl+C to stop the server"
echo "================================"
echo ""

# Start the frontend server
cd frontend && python3 -m http.server $PORT