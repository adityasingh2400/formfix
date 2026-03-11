#!/bin/bash

# FormFix Frontend Startup Script
# Run this from the frontend/ directory: ./front.sh

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

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the frontend directory (where this script lives)
cd "$SCRIPT_DIR"

# Verify index.html exists
if [[ ! -f "index.html" ]]; then
    echo "Could not find index.html. Make sure you're running from the frontend directory."
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
python3 -m http.server $PORT
