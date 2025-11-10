#!/bin/bash
# Simple test runner script for OpenIMC

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}OpenIMC Test Runner${NC}"
echo "===================="
echo ""

# Check if pytest is installed
if ! python -m pytest --version > /dev/null 2>&1; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install with: pip install -r requirements.txt"
    exit 1
fi

# Parse command line arguments
TEST_TYPE="${1:-all}"
COVERAGE="${2:-false}"

case "$TEST_TYPE" in
    unit)
        echo -e "${YELLOW}Running unit tests...${NC}"
        if [ "$COVERAGE" = "true" ]; then
            python -m pytest tests/unit/ -v --cov=openimc --cov-report=term-missing
        else
            python -m pytest tests/unit/ -v
        fi
        ;;
    integration)
        echo -e "${YELLOW}Running integration tests...${NC}"
        python -m pytest tests/integration/ -v
        ;;
    all)
        echo -e "${YELLOW}Running all tests...${NC}"
        if [ "$COVERAGE" = "true" ]; then
            python -m pytest tests/ -v --cov=openimc --cov-report=term-missing --cov-report=html
            echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        else
            python -m pytest tests/ -v
        fi
        ;;
    fast)
        echo -e "${YELLOW}Running fast tests (excluding slow)...${NC}"
        python -m pytest tests/ -v -m "not slow"
        ;;
    *)
        echo "Usage: $0 [unit|integration|all|fast] [coverage]"
        echo ""
        echo "Examples:"
        echo "  $0 unit              # Run unit tests"
        echo "  $0 all true          # Run all tests with coverage"
        echo "  $0 fast              # Run fast tests only"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Tests completed!${NC}"

