#!/bin/bash

# Install Quantra Python Dependencies
# This script installs all required Python packages for Quantra

echo "==============================================================================="
echo "Quantra Python Environment Setup"
echo "==============================================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or later"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is not installed or not in PATH"
    echo "Please install pip or reinstall Python"
    exit 1
fi

echo ""
echo "Installing core dependencies..."
echo ""

# Upgrade pip first
python3 -m pip install --upgrade pip

# Install requirements
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Some packages failed to install"
    echo "This may be normal for optional dependencies"
    echo ""
fi

echo ""
echo "==============================================================================="
echo "Checking installation..."
echo "==============================================================================="
echo ""

# Run dependency checker
python3 check_dependencies.py

echo ""
echo "==============================================================================="
echo "Setup complete!"
echo "==============================================================================="
echo ""
