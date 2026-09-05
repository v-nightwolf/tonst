#!/usr/bin/env bash
# One-command setup: installs dependencies, then runs the demo.
# Usage: ./setup.sh
set -e

echo "Installing dependencies (requests, numpy)..."
python3 -m pip install -r requirements.txt

echo ""
echo "Running demo..."
python3 demo.py
