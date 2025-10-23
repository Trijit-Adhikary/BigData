#!/bin/bash

# Minimal Playwright Init Script for Databricks
# Save this as: /dbfs/init-scripts/playwright-minimal.sh

echo "Installing Playwright (minimal setup)..."

# Update and install minimal dependencies
sudo apt-get update -y
sudo apt-get install -y wget ca-certificates

# Install Playwright
pip install playwright

# Install only Chromium (most commonly used)
python -m playwright install chromium
python -m playwright install-deps chromium

# Install xvfb for headless display
sudo apt-get install -y xvfb

echo "Minimal Playwright setup complete"
