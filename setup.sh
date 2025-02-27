#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Download spacy model
python3 -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p uploads
mkdir -p templates
mkdir -p data

echo "Setup completed successfully!" 