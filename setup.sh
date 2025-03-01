#!/bin/bash

# Install system dependencies
if [ -f /etc/debian_version ]; then
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils
elif [ -f /etc/redhat-release ]; then
    sudo yum install -y tesseract tesseract-devel poppler-utils
elif [ -f /etc/arch-release ]; then
    sudo pacman -S tesseract tesseract-data-eng poppler
else
    echo "Please install tesseract-ocr and poppler-utils manually for your system"
fi

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install torch first
pip install torch>=2.0.0

# Install huggingface-hub with specific version
pip install huggingface-hub==0.16.4

# Install transformers and sentence-transformers
pip install transformers==4.30.2
pip install sentence-transformers==2.2.2

# Install requirements
pip install -r requirements.txt

# Download spacy model
python3 -m spacy download en_core_web_sm

# Create data directory and ensure it exists
mkdir -p data
mkdir -p uploads
mkdir -p templates

echo "Setup completed successfully!" 