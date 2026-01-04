#!/bin/bash
# Install system dependencies
apt-get update
apt-get install -y tesseract-ocr libtesseract-dev ffmpeg libsm6 libxext6
# Install Python dependencies
pip install -r requirements.txt