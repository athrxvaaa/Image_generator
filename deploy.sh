#!/bin/bash

# Video Image Generator API Deployment Script

echo "🚀 Starting Video Image Generator API Deployment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg is not installed. Please install FFmpeg first."
    echo "   macOS: brew install ffmpeg"
    echo "   Ubuntu: sudo apt install ffmpeg"
    echo "   Windows: Download from https://ffmpeg.org/download.html"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp env_example.txt .env
    echo "📝 Please edit .env file with your API keys before continuing."
    echo "   Required: OPENAI_API_KEY"
    echo "   Optional: AWS credentials for S3 storage"
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create output directory if it doesn't exist
mkdir -p output

# Check if gunicorn is available for production
if command -v gunicorn &> /dev/null; then
    echo "🚀 Starting production server with gunicorn..."
    gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300
else
    echo "🚀 Starting development server with uvicorn..."
    echo "   Install gunicorn for production: pip install gunicorn"
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
fi 