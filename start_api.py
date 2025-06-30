#!/usr/bin/env python3
"""
Startup script for Video Image Generator API
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import openai
        import whisper
        import moviepy
        import numpy
        import requests
        from dotenv import load_dotenv
        print("✓ All Python dependencies are installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Check if FFmpeg is installed
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg is installed")
        else:
            print("✗ FFmpeg is not properly installed")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg is not installed")
        print("Please install FFmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        return False
    
    return True

def check_environment():
    """Check if environment variables are set"""
    print("Checking environment...")
    
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("✗ OPENAI_API_KEY environment variable is not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("  Or create a .env file with: OPENAI_API_KEY=your-api-key-here")
        return False
    
    print("✓ OpenAI API key is configured")
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    print("✓ Output directory created")
    
    return True

def start_server():
    """Start the FastAPI server"""
    print("\nStarting Video Image Generator API...")
    print("API will be available at:")
    print("  - Base URL: http://localhost:8000")
    print("  - Documentation: http://localhost:8000/docs")
    print("  - Alternative Docs: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        import uvicorn
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("Video Image Generator API Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Start server
    start_server()

if __name__ == "__main__":
    main() 