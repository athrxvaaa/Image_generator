#!/usr/bin/env python3
"""
Startup script for Video Image Generator API
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def test_imports():
    """Test all required imports before starting the application"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import moviepy
        print(f"✓ moviepy imported successfully (version: {moviepy.__version__})")
    except ImportError as e:
        print(f"✗ moviepy import failed: {e}")
        return False
    
    try:
        from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
        print("✓ moviepy.editor imports successful")
    except ImportError as e:
        print(f"✗ moviepy.editor import failed: {e}")
        return False
    
    try:
        import whisper
        print("✓ whisper imported successfully")
    except ImportError as e:
        print(f"✗ whisper import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv imported successfully")
    except ImportError as e:
        print(f"✗ python-dotenv import failed: {e}")
        return False
    
    try:
        import openai
        print("✓ openai imported successfully")
    except ImportError as e:
        print(f"✗ openai import failed: {e}")
        return False
    
    try:
        from fastapi import FastAPI
        print("✓ fastapi imported successfully")
    except ImportError as e:
        print(f"✗ fastapi import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✓ uvicorn imported successfully")
    except ImportError as e:
        print(f"✗ uvicorn import failed: {e}")
        return False
    
    try:
        from pydantic import BaseModel
        print("✓ pydantic imported successfully")
    except ImportError as e:
        print(f"✗ pydantic import failed: {e}")
        return False
    
    try:
        import boto3
        print("✓ boto3 imported successfully")
    except ImportError as e:
        print(f"✗ boto3 import failed: {e}")
        return False
    
    print("All imports successful!")
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
    """Main startup function"""
    print("Starting Video Image Generator API...")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")
    
    # Test imports first
    if not test_imports():
        print("Import test failed. Exiting.")
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