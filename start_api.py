#!/usr/bin/env python3
"""
Startup script for Video Image Generator API
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def quick_startup():
    """Quick startup with minimal checks"""
    print("Starting Video Image Generator API...")
    
    # Load environment variables
    load_dotenv()
    
    # Get port from environment variable
    port = int(os.getenv('PORT', 8000))
    print(f"Using port: {port}")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Start server immediately
    print("Starting uvicorn server...")
    try:
        import uvicorn
        uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False, workers=1, log_level="info")
    except Exception as e:
        print(f"Failed to start uvicorn: {e}")
        sys.exit(1)

if __name__ == "__main__":
    quick_startup() 