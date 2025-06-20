#!/usr/bin/env python3
"""
Setup script for Script Trimmer
Automatically installs dependencies and downloads Vosk model
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is already installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ FFmpeg not found")
    print("\nPlease install FFmpeg:")
    print("  macOS: brew install ffmpeg")
    print("  Ubuntu/Debian: sudo apt install ffmpeg")
    print("  Windows: Download from https://ffmpeg.org/download.html")
    return False

def download_vosk_model():
    """Download Vosk model if not present"""
    model_name = "vosk-model-small-en-us-0.15"
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    
    # Check if model already exists
    if os.path.exists(model_name):
        print(f"✅ Vosk model '{model_name}' already exists")
        return True
    
    # Check if model exists in models directory
    models_dir = Path("models")
    if models_dir.exists() and (models_dir / model_name).exists():
        print(f"✅ Vosk model '{model_name}' found in models directory")
        return True
    
    print(f"🔄 Downloading Vosk model '{model_name}'...")
    print("This may take a few minutes depending on your internet connection...")
    
    try:
        # Create models directory if it doesn't exist
        models_dir.mkdir(exist_ok=True)
        
        # Download the model
        zip_path = models_dir / f"{model_name}.zip"
        urllib.request.urlretrieve(model_url, zip_path)
        
        # Extract the model
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        
        # Remove the zip file
        zip_path.unlink()
        
        print(f"✅ Vosk model downloaded and extracted to models/{model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download Vosk model: {e}")
        print("\nPlease download manually from:")
        print("https://alphacephei.com/vosk/models")
        print("Extract to the project directory or models/ folder")
        return False

def create_env_template():
    """Create .env template file"""
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("# Pexels API Key (optional)\n")
            f.write("# Get your free API key from: https://www.pexels.com/api/\n")
            f.write("PEXELS_API_KEY=your_api_key_here\n")
        print("✅ Created .env template file")
        print("📝 Edit .env to add your Pexels API key (optional)")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup function"""
    print("🚀 Setting up Script Trimmer...\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Check FFmpeg
    if not check_ffmpeg():
        print("\n⚠️  Please install FFmpeg manually and run setup again")
        print("   Setup will continue but video processing won't work without FFmpeg")
    
    # Download Vosk model
    if not download_vosk_model():
        print("\n⚠️  Please download Vosk model manually")
        print("   Setup will continue but transcription won't work without the model")
    
    # Create .env template
    create_env_template()
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Install FFmpeg if not already installed")
    print("2. Download Vosk model if not already downloaded")
    print("3. (Optional) Get Pexels API key and add to .env file")
    print("4. Run: python script_trimmer.py your_video.mp4")
    
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main() 