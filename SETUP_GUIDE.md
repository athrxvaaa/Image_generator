# Quick Setup Guide

This guide will help you set up and run the Video Image Generator API in minutes.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **FFmpeg** installed (for video processing)
3. **OpenAI API key** (for AI services)

## Step 1: Install FFmpeg

### macOS

```bash
brew install ffmpeg
```

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install ffmpeg
```

### Windows

1. Download from https://ffmpeg.org/download.html
2. Add to your system PATH

## Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Set Up OpenAI API Key

### Option A: Environment Variable

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Option B: .env File

1. Copy `env_example.txt` to `.env`
2. Replace `your-openai-api-key-here` with your actual API key

## Step 4: Start the API

### Easy Way (Recommended)

```bash
python start_api.py
```

### Manual Way

```bash
python api.py
```

## Step 5: Test the API

The API will be available at:

- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Testing with Postman

1. **Import Collection**: Import `Video_Image_Generator_API.postman_collection.json`
2. **Set Environment**: Create environment with `base_url: http://localhost:8000`
3. **Test Endpoints**:
   - Health Check: Verify API is running
   - Upload Video: Upload a video file
   - Check Status: Monitor processing progress
   - Download Video: Get the processed video

## Testing with Python

```bash
python example_api_usage.py
```

## Troubleshooting

### Common Issues

1. **"FFmpeg not found"**

   - Install FFmpeg using the commands above
   - Ensure it's in your system PATH

2. **"OpenAI API key not set"**

   - Set the environment variable or create .env file
   - Get your API key from https://platform.openai.com/api-keys

3. **"Port 8000 already in use"**

   - Change the port in `api.py` or `start_api.py`
   - Or stop the process using port 8000

4. **"Missing dependencies"**
   - Run `pip install -r requirements.txt`
   - Ensure you're using Python 3.8+

### Getting Help

- Check the console output for detailed error messages
- Visit http://localhost:8000/docs for API documentation
- Review the full README.md for detailed information

## Next Steps

Once the API is running:

1. **Upload a video** using the `/upload-video` endpoint
2. **Monitor progress** using the `/status/{task_id}` endpoint
3. **Download results** using the `/download/{task_id}` endpoint
4. **Customize settings** by modifying the API parameters

## API Features

- ✅ Video upload and processing
- ✅ Audio transcription with Whisper
- ✅ Content analysis with ChatGPT
- ✅ Dynamic image generation with DALL-E
- ✅ Background processing with progress tracking
- ✅ Download processed videos
- ✅ Postman collection included
- ✅ Interactive API documentation
- ✅ Error handling and validation
