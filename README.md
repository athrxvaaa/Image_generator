# Video Image Generator API

A FastAPI-based service that processes videos by extracting audio, transcribing it with OpenAI Whisper API, analyzing content with ChatGPT, and generating contextual images with DALL-E 3.

## 🚀 Live Demo

**API is now live and deployed on Render:**

- **Live URL**: https://image-generator-6od6.onrender.com
- **API Documentation**: https://image-generator-6od6.onrender.com/docs
- **Health Check**: https://image-generator-6od6.onrender.com/health

## Features

- **Audio Extraction**: Extracts audio from uploaded videos using FFmpeg
- **Speech Transcription**: Uses OpenAI Whisper API for accurate transcription
- **Content Analysis**: ChatGPT analyzes content and generates detailed descriptions
- **Image Generation**: DALL-E 3 creates high-quality, contextual images
- **AWS S3 Integration**: Stores processing results in S3
- **Background Processing**: Asynchronous video processing with status tracking
- **RESTful API**: Clean endpoints for upload, status checking, and results

## Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg installed on your system
- OpenAI API key
- AWS S3 credentials (for cloud storage)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Image_generator
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   ```bash
   cp env_example.txt .env
   # Edit .env with your API keys
   ```

4. **Start the API server**
   ```bash
   python start_api.py
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

### Process Video

```http
POST /process-video
Content-Type: multipart/form-data

Parameters:
- video_file: Video file (MP4, AVI, MOV, etc.)
```

### Check Status

```http
GET /status/{task_id}
```

### Health Check

```http
GET /health
```

### Test Endpoint

```http
GET /test
```

### Root Endpoint

```http
GET /
```

## Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-south-1
S3_BUCKET_NAME=your_s3_bucket_name
```

## Deployment

### Local Development

```bash
python start_api.py
```

### Production Deployment (Docker)

```bash
# Build and run with Docker
docker build -t video-image-generator .
docker run -p 8000:8000 video-image-generator
```

### Render Deployment

The API is configured for automatic deployment on Render:

- **Runtime**: Docker
- **Port**: 8000
- **Health Check**: `/health` endpoint
- **Memory Optimized**: Uses OpenAI APIs instead of local models

## Processing Pipeline

1. **Video Upload**: Accept video file via multipart form data
2. **Audio Extraction**: Extract audio using FFmpeg
3. **Transcription**: Use OpenAI Whisper API for speech-to-text
4. **Content Analysis**: ChatGPT analyzes transcript and creates detailed description
5. **Image Generation**: DALL-E 3 generates 3 contextual images
6. **Results Storage**: Save results to AWS S3
7. **Status Tracking**: Real-time status updates via task ID

## Project Structure

```
Image_generator/
├── api.py                 # Main FastAPI application
├── start_api.py          # Server startup script
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── render.yaml          # Render deployment config
├── env_example.txt       # Environment variables template
├── README.md            # This file
├── SETUP_GUIDE.md       # Detailed setup instructions
├── API_README.md        # API documentation
├── DEPLOYMENT.md        # Deployment guide
├── RENDER_DEPLOYMENT.md # Render-specific deployment
└── output/              # Processed results storage
```

## Technical Details

- **Framework**: FastAPI with Uvicorn
- **AI Services**: OpenAI Whisper API, ChatGPT, DALL-E 3
- **Storage**: AWS S3
- **Processing**: Asynchronous background tasks
- **Memory Optimized**: No local AI models (uses APIs)
- **Containerized**: Docker support for easy deployment

## Documentation

- **API Documentation**: Available at `/docs` (Swagger UI)
- **Alternative Docs**: Available at `/redoc`
- **Postman Collection**: `Video_Image_Generator_API.postman_collection.json`

## Recent Updates

- ✅ **Live Deployment**: Successfully deployed on Render
- ✅ **Memory Optimization**: Replaced local Whisper model with API
- ✅ **Port Binding**: Fixed deployment port detection issues
- ✅ **Startup Optimization**: Faster application startup
- ✅ **Error Handling**: Improved error handling and fallbacks

## License

This project is licensed under the MIT License.
