# Video Image Generator API

A FastAPI-based service that processes videos by extracting audio, transcribing it with Whisper, analyzing content with GPT-4o-mini, generating contextual images with GPT-Image-1, and creating enhanced videos with inserted images.

## Features

- **Audio Extraction**: Extracts audio from uploaded videos
- **Speech Transcription**: Uses OpenAI Whisper for accurate transcription
- **Content Analysis**: GPT-4o-mini analyzes content and generates topic-specific image suggestions
- **Image Generation**: GPT-Image-1 creates realistic, contextual images
- **Video Enhancement**: Inserts generated images into videos at strategic intervals
- **AWS S3 Integration**: Stores processed videos in S3
- **RESTful API**: Clean endpoints for upload, status checking, and download

## Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg installed on your system
- OpenAI API key
- AWS S3 credentials (optional, for cloud storage)

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

### Upload Video

```http
POST /upload-video
Content-Type: multipart/form-data

Parameters:
- file: Video file (MP4, AVI, MOV, etc.)
- generate_images: boolean (default: true)
- image_interval: float (optional, seconds between images)
- max_images: integer (optional, maximum number of images)
```

### Check Status

```http
GET /status/{task_id}
```

### Download Video

```http
GET /download/{task_id}
```

### Health Check

```http
GET /health
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

### Production Deployment

```bash
# Using uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000

# Using gunicorn (recommended for production)
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Project Structure

```
Image_generator/
├── api.py                 # Main FastAPI application
├── start_api.py          # Server startup script
├── requirements.txt      # Python dependencies
├── env_example.txt       # Environment variables template
├── README.md            # This file
├── SETUP_GUIDE.md       # Detailed setup instructions
├── API_README.md        # API documentation
└── output/              # Processed video storage
```

## Documentation

- **API Documentation**: Available at `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: Available at `http://localhost:8000/redoc`
- **Postman Collection**: `Video_Image_Generator_API.postman_collection.json`

## License

This project is licensed under the MIT License.
