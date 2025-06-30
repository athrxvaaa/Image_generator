# Deployment Guide

This guide covers different ways to deploy the Video Image Generator API.

## Prerequisites

- Python 3.8+ or Docker
- FFmpeg installed (for non-Docker deployments)
- OpenAI API key
- AWS S3 credentials (optional, for cloud storage)

## Deployment Options

### 1. Local Development

```bash
# Clone the repository
git clone <repository-url>
cd Image_generator

# Set up environment
cp env_example.txt .env
# Edit .env with your API keys

# Install dependencies
pip install -r requirements.txt

# Start the server
python start_api.py
```

### 2. Production Server (Linux/macOS)

```bash
# Use the deployment script
chmod +x deploy.sh
./deploy.sh

# Or manually
pip install gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300
```

### 3. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t video-image-generator .
docker run -p 8000:8000 --env-file .env video-image-generator
```

### 4. Cloud Deployment

#### Heroku

1. Create a `Procfile`:

   ```
   web: gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 300
   ```

2. Deploy:
   ```bash
   heroku create your-app-name
   heroku config:set OPENAI_API_KEY=your_key
   git push heroku main
   ```

#### AWS EC2

1. Launch an EC2 instance with Ubuntu
2. Install dependencies:

   ```bash
   sudo apt update
   sudo apt install python3-pip ffmpeg
   ```

3. Clone and deploy:
   ```bash
   git clone <repository-url>
   cd Image_generator
   ./deploy.sh
   ```

#### Google Cloud Run

1. Build and deploy:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/video-image-generator
   gcloud run deploy --image gcr.io/PROJECT_ID/video-image-generator --platform managed
   ```

## Environment Variables

Required:

- `OPENAI_API_KEY`: Your OpenAI API key

Optional (for S3 storage):

- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region (default: ap-south-1)
- `S3_BUCKET_NAME`: S3 bucket name

## Health Check

The API includes a health check endpoint:

```bash
curl http://localhost:8000/health
```

## Monitoring

For production deployments, consider:

- Logging (structured logs)
- Metrics (response times, error rates)
- Alerting (API failures, high latency)
- Rate limiting
- Load balancing

## Security Considerations

- Use HTTPS in production
- Implement API key authentication
- Set up CORS properly
- Rate limit API endpoints
- Validate file uploads
- Monitor for abuse

## Scaling

For high traffic:

- Use multiple worker processes
- Implement Redis for caching
- Use a CDN for static assets
- Consider async processing for video uploads
- Implement queue systems (Celery/RQ)
