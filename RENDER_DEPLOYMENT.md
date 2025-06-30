# Deploy to Render

This guide will help you deploy the Video Image Generator API to Render.

## Prerequisites

- A Render account (free tier available)
- OpenAI API key
- AWS S3 credentials (optional, for cloud storage)

## Step 1: Prepare Your Repository

1. **Push your code to GitHub/GitLab**

   ```bash
   git add .
   git commit -m "Ready for Render deployment"
   git push origin main
   ```

2. **Ensure these files are in your repository:**
   - `api.py` (main FastAPI app)
   - `requirements.txt` (with gunicorn included)
   - `render.yaml` (deployment config)
   - `README.md`

## Step 2: Deploy on Render

### Option A: Using render.yaml (Recommended)

1. **Connect your repository:**

   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Blueprint"
   - Connect your GitHub/GitLab repository
   - Select the repository with your code

2. **Render will automatically detect the render.yaml file**
   - It will create the service with the configuration
   - You'll be prompted to set environment variables

### Option B: Manual Setup

1. **Create a new Web Service:**

   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Web Service"
   - Connect your repository

2. **Configure the service:**
   - **Name**: `video-image-generator-api`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 300`

## Step 3: Set Environment Variables

In your Render service settings, add these environment variables:

### Required:

- `OPENAI_API_KEY`: Your OpenAI API key

### Optional (for S3 storage):

- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_REGION`: `ap-south-1` (or your preferred region)
- `S3_BUCKET_NAME`: Your S3 bucket name

## Step 4: Deploy

1. **Click "Create Web Service"**
2. **Wait for the build to complete** (usually 2-5 minutes)
3. **Your API will be available at**: `https://your-app-name.onrender.com`

## Step 5: Test Your Deployment

1. **Health Check:**

   ```bash
   curl https://your-app-name.onrender.com/health
   ```

2. **API Documentation:**
   - Visit: `https://your-app-name.onrender.com/docs`
   - Test the endpoints directly from the Swagger UI

## Important Notes

### Free Tier Limitations

- **Sleep after inactivity**: Free services sleep after 15 minutes of inactivity
- **Build time**: 500 minutes per month
- **Bandwidth**: 100 GB per month
- **Storage**: 1 GB

### Production Considerations

- **Upgrade to paid plan** for always-on service
- **Use environment variables** for all sensitive data
- **Monitor logs** in Render dashboard
- **Set up custom domain** if needed

### Troubleshooting

1. **Build fails:**

   - Check that `requirements.txt` includes `gunicorn`
   - Ensure all dependencies are compatible
   - Check build logs in Render dashboard

2. **Service won't start:**

   - Verify the start command is correct
   - Check environment variables are set
   - Review logs for error messages

3. **API not responding:**
   - Free tier services sleep after inactivity
   - First request may take 30-60 seconds to wake up
   - Consider upgrading to paid plan for always-on service

## Environment Variables Reference

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key-here

# Optional (for S3)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=ap-south-1
S3_BUCKET_NAME=your-s3-bucket-name
```

## Support

- **Render Documentation**: [https://render.com/docs](https://render.com/docs)
- **Render Community**: [https://community.render.com](https://community.render.com)
- **Project Issues**: Open an issue on your repository

Your API will be live and ready to process videos once deployed!
