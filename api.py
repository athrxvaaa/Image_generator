#!/usr/bin/env python3
"""
FastAPI application for video processing with AI-powered image generation
"""

import os
import json
import tempfile
import shutil
import asyncio
from typing import List, Dict, Optional
from pathlib import Path
import time
import hashlib
from datetime import datetime

import numpy as np
# Optional moviepy imports - will be None if not available
try:
    from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    VideoFileClip = None
    ImageClip = None
    CompositeVideoClip = None
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy.editor not available. Video processing features will be limited.")

import whisper
from dotenv import load_dotenv
import openai
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import uuid
import boto3
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Video Image Generator API",
    description="API for processing videos and generating relevant images using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    message: str
    progress: Optional[float] = None
    download_url: Optional[str] = None
    s3_url: Optional[str] = None

# Global storage for task status
task_status = {}

class VideoProcessor:
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the VideoProcessor with OpenAI API key
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client for v1.x API
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Initialize Whisper model
        self.whisper_model = None
        self._load_whisper_model()
        
        # Temporary directory for processing
        self.temp_dir = tempfile.gettempdir()
        
        # Create output directory
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize AWS S3 client
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'ap-south-1')
        s3_bucket = os.getenv('S3_BUCKET_NAME')

        if not aws_access_key_id or not aws_secret_access_key or not s3_bucket:
            raise ValueError("AWS credentials and S3 bucket name are required. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET_NAME environment variables.")

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        self.s3_bucket = s3_bucket

    def _load_whisper_model(self):
        """Load Whisper model for transcription"""
        try:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise

# Initialize video processor
try:
    processor = VideoProcessor()
except Exception as e:
    print(f"Failed to initialize VideoProcessor: {e}")
    processor = None

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Image Generator API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "moviepy_available": MOVIEPY_AVAILABLE,
        "processor_ready": processor is not None
    }

@app.post("/process-video", response_model=ProcessingStatus)
async def process_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...)
):
    """Process uploaded video and generate relevant images"""
    
    if not processor:
        raise HTTPException(status_code=500, detail="Video processor not initialized")
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    task_status[task_id] = ProcessingStatus(
        task_id=task_id,
        status="uploading",
        message="Video uploaded, starting processing..."
    )
    
    # Add background task
    background_tasks.add_task(process_video_background, task_id, video_file)
    
    return task_status[task_id]

@app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_status(task_id: str):
    """Get processing status for a task"""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_status[task_id]

async def process_video_background(task_id: str, video_file: UploadFile):
    """Background task to process video"""
    try:
        # Update status
        task_status[task_id].status = "processing"
        task_status[task_id].message = "Processing video..."
        
        # Save uploaded file
        temp_video_path = os.path.join(processor.temp_dir, f"{task_id}_{video_file.filename}")
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        # TODO: Implement actual video processing logic here
        # For now, just simulate processing
        await asyncio.sleep(2)
        
        # Update status to completed
        task_status[task_id].status = "completed"
        task_status[task_id].message = "Video processing completed"
        task_status[task_id].download_url = f"/download/{task_id}"
        
    except Exception as e:
        task_status[task_id].status = "error"
        task_status[task_id].message = f"Processing failed: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 