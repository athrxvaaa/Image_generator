#!/usr/bin/env python3
"""
FastAPI application for video processing with AI-powered image generation
"""

import os
import json
import tempfile
import shutil
from typing import List, Dict, Optional
from pathlib import Path
import time
import hashlib
from datetime import datetime

import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
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

    # ... rest of the code remains unchanged ... 