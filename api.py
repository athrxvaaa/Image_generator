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

    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio using OpenAI Whisper API"""
        try:
            print(f"Transcribing audio file: {audio_file_path}")
            
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            print("Transcription completed successfully")
            return transcript
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise

    def analyze_content(self, transcript: str) -> str:
        """Analyze transcript content using ChatGPT"""
        try:
            print("Analyzing transcript content...")
            
            prompt = f"""
            Analyze the following video transcript and provide a detailed description of the content, 
            including key themes, topics, people, actions, and visual elements mentioned. 
            Focus on elements that would be useful for generating relevant images.
            
            Transcript: {transcript}
            
            Please provide a comprehensive analysis that includes:
            1. Main topics and themes
            2. Key people or characters mentioned
            3. Actions and activities described
            4. Visual elements, settings, or scenes
            5. Emotions or moods conveyed
            6. Any specific objects, locations, or events
            
            Format your response as a detailed description suitable for image generation.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert content analyst who helps create detailed descriptions for image generation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content
            print("Content analysis completed")
            return analysis
            
        except Exception as e:
            print(f"Error analyzing content: {e}")
            raise

    def generate_images(self, description: str, num_images: int = 3) -> List[str]:
        """Generate images based on description using DALL-E"""
        try:
            print(f"Generating {num_images} images based on description...")
            
            # Create a detailed prompt for image generation
            image_prompt = f"""
            Create a high-quality, realistic image based on this description: {description}
            
            Requirements:
            - High resolution and professional quality
            - Realistic and detailed
            - Full-screen composition
            - Natural lighting and colors
            - Clear and focused subject matter
            
            Style: Photorealistic, professional photography
            """
            
            image_urls = []
            for i in range(num_images):
                print(f"Generating image {i+1}/{num_images}...")
                
                response = self.openai_client.images.generate(
                    model="dall-e-3",
                    prompt=image_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                
                image_url = response.data[0].url
                image_urls.append(image_url)
                print(f"Image {i+1} generated successfully")
            
            print(f"All {num_images} images generated successfully")
            return image_urls
            
        except Exception as e:
            print(f"Error generating images: {e}")
            raise

    def upload_to_s3(self, file_path: str, s3_key: str) -> str:
        """Upload file to S3 and return public URL"""
        try:
            print(f"Uploading {file_path} to S3 as {s3_key}...")
            
            self.s3_client.upload_file(
                file_path,
                self.s3_bucket,
                s3_key,
                ExtraArgs={'ACL': 'public-read'}
            )
            
            s3_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
            print(f"File uploaded successfully: {s3_url}")
            return s3_url
            
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            raise

# Initialize video processor lazily
processor = None

def get_processor():
    """Get or create video processor instance"""
    global processor
    if processor is None:
        try:
            processor = VideoProcessor()
        except Exception as e:
            print(f"Failed to initialize VideoProcessor: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize video processor")
    return processor

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
        "processor_ready": processor is not None,
        "memory_optimized": True
    }

@app.post("/process-video", response_model=ProcessingStatus)
async def process_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...)
):
    """Process uploaded video and generate relevant images"""
    try:
        processor = get_processor()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize processor: {str(e)}")

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Save the uploaded file immediately
    temp_video_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{video_file.filename}")
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)

    # Initialize task status
    task_status[task_id] = ProcessingStatus(
        task_id=task_id,
        status="uploading",
        message="Video uploaded, starting processing..."
    )

    # Add background task with the saved file path
    background_tasks.add_task(process_video_background, task_id, temp_video_path)

    return task_status[task_id]

@app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_status(task_id: str):
    """Get processing status for a task"""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_status[task_id]

async def process_video_background(task_id: str, temp_video_path: str):
    """Background task to process video"""
    try:
        task_status[task_id].status = "processing"
        task_status[task_id].message = "Processing video..."
        processor = get_processor()

        # Step 1: Extract audio from video
        task_status[task_id].message = "Extracting audio from video..."
        temp_audio_path = os.path.join(processor.temp_dir, f"{task_id}_audio.mp3")
        if MOVIEPY_AVAILABLE and VideoFileClip:
            try:
                video = VideoFileClip(temp_video_path)
                video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                video.close()
            except Exception as e:
                print(f"MoviePy audio extraction failed: {e}")
                os.system(f'ffmpeg -i "{temp_video_path}" -vn -acodec mp3 "{temp_audio_path}" -y')
        else:
            os.system(f'ffmpeg -i "{temp_video_path}" -vn -acodec mp3 "{temp_audio_path}" -y')

        # Step 2: Transcribe audio using Whisper API
        task_status[task_id].message = "Transcribing audio..."
        transcript = processor.transcribe_audio(temp_audio_path)

        # Step 3: Analyze content using ChatGPT
        task_status[task_id].message = "Analyzing content..."
        analysis = processor.analyze_content(transcript)

        # Step 4: Generate images using DALL-E
        task_status[task_id].message = "Generating images..."
        image_urls = processor.generate_images(analysis, num_images=3)

        # Step 5: Create results file
        task_status[task_id].message = "Creating results file..."
        results = {
            "task_id": task_id,
            "transcript": transcript,
            "analysis": analysis,
            "image_urls": image_urls,
            "timestamp": datetime.now().isoformat()
        }
        results_file_path = os.path.join(processor.output_dir, f"{task_id}_results.json")
        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=2)

        # Step 6: Upload results to S3
        task_status[task_id].message = "Uploading results to S3..."
        s3_key = f"results/{task_id}_results.json"
        s3_url = processor.upload_to_s3(results_file_path, s3_key)

        # Clean up temporary files
        try:
            os.remove(temp_video_path)
            os.remove(temp_audio_path)
        except:
            pass

        # Update status to completed
        task_status[task_id].status = "completed"
        task_status[task_id].message = "Video processing completed successfully"
        task_status[task_id].s3_url = s3_url

    except Exception as e:
        task_status[task_id].status = "error"
        task_status[task_id].message = f"Processing failed: {str(e)}"
        print(f"Error in background processing: {e}")

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint for port detection"""
    return {
        "message": "API is running",
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 