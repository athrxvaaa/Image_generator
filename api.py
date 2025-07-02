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
import base64
from PIL import Image

import numpy as np
import subprocess
from dotenv import load_dotenv
import openai
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
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
    s3_presigned_url: Optional[str] = None

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

    def analyze_content(self, transcript: str) -> list:
        """Analyze transcript content and divide into segments with keywords"""
        try:
            print("Analyzing transcript content and dividing into segments...")
            
            # First, divide transcript into sentences/segments
            sentences = self._divide_transcript(transcript)
            
            segments_with_keywords = []
            
            for i, sentence in enumerate(sentences):
                prompt = f"""
                Analyze this sentence from a video transcript and extract key visual elements for image generation:
                
                Sentence: {sentence}
                
                Please provide:
                1. A concise visual description (2-3 sentences) focused on the main visual elements
                2. Key keywords that represent the most important visual aspects
                3. The mood or atmosphere of this scene
                
                Format your response as JSON:
                {{
                    "description": "visual description here",
                    "keywords": ["keyword1", "keyword2", "keyword3"],
                    "mood": "mood description",
                    "sentence": "original sentence"
                }}
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert content analyst who extracts visual elements for image generation. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                
                try:
                    import json
                    analysis = json.loads(response.choices[0].message.content)
                    analysis['segment_id'] = i + 1
                    segments_with_keywords.append(analysis)
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    segments_with_keywords.append({
                        "description": sentence,
                        "keywords": sentence.split()[:5],  # First 5 words as keywords
                        "mood": "neutral",
                        "sentence": sentence,
                        "segment_id": i + 1
                    })
            
            print(f"Content analysis completed - {len(segments_with_keywords)} segments created")
            return segments_with_keywords
            
        except Exception as e:
            print(f"Error analyzing content: {e}")
            raise

    def _divide_transcript(self, transcript: str) -> list:
        """Divide transcript into meaningful segments"""
        try:
            # Split by sentences (periods, exclamation marks, question marks)
            import re
            sentences = re.split(r'[.!?]+', transcript)
            
            # Clean up sentences and filter out empty ones
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # If we have too many sentences, group them into segments
            if len(sentences) > 10:
                # Group sentences into segments of 2-3 sentences each
                segments = []
                for i in range(0, len(sentences), 2):
                    segment = ' '.join(sentences[i:i+2])
                    segments.append(segment)
                return segments
            
            return sentences
            
        except Exception as e:
            print(f"Error dividing transcript: {e}")
            # Fallback: return transcript as single segment
            return [transcript]

    def generate_images(self, segments: list, num_images_per_segment: int = 1, task_id: str = None) -> dict:
        """Generate images based on each segment's description and keywords"""
        all_images = {}
        try:
            print(f"Generating images for {len(segments)} segments...")
            
            for segment in segments:
                segment_id = segment['segment_id']
                description = segment['description']
                keywords = segment['keywords']
                mood = segment['mood']
                
                print(f"Generating image for segment {segment_id}: {description[:50]}...")
                
                # Create specific prompt for this segment
                image_prompt = f"""
                Create a high-quality, realistic image based on this specific scene: {description}
                
                Key visual elements: {', '.join(keywords)}
                Mood/atmosphere: {mood}
                
                Requirements:
                - High resolution and professional quality
                - Realistic and detailed
                - Focus on the specific visual elements mentioned
                - Match the described mood and atmosphere
                - Clear and focused subject matter
                Style: Photorealistic, professional photography
                """
                
                segment_images = []
                for i in range(num_images_per_segment):
                    try:
                        response = self.openai_client.images.generate(
                            model="gpt-image-1",
                            prompt=image_prompt,
                            size="1024x1536",
                            quality="medium",
                            n=1
                        )
                        
                        image_url = None
                        if hasattr(response, "data") and response.data:
                            data = response.data[0]
                            if hasattr(data, "url") and data.url:
                                image_url = data.url
                            elif hasattr(data, "b64_json") and data.b64_json:
                                # Save base64 image to PNG
                                img_data = base64.b64decode(data.b64_json)
                                img_path = os.path.join(self.output_dir, f"{task_id}_segment_{segment_id}_image_{i+1}.png")
                                with open(img_path, "wb") as img_file:
                                    img_file.write(img_data)
                                # Upload to S3
                                s3_key = f"results/{task_id}_segment_{segment_id}_image_{i+1}.png"
                                image_url = self.upload_to_s3(img_path, s3_key)
                            else:
                                image_url = f"ERROR: No image data returned for segment {segment_id}"
                        else:
                            image_url = f"ERROR: No image data returned for segment {segment_id}"
                        
                        segment_images.append({
                            "url": image_url,
                            "description": description,
                            "keywords": keywords,
                            "mood": mood
                        })
                        
                    except Exception as e:
                        print(f"Error generating image for segment {segment_id}: {e}")
                        segment_images.append({
                            "url": f"ERROR: {str(e)}",
                            "description": description,
                            "keywords": keywords,
                            "mood": mood
                        })
                
                all_images[f"segment_{segment_id}"] = {
                    "images": segment_images,
                    "segment_data": segment
                }
            
            print(f"All images generated for {len(segments)} segments")
            return all_images
            
        except Exception as e:
            print(f"Error in generate_images: {e}")
            return {"error": str(e)}

    def _fallback_analyze_content(self, transcript: str) -> str:
        """Fallback method for content analysis (original approach)"""
        try:
            print("Using fallback content analysis...")
            
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
            print("Fallback content analysis completed")
            return analysis
            
        except Exception as e:
            print(f"Error in fallback content analysis: {e}")
            return transcript

    def _fallback_generate_images(self, description: str, num_images: int = 3, task_id: str = None) -> dict:
        """Fallback method for image generation (original approach)"""
        try:
            print(f"Using fallback image generation for {num_images} images...")
            
            image_prompt = f"""
            Create a high-quality, realistic portrait image based on this description: {description}
            Requirements:
            - High resolution and professional quality
            - Realistic and detailed
            - Full-screen composition
            - Natural lighting and colors
            - Clear and focused subject matter
            Style: Photorealistic, professional photography
            Orientation: Portrait
            """
            
            segment_images = []
            for i in range(num_images):
                print(f"Generating fallback image {i+1}/{num_images}...")
                try:
                    response = self.openai_client.images.generate(
                        model="gpt-image-1",
                        prompt=image_prompt,
                        size="1024x1536",
                        quality="medium",
                        n=1
                    )
                    
                    image_url = None
                    if hasattr(response, "data") and response.data:
                        data = response.data[0]
                        if hasattr(data, "url") and data.url:
                            image_url = data.url
                        elif hasattr(data, "b64_json") and data.b64_json:
                            # Save base64 image to PNG
                            img_data = base64.b64decode(data.b64_json)
                            img_path = os.path.join(self.output_dir, f"{task_id}_fallback_image_{i+1}.png")
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_data)
                            # Upload to S3
                            s3_key = f"results/{task_id}_fallback_image_{i+1}.png"
                            image_url = self.upload_to_s3(img_path, s3_key)
                        else:
                            image_url = f"ERROR: No image data returned for fallback image {i+1}"
                    else:
                        image_url = f"ERROR: No image data returned for fallback image {i+1}"
                    
                    segment_images.append({
                        "url": image_url,
                        "description": description,
                        "keywords": [],
                        "mood": "neutral"
                    })
                    
                except Exception as e:
                    print(f"Error generating fallback image {i+1}: {e}")
                    segment_images.append({
                        "url": f"ERROR: {str(e)}",
                        "description": description,
                        "keywords": [],
                        "mood": "neutral"
                    })
            
            return {
                "segment_1": {
                    "images": segment_images,
                    "segment_data": {
                        "segment_id": 1,
                        "description": description,
                        "keywords": [],
                        "mood": "neutral",
                        "sentence": description
                    }
                }
            }
            
        except Exception as e:
            print(f"Error in fallback image generation: {e}")
            return {"error": str(e)}

    def upload_to_s3(self, file_path: str, s3_key: str) -> str:
        """Upload file to S3 and return public URL"""
        try:
            print(f"Uploading {file_path} to S3 as {s3_key}...")
            
            self.s3_client.upload_file(
                file_path,
                self.s3_bucket,
                s3_key
            )
            
            s3_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
            print(f"File uploaded successfully: {s3_url}")
            return s3_url
            
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            raise

    def extract_audio_ffmpeg(self, video_path: str, audio_path: str):
        """Extract audio from video using FFmpeg."""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'mp3', audio_path
            ]
            print(f"Running FFmpeg for audio extraction: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"FFmpeg audio extraction failed: {e}")
            raise

    def create_video_with_images(self, video_path: str, image_paths: list, output_path: str) -> str:
        """Insert images at intervals into the video using FFmpeg and save as a new video file. Convert overlays to JPEG and optimize FFmpeg for low memory."""
        try:
            duration = self.get_video_duration(video_path)
            n_images = len(image_paths)
            interval = duration / (n_images + 1)
            filter_complex = []
            input_args = ['-i', video_path]
            jpeg_image_paths = []
            for idx, img_path in enumerate(image_paths):
                # Convert PNG to JPEG for lower memory usage
                jpg_path = img_path.replace('.png', '.jpg')
                if not os.path.exists(jpg_path):
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    img.save(jpg_path, 'JPEG', quality=90)
                jpeg_image_paths.append(jpg_path)
                input_args += ['-i', jpg_path]
            overlay_stream = '[0:v]'
            for idx, jpg_path in enumerate(jpeg_image_paths):
                start_time = (idx + 1) * interval
                filter_complex.append(f"[{idx+1}:v]format=rgba[img{idx}];")
                if idx < n_images - 1:
                    overlay_stream = f"{overlay_stream}[img{idx}]overlay=enable='between(t,{start_time},{start_time+2})':x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2[tmp{idx}];[tmp{idx}]"
                else:
                    overlay_stream = f"{overlay_stream}[img{idx}]overlay=enable='between(t,{start_time},{start_time+2})':x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2[v]"
            filter_complex_str = ''.join(filter_complex) + overlay_stream
            cmd = [
                'ffmpeg', '-y', *input_args,
                '-filter_complex', filter_complex_str,
                '-map', '[v]', '-map', '0:a?',
                '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
                '-movflags', '+faststart',
                '-threads', '1', '-preset', 'ultrafast',
                output_path
            ]
            print(f"Running FFmpeg for video composition: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            return output_path
        except Exception as e:
            print(f"Error creating video with images using FFmpeg: {e}")
            return None

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return float(result.stdout)
        except Exception as e:
            print(f"Error getting video duration: {e}")
            return 0.0

    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """Generate a pre-signed S3 URL for downloading a file."""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            print(f"Error generating pre-signed URL: {e}")
            return None

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
        "processor_ready": processor is not None,
        "memory_optimized": True
    }

@app.post("/process-video", response_model=ProcessingStatus)
async def process_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    num_images_per_segment: int = Form(1)
):
    """Process uploaded video and generate relevant images for each transcript segment"""
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

    # Add background task with the saved file path and num_images_per_segment
    background_tasks.add_task(process_video_background, task_id, temp_video_path, num_images_per_segment)

    return task_status[task_id]

@app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_status(task_id: str):
    """Get processing status for a task"""
    status = task_status.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task ID not found")
    processor = None
    try:
        processor = get_processor()
    except Exception:
        pass
    if processor:
        results_file_path = os.path.join(processor.output_dir, f"{task_id}_results.json")
        if os.path.exists(results_file_path):
            status.download_url = f"/download/{task_id}"
        s3_key = f"results/{task_id}_with_images.mp4"
        status.s3_presigned_url = processor.generate_presigned_url(s3_key)
        s3_key_json = f"results/{task_id}_results.json"
        s3_url = f"https://{processor.s3_bucket}.s3.amazonaws.com/{s3_key_json}"
        status.s3_url = s3_url
    return status

async def process_video_background(task_id: str, temp_video_path: str, num_images_per_segment: int = 1):
    """Background task to process video and generate images for each segment"""
    try:
        task_status[task_id].status = "processing"
        task_status[task_id].message = "Processing video..."
        processor = get_processor()
        # Step 1: Extract audio from video using FFmpeg
        task_status[task_id].message = "Extracting audio from video..."
        temp_audio_path = os.path.join(processor.temp_dir, f"{task_id}_audio.mp3")
        processor.extract_audio_ffmpeg(temp_video_path, temp_audio_path)
        # Step 2: Transcribe audio using Whisper API
        task_status[task_id].message = "Transcribing audio..."
        transcript = processor.transcribe_audio(temp_audio_path)
        # Step 3: Analyze content using ChatGPT and divide into segments
        task_status[task_id].message = "Analyzing content and dividing into segments..."
        try:
            segments = processor.analyze_content(transcript)
            # Step 4: Generate images using DALL-E for each segment
            task_status[task_id].message = "Generating images for each segment..."
            all_images = processor.generate_images(segments, num_images_per_segment=num_images_per_segment, task_id=task_id)
        except Exception as e:
            print(f"Error in new segmented approach, falling back to original method: {e}")
            # Fallback to original approach
            task_status[task_id].message = "Using fallback method for content analysis..."
            analysis = processor._fallback_analyze_content(transcript)
            all_images = processor._fallback_generate_images(analysis, num_images_per_segment, task_id)
            segments = [{"segment_id": 1, "description": analysis, "keywords": [], "mood": "neutral", "sentence": transcript}]
        # Step 5: Create video with images (robust check)
        # Get all generated image paths
        image_paths = []
        for segment_key, segment_data in all_images.items():
            if "images" in segment_data and isinstance(segment_data["images"], list):
                for i, image_data in enumerate(segment_data["images"]):
                    if isinstance(image_data, dict) and "url" in image_data and not image_data["url"].startswith("ERROR"):
                        try:
                            # Handle both segmented and fallback approaches
                            if segment_key.startswith("segment_"):
                                # New segmented approach
                                segment_id = segment_key.split("_")[1]
                                img_path = os.path.join(processor.output_dir, f"{task_id}_segment_{segment_id}_image_{i+1}.png")
                            else:
                                # Fallback approach
                                img_path = os.path.join(processor.output_dir, f"{task_id}_fallback_image_{i+1}.png")
                            
                            if os.path.exists(img_path):
                                image_paths.append(img_path)
                        except (IndexError, KeyError) as e:
                            print(f"Error extracting segment_id from {segment_key}: {e}")
                            continue
        existing_image_paths = [p for p in image_paths if os.path.exists(p)]
        missing_images = [p for p in image_paths if not os.path.exists(p)]
        processed_video_path = os.path.join(processor.output_dir, f"{task_id}_with_images.mp4")
        s3_presigned_url = None
        if existing_image_paths:
            print(f"Images found for video creation: {existing_image_paths}")
            if missing_images:
                print(f"Warning: The following images are missing and will not be included: {missing_images}")
            result = processor.create_video_with_images(temp_video_path, existing_image_paths, processed_video_path)
            if result and os.path.exists(processed_video_path):
                s3_video_key = f"results/{task_id}_with_images.mp4"
                processor.upload_to_s3(processed_video_path, s3_video_key)
                s3_presigned_url = processor.generate_presigned_url(s3_video_key)
            else:
                print(f"Error: Video creation failed for task {task_id}.")
                task_status[task_id].status = "error"
                task_status[task_id].message = f"Video creation failed. Check logs for details."
        else:
            print(f"No images found for video creation for task {task_id}. Skipping video creation.")
            task_status[task_id].status = "error"
            task_status[task_id].message = "No images were generated for video creation."
        # Step 6: Create results file
        task_status[task_id].message = "Creating results file..."
        results = {
            "task_id": task_id,
            "transcript": transcript,
            "segments": segments,
            "all_images": all_images,
            "total_segments": len(segments),
            "total_images_generated": len(image_paths),
            "timestamp": datetime.now().isoformat()
        }
        results_file_path = os.path.join(processor.output_dir, f"{task_id}_results.json")
        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=2)
        # Step 7: Upload results to S3
        task_status[task_id].message = "Uploading results to S3..."
        s3_key = f"results/{task_id}_results.json"
        s3_url = processor.upload_to_s3(results_file_path, s3_key)
        # Clean up temporary files
        try:
            os.remove(temp_video_path)
            os.remove(temp_audio_path)
        except:
            pass
        # Update status to completed if not already error
        if task_status[task_id].status != "error":
            task_status[task_id].status = "completed"
            task_status[task_id].message = "Video processing completed successfully"
            task_status[task_id].s3_url = s3_url
            task_status[task_id].s3_presigned_url = s3_presigned_url
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

@app.get("/download/{task_id}")
async def download_results(task_id: str):
    processor = get_processor()
    processed_video_path = os.path.join(processor.output_dir, f"{task_id}_with_images.mp4")
    if os.path.exists(processed_video_path):
        return FileResponse(processed_video_path, media_type="video/mp4", filename=f"{task_id}_with_images.mp4")
    else:
        raise HTTPException(status_code=404, detail="Processed video not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 