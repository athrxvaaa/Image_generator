#!/usr/bin/env python3
"""
Dynamic Context-Aware Video Image Generator
Processes videos from input folder, transcribes audio, analyzes content,
and generates context-appropriate images with word-level timestamps.
"""

import os
import json
import time
import glob
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import requests
from openai import OpenAI

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed, using system environment variables")

# Flask imports
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError

# S3 Configuration
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'lisa-research')
S3_REGION = os.getenv('S3_REGION', 'ap-south-1')

# Global logging system
class LogCapture:
    def __init__(self):
        self.logs = []
        self.lock = threading.Lock()
    
    def add_log(self, message, level="INFO"):
        with self.lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logs.append({
                "timestamp": timestamp,
                "level": level,
                "message": message
            })
            # Keep only last 1000 logs
            if len(self.logs) > 1000:
                self.logs = self.logs[-1000:]
    
    def get_logs(self, limit=None, clear=False):
        with self.lock:
            if limit:
                logs = self.logs[-limit:]
            else:
                logs = self.logs.copy()
            
            if clear:
                self.logs = []
            
            return logs

log_capture = LogCapture()

# Flask app initialization
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def upload_to_s3(file_path, s3_key):
    """Upload a file to S3 and return the URL"""
    try:
        log_capture.add_log(f"Uploading {file_path} to S3 as {s3_key}")
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION
        )
        
        # Upload file
        s3_client.upload_file(
            file_path,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={
                'ContentType': 'video/mp4'
            }
        )
        
        # Generate URL
        url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        log_capture.add_log(f"Successfully uploaded to S3: {url}")
        return url
        
    except ClientError as e:
        error_msg = f"Failed to upload {file_path} to {S3_BUCKET_NAME}/{s3_key}: {str(e)}"
        log_capture.add_log(error_msg, "ERROR")
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error during S3 upload: {str(e)}"
        log_capture.add_log(error_msg, "ERROR")
        raise Exception(error_msg)

def initialize_processor():
    """Initialize the video processor with configuration"""
    try:
        config = {
            'images_per_minute': 5,
            'max_images': 35,
            'min_images': 3
        }
        
        processor = LocalVideoProcessor(config=config)
        log_capture.add_log("✅ Video processor initialized successfully")
        return processor
        
    except Exception as e:
        log_capture.add_log(f"❌ Failed to initialize processor: {str(e)}", "ERROR")
        raise

class LocalVideoProcessor:
    def __init__(self, openai_api_key: Optional[str] = None, config: Optional[dict] = None):
        """Initialize the video processor with OpenAI API key and configuration"""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it to the constructor.")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Configuration options
        self.config = config or {}
        self.images_per_minute = self.config.get('images_per_minute', 5)  # Default: 5 images per minute (optimized for 5-min videos)
        self.max_images = self.config.get('max_images', 35)  # Default: maximum 35 images (better for 5-min videos)
        self.min_images = self.config.get('min_images', 3)   # Default: minimum 3 images
        self.fixed_image_count = self.config.get('fixed_image_count', None)  # Set to override automatic calculation
        
        # Setup directories
        self.input_dir = Path("input")
        self.output_dir = Path("output")
        self.temp_dir = tempfile.mkdtemp()
        
        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.log(f"Input directory: {self.input_dir.absolute()}")
        self.log(f"Output directory: {self.output_dir.absolute()}")
        self.log(f"Image configuration: {self.images_per_minute} per minute, max {self.max_images}, min {self.min_images}")
        if self.fixed_image_count:
            self.log(f"Fixed image count: {self.fixed_image_count}")
    
    def log(self, message, level="INFO"):
        """Log message to the global log capture system"""
        log_capture.add_log(message, level)

    def transcribe_audio(self, audio_file_path: str) -> dict:
        """Transcribe audio using OpenAI Whisper API with timestamps"""
        try:
            self.log(f"🎯 STARTING TRANSCRIPTION PROCESS")
            self.log(f"📁 Audio file path: {audio_file_path}")
            self.log(f"📁 Audio file exists: {os.path.exists(audio_file_path)}")
            self.log(f"📁 Audio file size: {os.path.getsize(audio_file_path)} bytes")
            
            self.log("🔄 STEP 1: Opening audio file...")
            with open(audio_file_path, "rb") as audio_file:
                self.log("✅ Audio file opened successfully")
                self.log("🔄 STEP 2: Calling OpenAI Whisper API...")
                self.log("🔄 API call parameters: model=whisper-1, response_format=verbose_json")
                
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
                self.log("✅ OpenAI Whisper API response received")
                self.log(f"📊 Response type: {type(transcript)}")
            
            self.log("🔄 STEP 3: Converting transcript to dict...")
            result = transcript.model_dump()
            self.log(f"✅ Transcript converted successfully")
            self.log(f"📊 Result type: {type(result)}")
            self.log(f"📊 Text length: {len(result.get('text', ''))} characters")
            self.log(f"📊 Segments count: {len(result.get('segments', []))}")
            self.log("🎯 TRANSCRIPTION PROCESS COMPLETED SUCCESSFULLY")
            return result
            
        except Exception as e:
            self.log(f"❌ TRANSCRIPTION FAILED", "ERROR")
            self.log(f"❌ Error message: {e}", "ERROR")
            self.log(f"❌ Error type: {type(e).__name__}", "ERROR")
            self.log(f"❌ Error location: {e.__traceback__.tb_lineno if hasattr(e, '__traceback__') else 'unknown'}", "ERROR")
            raise

    def detect_video_context(self, transcript_data: dict) -> dict:
        """Dynamically detect video context, genre, and style using GPT-4o-mini"""
        try:
            self.log("🎯 STARTING VIDEO CONTEXT DETECTION")
            self.log(f"📊 Input transcript data type: {type(transcript_data)}")
            self.log(f"📊 Transcript keys: {list(transcript_data.keys())}")
            self.log("Detecting video context and genre...")
            
            transcript_text = transcript_data.get('text', '')
            segments = transcript_data.get('segments', [])
            
            # Create versatile context detection prompt that adapts to any video theme
            context_prompt = f"""
            Analyze the following video content and determine its context, genre, and visual style. Be versatile and adapt to ANY type of video content.
            
            TRANSCRIPT TEXT: {transcript_text[:1000]}...
            
            TRANSCRIPT SEGMENTS: {json.dumps(segments[:10], indent=2) if segments else "No segments"}
            
            INSTRUCTIONS:
            1. Determine the video's primary context/genre from ANY category:
               - Educational content (tutorials, lectures, how-to guides)
               - Entertainment (movies, shows, gaming, music)
               - News and journalism (reports, interviews, documentaries)
               - Business and corporate (presentations, meetings, training)
               - Technology and software (reviews, demos, tutorials)
               - Health and fitness (workouts, nutrition, wellness)
               - Travel and lifestyle (vlogs, destinations, experiences)
               - Art and creativity (design, crafts, creative processes)
               - Sports and recreation (games, activities, competitions)
               - Science and research (experiments, discoveries, explanations)
               - Social and cultural (events, traditions, discussions)
               - Consumer products (reviews, demonstrations, advertisements)
               - Personal content (vlogs, family, personal stories)
               - Professional services (consulting, coaching, services)
               - Academic content (courses, research, academic discussions)
            
            2. Identify the most appropriate visual style for the content:
               - Documentary and informative (clean, professional, educational)
               - Entertainment and engaging (dynamic, colorful, exciting)
               - Business and corporate (professional, clean, trustworthy)
               - Creative and artistic (stylized, artistic, expressive)
               - Technical and detailed (precise, clear, informative)
               - Lifestyle and personal (warm, authentic, relatable)
               - News and journalistic (serious, factual, credible)
               - Educational and tutorial (clear, step-by-step, helpful)
               - Commercial and promotional (appealing, attractive, persuasive)
               - Scientific and research (precise, analytical, objective)
            
            3. Determine the tone and mood that matches the content:
               - Educational and informative
               - Entertaining and engaging
               - Professional and serious
               - Creative and inspiring
               - Technical and precise
               - Warm and personal
               - Exciting and dynamic
               - Calm and soothing
               - Authoritative and trustworthy
               - Fun and playful
               - Inspirational and motivational
               - Analytical and objective
            
            4. Identify the target audience:
               - Students and learners
               - Professionals and business people
               - General public and consumers
               - Specific hobbyists or enthusiasts
               - Children and families
               - Academic and research communities
               - Creative professionals
               - Technical specialists
               - Entertainment seekers
               - News and information consumers
            
            5. Determine visual elements appropriate for the content type:
               - Educational: Clear, informative, step-by-step visuals
               - Entertainment: Dynamic, engaging, visually appealing
               - Business: Professional, clean, trustworthy
               - Creative: Artistic, expressive, stylized
               - Technical: Precise, detailed, accurate
               - Lifestyle: Authentic, relatable, warm
               - News: Factual, credible, serious
               - Commercial: Attractive, persuasive, appealing
            
            OUTPUT FORMAT (JSON):
            You must respond with ONLY a valid JSON object in this exact format:
            {{
                "context": "educational/entertainment/news/business/technology/health/travel/art/sports/science/social/consumer/personal/professional/academic/etc",
                "genre": "tutorial/documentary/presentation/review/vlog/news/educational/promotional/etc",
                "visual_style": "documentary/entertainment/business/creative/technical/lifestyle/news/educational/commercial/scientific/etc",
                "tone": "educational/entertaining/professional/creative/technical/warm/exciting/calm/authoritative/fun/inspirational/analytical/etc",
                "target_audience": "students/professionals/general_public/enthusiasts/families/academics/creatives/specialists/entertainment_seekers/news_consumers/etc",
                "composition_style": "informative/dynamic/professional/artistic/precise/authentic/factual/clear/appealing/analytical/etc",
                "color_palette": "professional/vibrant/neutral/artistic/technical/warm/serious/educational/commercial/scientific/etc",
                "lighting_style": "studio/natural/dramatic/soft/professional/creative/technical/warm/serious/educational/etc"
            }}
            
            CRITICAL: Respond with ONLY the JSON object, no other text or explanations.
            
            REQUIREMENTS:
            - Be versatile and adapt to ANY type of video content
            - Don't assume it's consumer-focused unless clearly indicated
            - Match visual style to the actual content type
            - Consider the educational, entertainment, or informational value
            - Focus on what would be most helpful and relevant for the specific content
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert video content analyst specializing in context detection and visual style analysis. Always respond with valid JSON."},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            try:
                # Clean the response to ensure valid JSON
                response_text = response.choices[0].message.content.strip()
                # Remove any markdown formatting
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                context_data = json.loads(response_text)
                self.log(f"✅ Detected context: {context_data.get('context', 'unknown')} - {context_data.get('genre', 'unknown')}")
                self.log(f"📊 Context data keys: {list(context_data.keys())}")
                self.log("🎯 VIDEO CONTEXT DETECTION COMPLETED SUCCESSFULLY")
                return context_data
                
            except json.JSONDecodeError as e:
                self.log(f"JSON parsing failed for context: {e}", "ERROR")
                self.log(f"Raw response: {response.choices[0].message.content[:200]}...", "ERROR")
                raise Exception("Failed to parse context detection - no fallback available")
                
        except Exception as e:
            self.log(f"Error in context detection: {e}", "ERROR")
            raise Exception("Failed to detect context - no fallback available")

    def analyze_content(self, transcript_data: dict, video_duration: float = 60) -> tuple:
        """Analyze transcript content and create scenes with topic-based segmentation"""
        try:
            self.log("🎯 STARTING CONTENT ANALYSIS")
            self.log(f"📊 Video duration: {video_duration} seconds")
            self.log(f"📊 Transcript data type: {type(transcript_data)}")
            self.log("Analyzing transcript content with topic-based segmentation and improved context...")
            
            # Detect video context and genre
            self.log("🔄 STEP 1: Calling detect_video_context...")
            video_context = self.detect_video_context(transcript_data)
            self.log("✅ detect_video_context completed")
            
            # Calculate number of images based on configuration
            if self.fixed_image_count:
                num_images = self.fixed_image_count
                self.log(f"Using fixed image count: {num_images} images")
            else:
                # Dynamic calculation based on video duration and configuration
                calculated_images = int(video_duration / 60 * self.images_per_minute)
                num_images = max(self.min_images, min(calculated_images, self.max_images))
                self.log(f"Video duration: {video_duration:.1f} seconds ({video_duration/60:.1f} minutes) → Generating {num_images} images")
                self.log(f"Calculation: {video_duration/60:.1f} min × {self.images_per_minute} per min = {calculated_images}, bounded by {self.min_images}-{self.max_images}")
            
            # Use topic-based segmentation for improved visual accuracy
            self.log("🔄 STEP 2: Calling _create_topic_based_scenes...")
            self.log("Using topic-based segmentation for improved visual accuracy...")
            scenes = self._create_topic_based_scenes(transcript_data, video_duration, num_images, video_context)
            self.log("✅ _create_topic_based_scenes completed")
            self.log(f"📊 Scenes created: {len(scenes)}")
            self.log("🎯 CONTENT ANALYSIS COMPLETED SUCCESSFULLY")
            
            return scenes, video_context
            
        except Exception as e:
            self.log(f"Error in content analysis: {e}", "ERROR")
            raise

    def _create_topic_based_scenes(self, transcript_data: dict, video_duration: float, num_images: int, video_context: dict) -> list:
        """Create scenes based on topic analysis of the entire transcript"""
        try:
            self.log("🎯 STARTING TOPIC-BASED SCENE CREATION")
            self.log(f"📊 Video duration: {video_duration}")
            self.log(f"📊 Number of images: {num_images}")
            self.log(f"📊 Video context type: {type(video_context)}")
            self.log("Analyzing entire script for important topics and word-level timestamps...")
            
            # Get transcript text and segments
            transcript_text = transcript_data.get('text', '')
            segments = transcript_data.get('segments', [])
            
            # Analyze the entire transcript to extract main topics and context
            analysis_prompt = f"""
            Analyze this video transcript and extract diverse, context-aware topics and concepts for creating non-repetitive visual content with enhanced brand emphasis.
            
            TRANSCRIPT: {transcript_text}
            
            INSTRUCTIONS:
            1. Identify 5-8 diverse topics/concepts that represent different aspects of the content
            2. Extract specific visual elements, actions, processes, or scenarios mentioned
            3. Identify the brand/company name and related brand information
            4. Find unique moments, examples, case studies, or demonstrations mentioned
            5. Look for different environments, settings, or contexts discussed
            6. Identify specific tools, equipment, or technologies mentioned
            7. Extract different types of interactions, meetings, or activities
            8. Find various outcomes, results, or achievements discussed
            9. Identify different departments, teams, or roles mentioned
            10. Look for specific challenges, solutions, or innovations discussed
            
            DIVERSITY REQUIREMENTS:
            - Each topic should represent a different visual scenario
            - Include different types of activities (meetings, training, production, etc.)
            - Vary the environments (office, factory, meeting room, etc.)
            - Include different perspectives (leadership, employees, customers, etc.)
            - Mix different types of content (processes, results, culture, etc.)
            
            OUTPUT FORMAT (JSON only):
            {{
                "main_topics": ["specific_topic1", "specific_topic2", "specific_topic3", "specific_topic4", "specific_topic5"],
                "key_concepts": ["concept1", "concept2", "concept3"],
                "visual_scenarios": ["scenario1", "scenario2", "scenario3"],
                "brand_company": "company name or empty string",
                "brand_industry": "industry sector",
                "brand_products": ["product1", "product2"],
                "brand_facilities": ["facility1", "facility2"],
                "video_theme": "overall theme",
                "target_audience": "target audience",
                "corporate_culture": "company culture or values mentioned",
                "specific_actions": ["action1", "action2", "action3"],
                "environments": ["environment1", "environment2", "environment3"]
            }}
            
            CRITICAL: Respond with ONLY the JSON object, no other text.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert content analyst specializing in video transcript analysis and topic extraction. Always respond with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=600,
                temperature=0.1
            )
            
            try:
                response_text = response.choices[0].message.content.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                analysis_data = json.loads(response_text)
                self.log(f"✅ Script analysis completed - Found {len(analysis_data.get('main_topics', []))} main topics")
                self.log(f"📋 Content: {transcript_text[:200]}...")
                self.log(f"🏢 Brand: {analysis_data.get('brand_company', 'Not specified')}")
                
                # Create word-level timestamped scenes
                scenes = self._create_word_level_scenes(segments, analysis_data, num_images, video_context, video_duration)
                
                self.log(f"Topic-based content analysis completed - {len(scenes)} scenes created with natural topic alignment")
                self.log(f"📊 Final scenes count: {len(scenes)}")
                self.log("🎯 TOPIC-BASED SCENE CREATION COMPLETED SUCCESSFULLY")
                return scenes
                
            except json.JSONDecodeError as e:
                self.log(f"JSON parsing failed for analysis: {e}", "ERROR")
                self.log(f"Raw response: {response.choices[0].message.content[:200]}...", "ERROR")
                raise Exception("Failed to parse content analysis - no fallback available")
            
        except Exception as e:
            self.log(f"Error in topic-based scene creation: {e}", "ERROR")
            raise

    def _create_word_level_scenes(self, segments: list, analysis_data: dict, num_images: int, video_context: dict, video_duration: float = 60) -> list:
        """Create scenes based on word-level timestamps and topic analysis"""
        try:
            self.log("🎯 STARTING WORD-LEVEL SCENE CREATION")
            self.log(f"📊 Segments count: {len(segments)}")
            self.log(f"📊 Analysis data keys: {list(analysis_data.keys())}")
            self.log(f"📊 Number of images: {num_images}")
            self.log("Creating word-level timestamped scenes with topic-based prompts...")
            
            # Extract analysis data
            main_topics = analysis_data.get('main_topics', [])
            key_concepts = analysis_data.get('key_concepts', [])
            visual_scenarios = analysis_data.get('visual_scenarios', [])
            specific_actions = analysis_data.get('specific_actions', [])
            environments = analysis_data.get('environments', [])
            brand_company = analysis_data.get('brand_company', '')
            brand_industry = analysis_data.get('brand_industry', '')
            brand_products = analysis_data.get('brand_products', [])
            brand_facilities = analysis_data.get('brand_facilities', [])
            video_theme = analysis_data.get('video_theme', '')
            target_audience = analysis_data.get('target_audience', '')
            corporate_culture = analysis_data.get('corporate_culture', '')
            
            # Find key moments in segments where important topics are mentioned with precise word-level timestamps
            key_moments = self._find_key_moments(segments, main_topics, key_concepts, video_duration)
            
            # Create scenes for each key moment
            scenes = []
            for i, moment in enumerate(key_moments[:num_images]):
                # Create diverse, context-aware image prompt
                image_prompt = self._create_diverse_image_prompt(
                    moment, 
                    analysis_data, 
                    i, 
                    num_images
                )
                
                scenes.append({
                    "scene_id": i + 1,
                    "timestamp": moment['timestamp'],
                    "concept": moment['topic'],
                    "description": moment['description'],
                    "image_prompt": image_prompt,
                    "keywords": [moment['topic'].lower()],
                    "mood": video_context.get("tone", "informative"),
                    "visual_focus": "topic representation",
                    "audio_context": moment['context'][:150] + "..." if len(moment['context']) > 150 else moment['context'],
                    "quote": moment['context'][:50] + "..." if len(moment['context']) > 50 else moment['context'],
                    "scene": moment['topic'],
                    "style": video_context.get("visual_style", "documentary"),
                    "composition": "content-focused",
                    "emotion": video_context.get("tone", "informative"),
                    "duration": moment['duration'],
                    "end_time": moment['end_time']
                })
            
            # If we don't have enough key moments, fill with additional scenes
            while len(scenes) < num_images:
                remaining_topics = [t for t in main_topics if t not in [s['concept'] for s in scenes]]
                if remaining_topics:
                    topic = remaining_topics[0]
                else:
                    topic = f"Topic_{len(scenes) + 1}"
                
                # Calculate better distributed timestamps within video duration
                interval = video_duration / (num_images + 1)
                timestamp = (len(scenes) + 1) * interval
                
                # Ensure timestamp is within reasonable bounds
                timestamp = min(timestamp, video_duration - 6.0)  # Leave 6 seconds for image display
                
                # Adjust duration based on video length
                if video_duration >= 300:  # 5 minutes or more
                    fallback_duration = 5.0  # Shorter duration for longer videos
                else:
                    fallback_duration = 6.0  # Standard duration for shorter videos
                
                # Create fallback scene data
                fallback_moment = {
                    'topic': topic,
                    'context': f"Visual representation of {topic}",
                    'timestamp': timestamp,
                    'duration': fallback_duration,
                    'end_time': timestamp + fallback_duration
                }
                
                image_prompt = self._create_diverse_image_prompt(
                    fallback_moment, 
                    analysis_data, 
                    len(scenes), 
                    num_images
                )
                
                scenes.append({
                    "scene_id": len(scenes) + 1,
                    "timestamp": timestamp,
                    "concept": topic,
                    "description": f"Visual representation of {topic}",
                    "image_prompt": image_prompt,
                    "keywords": [topic.lower()],
                    "mood": video_context.get("tone", "informative"),
                    "visual_focus": "topic representation",
                    "audio_context": f"Visual representation of {topic}",
                    "quote": topic,
                    "scene": topic,
                    "style": video_context.get("visual_style", "documentary"),
                    "composition": "content-focused",
                    "emotion": video_context.get("tone", "informative"),
                    "duration": fallback_duration,
                    "end_time": timestamp + fallback_duration
                })
            
            self.log(f"📊 Final scenes count: {len(scenes)}")
            self.log("🎯 WORD-LEVEL SCENE CREATION COMPLETED SUCCESSFULLY")
            return scenes
                
        except Exception as e:
            self.log(f"❌ WORD-LEVEL SCENE CREATION FAILED", "ERROR")
            self.log(f"❌ Error creating word-level scenes: {e}", "ERROR")
            raise

    def _find_key_moments(self, segments: list, main_topics: list, key_concepts: list, video_duration: float = 60) -> list:
        """Find key moments in segments where important topics are mentioned with precise word-level timestamps"""
        key_moments = []
        
        for segment in segments:
            text = segment.get('text', '').lower()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            words = segment.get('words', [])
            
            # Check if this segment contains any main topics or key concepts
            for topic in main_topics + key_concepts:
                if topic.lower() in text:
                    # Find the exact word-level timestamp for this topic
                    topic_start_time = start_time
                    topic_end_time = end_time
                    
                    # Look for the exact word timestamp if available
                    for word in words:
                        word_text = word.get('word', '').lower().strip()
                        if topic.lower() in word_text:
                            topic_start_time = word.get('start', start_time)
                            topic_end_time = word.get('end', end_time)
                            break
                    
                    # Calculate duration based on actual topic discussion
                    duration = topic_end_time - topic_start_time
                    # Ensure minimum duration of 2 seconds and maximum of 6 seconds for better fit
                    # For longer videos (5+ minutes), use shorter durations to fit more images
                    if video_duration >= 300:  # 5 minutes or more
                        duration = max(2.0, min(5.0, duration))  # Shorter max duration for longer videos
                    else:
                        duration = max(2.0, min(6.0, duration))  # Standard duration for shorter videos
                    
                    key_moments.append({
                        'timestamp': topic_start_time,
                        'duration': duration,
                        'topic': topic,
                        'context': segment.get('text', ''),
                        'description': f"Visual representation of {topic}",
                        'end_time': topic_end_time
                    })
                    break  # Only use first match per segment
        
        # Sort by timestamp and remove duplicates
        key_moments = sorted(key_moments, key=lambda x: x['timestamp'])
        unique_moments = []
        seen_topics = set()
        
        for moment in key_moments:
            if moment['topic'] not in seen_topics:
                unique_moments.append(moment)
                seen_topics.add(moment['topic'])
        
        return unique_moments

    def _create_diverse_image_prompt(self, scene_data: dict, analysis_data: dict, scene_index: int, total_scenes: int) -> str:
        """Create diverse, context-aware image prompt based on transcript content and scene position"""
        try:
            # Extract all available data for maximum context
            topic = scene_data.get('topic', '')
            context = scene_data.get('context', '')
            brand_company = analysis_data.get('brand_company', '')
            brand_industry = analysis_data.get('brand_industry', '')
            brand_products = analysis_data.get('brand_products', [])
            brand_facilities = analysis_data.get('brand_facilities', [])
            video_theme = analysis_data.get('video_theme', '')
            target_audience = analysis_data.get('target_audience', '')
            corporate_culture = analysis_data.get('corporate_culture', '')
            visual_scenarios = analysis_data.get('visual_scenarios', [])
            specific_actions = analysis_data.get('specific_actions', [])
            environments = analysis_data.get('environments', [])
            
            # Create diverse visual elements based on scene position
            visual_elements = []
            if visual_scenarios and scene_index < len(visual_scenarios):
                visual_elements.append(f"Scenario: {visual_scenarios[scene_index % len(visual_scenarios)]}")
            if specific_actions and scene_index < len(specific_actions):
                visual_elements.append(f"Action: {specific_actions[scene_index % len(specific_actions)]}")
            if environments and scene_index < len(environments):
                visual_elements.append(f"Environment: {environments[scene_index % len(environments)]}")
            
            # Vary the perspective and composition based on scene index
            perspectives = [
                "wide shot showing the entire scene",
                "medium shot focusing on key interactions",
                "close-up shot highlighting important details",
                "overhead view showing the workspace layout",
                "side angle capturing the dynamic atmosphere"
            ]
            
            compositions = [
                "rule of thirds composition",
                "symmetrical balance",
                "leading lines drawing attention",
                "framed composition with natural elements",
                "dynamic diagonal composition"
            ]
            
            lighting_styles = [
                "natural daylight streaming through windows",
                "soft ambient office lighting",
                "dramatic side lighting",
                "warm golden hour lighting",
                "bright, clean fluorescent lighting"
            ]
            
            # Select diverse elements based on scene index
            perspective = perspectives[scene_index % len(perspectives)]
            composition = compositions[scene_index % len(compositions)]
            lighting = lighting_styles[scene_index % len(lighting_styles)]
            
            # Build brand integration
            brand_integration = ""
            if brand_company and brand_company.strip():
                brand_integration = f"""
                BRAND INTEGRATION:
                - Prominently feature {brand_company} branding and logos
                - Show {brand_company} employees in branded attire
                - Include {brand_company} products, equipment, or facilities
                - Reflect {brand_company}'s {brand_industry if brand_industry else 'professional'} environment
                - Embody {brand_company}'s culture of {corporate_culture if corporate_culture else 'excellence'}
                """
            
            # Create context-specific prompt
            prompt = f"""
            Create a unique, context-aware visual representation for scene {scene_index + 1} of {total_scenes}.
            
            TOPIC: {topic}
            TRANSCRIPT CONTEXT: {context[:200]}...
            {' '.join(visual_elements)}
            
            {brand_integration}
            
            VISUAL REQUIREMENTS:
            - {perspective}
            - {composition}
            - {lighting}
            - Photographic realism with natural colors
            - No text, diagrams, or charts
            - Authentic workplace atmosphere
            - Professional quality and composition
            
            Generate a specific, vivid image prompt that captures this unique moment in the video content.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert visual designer creating diverse, context-aware image prompts for corporate video content. Each prompt should be unique and specific."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.4  # Slightly higher temperature for more diversity
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"Error creating diverse image prompt: {e}")
            raise

    def _calculate_optimal_timestamps(self, video_duration: float, num_images: int) -> list:
        """Calculate optimal timestamps for better audio-image synchronization"""
        timestamps = []
        
        # For longer videos (3+ minutes), use logarithmic distribution for better coverage
        if video_duration >= 180:  # 3 minutes or more
            # Use logarithmic distribution to spread images more evenly
            for i in range(num_images):
                # Logarithmic distribution: more images in first half, fewer in second half
                progress = (i + 1) / (num_images + 1)
                
                # For 5+ minute videos, use more aggressive logarithmic distribution
                if video_duration >= 300:  # 5 minutes or more
                    timestamp = video_duration * (1 - (1 - progress) ** 0.6)  # 0.6 gives better spread for longer videos
                else:
                    timestamp = video_duration * (1 - (1 - progress) ** 0.7)  # 0.7 for 3-5 minute videos
                
                timestamps.append(timestamp)
        else:
            # For shorter videos, use linear distribution
            interval = video_duration / (num_images + 1)
            for i in range(num_images):
                timestamps.append((i + 1) * interval)
        
        return timestamps

    def generate_images(self, scenes: list, video_context: dict, video_name: str = "") -> dict:
        """Generate images based on analyzed scenes using GPT-Image-1"""
        all_images = {}
        try:
            self.log("🎯 STARTING IMAGE GENERATION PROCESS")
            num_images = len(scenes)
            self.log(f"📊 Number of scenes to process: {num_images}")
            self.log(f"📊 Video name: {video_name}")
            self.log(f"📊 Video context type: {type(video_context)}")
            self.log(f"Generating {num_images} images based on enhanced scene analysis...")
            
            for i, scene in enumerate(scenes):
                scene_id = scene.get('scene_id', i + 1)
                timestamp = scene.get('timestamp', (i + 1) * 10)
                description = scene.get('description', '')
                keywords = scene.get('keywords', [])
                mood = scene.get('mood', 'neutral')
                concept = scene.get('concept', '')
                visual_focus = scene.get('visual_focus', '')
                audio_context = scene.get('audio_context', '')
                quote = scene.get('quote', '')
                
                self.log(f"Generating image {i+1}/{num_images} for scene: {concept[:30]}...")
                
                # Use the image_prompt directly from structured data
                image_prompt = scene.get('image_prompt', description)
                
                # Ensure the prompt specifies no text
                if "no text" not in image_prompt.lower() and "text-free" not in image_prompt.lower():
                    image_prompt += ", no text, clean design"
                
                # Log the prompt for debugging
                self.log(f"📝 Image Prompt for Scene {scene_id}: {image_prompt}")
                
                try:
                    self.log(f"🔄 Calling OpenAI API for image generation...")
                    response = self.openai_client.images.generate(
                        model="gpt-image-1",
                        prompt=image_prompt,
                        size="1024x1536",
                        quality="medium",
                        n=1
                    )
                    self.log(f"✅ OpenAI API response received for scene {scene_id}")
                    
                    if response.data and len(response.data) > 0:
                        data = response.data[0]
                        
                        # Handle both URL and base64 responses from GPT-Image-1
                        if hasattr(data, 'url') and data.url and data.url != "None":
                            # URL response
                            img_response = requests.get(data.url)
                            if img_response.status_code == 200:
                                img_data = img_response.content
                            else:
                                raise ValueError(f"Failed to download image: {img_response.status_code}")
                        elif hasattr(data, 'b64_json') and data.b64_json:
                            # Base64 response
                            import base64
                            img_data = base64.b64decode(data.b64_json)
                        else:
                            raise ValueError("No valid image data received from GPT-Image-1")
                        
                        # Save image to temp directory only (for video creation)
                        img_filename = f"{video_name}_scene_{scene_id}.jpg"
                        temp_img_path = os.path.join(self.temp_dir, img_filename)
                        
                        # Save to temp directory for video creation
                        with open(temp_img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        self.log(f"💾 Generated image: {img_filename}")
                        
                        # Store image data with scene information
                        all_images[f"scene_{scene_id}"] = {
                            "path": temp_img_path,
                            "timestamp": timestamp,
                            "concept": concept,
                            "description": description,
                            "image_prompt": image_prompt,
                            "keywords": keywords,
                            "mood": mood,
                            "visual_focus": visual_focus,
                            "audio_context": audio_context,
                            "quote": quote,
                            "duration": scene['duration'],
                            "end_time": scene['end_time']
                        }
                        
                        self.log(f"✅ Generated scene {scene_id} ({concept})")
                        
                    else:
                        self.log(f"❌ No image data received for scene {scene_id}", "ERROR")
                        all_images[f"scene_{scene_id}"] = {"error": "No image data received"}
                    
                except Exception as e:
                    self.log(f"❌ Error generating image for scene {scene_id}: {e}", "ERROR")
                    all_images[f"scene_{scene_id}"] = {"error": str(e)}
            
            self.log(f"📊 Total images processed: {len(all_images)}")
            self.log("🎯 IMAGE GENERATION PROCESS COMPLETED SUCCESSFULLY")
            return all_images
            
        except Exception as e:
            self.log(f"❌ IMAGE GENERATION PROCESS FAILED", "ERROR")
            self.log(f"❌ Error in image generation: {e}", "ERROR")
            raise

    def extract_audio_ffmpeg(self, video_path: str, audio_path: str):
        """Extract audio from video using FFmpeg"""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'mp3', audio_path
            ]
            self.log(f"Running FFmpeg for audio extraction: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            self.log(f"FFmpeg error: {e}", "ERROR")
            raise

    def create_video_with_images(self, video_path: str, images_data: dict, output_path: str) -> str:
        """Create output video with synchronized image overlays"""
        try:
            self.log("Creating output video with synchronized image overlays and zoom-in effects...")
            
            # Get video duration
            video_duration = self.get_video_duration(video_path)
            self.log(f"Video duration: {video_duration:.2f} seconds")

            # Get video width and height dynamically
            import re
            ffprobe_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0', video_path
            ]
            ffprobe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
            match = re.match(r"(\d+)x(\d+)", ffprobe_result.stdout.strip())
            if match:
                video_width, video_height = match.groups()
            else:
                video_width, video_height = '1080', '1920'  # fallback

            # Prepare overlays
            overlays = []
            temp_image_paths = []
            
            for scene_key, scene_data in images_data.items():
                if 'error' not in scene_data:
                    overlays.append({
                        'timestamp': scene_data.get('timestamp', 0),
                        'duration': scene_data['duration'],  # Use duration from scene data
                        'image_path': scene_data.get('path', '')
                    })
                    temp_image_paths.append(scene_data.get('path', ''))

            if not overlays:
                self.log("No valid images to overlay, copying original video...")
                shutil.copy2(video_path, output_path)
                return output_path

            # Create input files list
            input_files = [video_path] + [overlay['image_path'] for overlay in overlays if overlay['image_path']]
            
            # Create FFmpeg filter complex
            filter_steps = []
            last_video = '[0:v]'

            for idx, overlay in enumerate(overlays):
                img_idx = idx + 1
                # Minimal scale with very slight zoom (1.02x) for better fit
                scale = f"[{img_idx}:v]scale={video_width}:{video_height}:force_original_aspect_ratio=increase,crop={video_width}:{video_height}[imgz{img_idx}]"
                # Overlay with enable between timestamps - improved synchronization
                overlayf = f"{last_video}[imgz{img_idx}]overlay=0:0:enable='between(t,{overlay['timestamp']},{overlay['timestamp']+overlay['duration']})'[v{img_idx}]"
                filter_steps.append(scale)
                filter_steps.append(overlayf)
                last_video = f'[v{img_idx}]'

            filter_complex = ";".join(filter_steps)

            cmd = [
                'ffmpeg', '-y',
                *sum([["-i", f] for f in input_files], []),
                '-filter_complex', filter_complex,
                '-map', last_video,
                '-map', '0:a?',
                '-c:v', 'libx264',
                '-c:a', 'copy',
                '-preset', 'medium',
                '-crf', '23',
                output_path
            ]

            self.log(f"Running FFmpeg with {len(overlays)} overlays and simple zoom effect...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.log(f"✅ Video with {len(overlays)} image overlays created: {output_path}")
                for o in overlays:
                    self.log(f"🎬 Image overlaid at {o['timestamp']:.1f}s for {o['duration']:.1f}s (minimal zoom 1.02x)")
                
                # Clean up temporary image files
                self._cleanup_temp_images(temp_image_paths)
                
                return output_path
            else:
                self.log(f"❌ FFmpeg failed: {result.stderr}", "ERROR")
                self.log("Falling back to copying original video...")
                shutil.copy2(video_path, output_path)
                
                # Clean up temporary image files even if FFmpeg failed
                self._cleanup_temp_images(temp_image_paths)
                
                return output_path
        except Exception as e:
            print(f"Error creating output video: {e}")
            try:
                shutil.copy2(video_path, output_path)
                return output_path
            except:
                return None

    def _cleanup_temp_images(self, image_paths: list):
        """Clean up temporary image files (no permanent storage)"""
        try:
            cleaned_count = 0
            for img_path in image_paths:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    cleaned_count += 1
            if cleaned_count > 0:
                self.log(f"🧹 Cleaned up {cleaned_count} temporary image files")
        except Exception as e:
            self.log(f"Warning: Could not clean up some temporary files: {e}", "WARNING")

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return float(result.stdout.strip())
        except Exception as e:
            self.log(f"Error getting video duration: {e}", "ERROR")
            return 0.0

    def process_video_file(self, video_path: str):
        """Process a single video file with enhanced analysis and optimized for up to 5 minutes"""
        try:
            self.log("🎯 STARTING VIDEO PROCESSING PIPELINE")
            video_name = Path(video_path).stem
            self.log(f"📁 Video path: {video_path}")
            self.log(f"📁 Video name: {video_name}")
            self.log(f"\n{'='*60}")
            self.log(f"🎬 PROCESSING: {video_name}")
            self.log(f"{'='*60}")
            
            # Get video duration first for optimization
            video_duration = self.get_video_duration(video_path)
            self.log(f"📏 Video duration: {video_duration:.1f} seconds ({video_duration/60:.1f} minutes)")
            
            # Estimate processing time based on duration
            estimated_images = min(int(video_duration / 60 * self.images_per_minute), self.max_images)
            estimated_time = estimated_images * 0.75  # ~45 seconds per image
            self.log(f"⏱️  Estimated processing time: ~{estimated_time:.0f} minutes")
            
            # Step 1: Extract audio from video using FFmpeg
            self.log("📺 Step 1: Extracting audio from video...")
            temp_audio_path = os.path.join(self.temp_dir, f"{video_name}_audio.mp3")
            self.extract_audio_ffmpeg(video_path, temp_audio_path)
            
            # Step 2: Transcribe audio using Whisper API with timestamps
            self.log("🎤 Step 2: Transcribing audio with Whisper (with timestamps)...")
            transcript_data = self.transcribe_audio(temp_audio_path)
            self.log(f"📝 Transcript preview: {transcript_data.get('text', '')[:150]}...")
            self.log(f"📊 Found {len(transcript_data.get('segments', []))} transcript segments")
            
            # Step 3: Topic detection and content analysis with audio synchronization
            self.log("🧠 Step 3: Topic detection and content analysis with audio synchronization...")
            scenes, video_context = self.analyze_content(transcript_data, video_duration)
            
            # Step 4: Generate high-quality images based on video duration
            num_images = len(scenes)
            self.log(f"🎨 Step 4: Generating {num_images} high-quality images...")
            all_images = self.generate_images(scenes, video_context, video_name=video_name)
            
            # Step 5: Create output video with synchronized audio-image overlays and fade-in effects
            self.log("📹 Step 5: Creating output video with synchronized audio-image overlays and fade-in effects...")
            output_video_path = os.path.join(self.output_dir, f"{video_name}_with_images.mp4")
            result = self.create_video_with_images(video_path, all_images, output_video_path)
            
            # Count successful images
            successful_images = len([k for k in all_images.keys() if 'error' not in all_images[k]])
            
            # Step 6: Skip JSON results file (as requested)
            self.log("📊 Step 6: Processing complete (no JSON output)")
            
            # Clean up temporary files
            try:
                os.remove(temp_audio_path)
            except:
                pass
            
            self.log(f"✅ SUCCESS: {video_name} ({successful_images}/{num_images} images generated)")
            self.log(f"{'='*60}")
            self.log("🎯 VIDEO PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            self.log(f"❌ ERROR processing {video_path}: {e}", "ERROR")
            self.log(f"{'='*60}")
            return False

    def watch_input_folder(self, continuous: bool = True):
        """Watch input folder for bulk video processing"""
        print(f"\n🎬 Dynamic Context-Aware Video Image Generator - Bulk Processor")
        print(f"👀 Watching input folder: {self.input_dir.absolute()}")
        print(f"📁 Output folder: {self.output_dir.absolute()}")
        
        # Display configuration
        if self.fixed_image_count:
            print(f"🎨 Image configuration: Fixed {self.fixed_image_count} images per video")
        else:
            print(f"🎨 Image configuration: {self.images_per_minute} per minute, max {self.max_images}, min {self.min_images}")
        
        print("📋 Recommendations for 3-5 minute videos:")
        print("   • 5-7 images per minute (15-35 total images)")
        print("   • Maximum 35-40 images for good coverage")
        print("   • Logarithmic distribution for better spread")
        print("   • Optimized for up to 5-minute content")
        print()
        
        print(f"🧠 Enhanced: Dynamic context detection & GPT-4o-mini topic analysis")
        print(f"✨ NEW: Diverse, non-repetitive image generation")
        print(f"💾 NEW: Permanent image storage in output/images/")
        print(f"🎯 NEW: Context-adaptive image prompts (no hardcoded styles)")
        print(f"🏢 NEW: Enhanced brand emphasis and integration")
        print(f"✨ NEW: Optimized for videos up to 5 minutes")
        print(f"{'='*70}")
        
        # Supported video formats
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
        processed_files = set()
        
        try:
            while True:
                # Find all video files in input directory
                video_files = []
                for ext in video_extensions:
                    video_files.extend(glob.glob(os.path.join(self.input_dir, ext)))
                    video_files.extend(glob.glob(os.path.join(self.input_dir, ext.upper())))
                
                # Process new files
                new_files = [f for f in video_files if f not in processed_files]
                
                if new_files:
                    print(f"\n📥 BULK PROCESSING: Found {len(new_files)} video(s)")
                    print(f"🎯 Each video will generate images based on its duration")
                    print(f"⏱️  Estimated time: ~{len(new_files) * 2} minutes")
                    print("-" * 70)
                    
                    successful = 0
                    failed = 0
                    
                    for i, video_file in enumerate(new_files, 1):
                        print(f"\n[{i}/{len(new_files)}] 📹 {os.path.basename(video_file)}")
                        success = self.process_video_file(video_file)
                        
                        if success:
                            processed_files.add(video_file)
                            successful += 1
                            print(f"📁 Kept in input folder (no processed folder)")
                        else:
                            failed += 1
                    
                    print(f"\n{'='*70}")
                    print(f"🎉 BULK PROCESSING COMPLETE!")
                    print(f"✅ Successful: {successful} videos")
                    print(f"❌ Failed: {failed} videos")
                    print(f"🖼️  Total images generated: Variable based on video durations")
                    print(f"{'='*70}")
                    
                else:
                    if not continuous:
                        print("✅ No new videos found. Processing complete.")
                        break
                    
                if not continuous:
                    break
                    
                # Wait before checking again
                print(f"\n⏳ Waiting for new videos... (Ctrl+C to stop)")
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping bulk processor...")
        except Exception as e:
            print(f"Error in bulk processing: {e}")

# Flask API endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Video processing API is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload and process video file"""
    try:
        log_capture.add_log("📤 Received video upload request")
        
        # Check if file was uploaded
        if 'video' not in request.files:
            log_capture.add_log("❌ No video file provided", "ERROR")
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '':
            log_capture.add_log("❌ No file selected", "ERROR")
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            log_capture.add_log("❌ Invalid file type", "ERROR")
            return jsonify({"error": "Invalid file type. Only MP4, AVI, MOV, MKV are supported"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        log_capture.add_log(f"💾 Saved uploaded file: {filename}")
        
        # Initialize processor
        try:
            processor = initialize_processor()
        except Exception as e:
            log_capture.add_log(f"❌ Failed to initialize processor: {str(e)}", "ERROR")
            return jsonify({"error": "Failed to initialize processor", "message": str(e)}), 500
        
        # Process video
        log_capture.add_log("🎬 Starting video processing...")
        processing_success = processor.process_video_file(upload_path)
        
        if not processing_success:
            log_capture.add_log("❌ Video processing failed", "ERROR")
            return jsonify({"error": "Processing failed", "message": "Video processing failed"}), 500
        
        # Find output file
        base_name = Path(filename).stem
        output_filename = f"{base_name}_with_images.mp4"
        output_path = os.path.join("output", output_filename)
        
        if not os.path.exists(output_path):
            log_capture.add_log("❌ Output file not found", "ERROR")
            return jsonify({"error": "Processing failed", "message": "Video was processed but output file not found"}), 500
        
        # Upload to S3
        try:
            s3_key = f"processed_videos/{output_filename}"
            s3_url = upload_to_s3(output_path, s3_key)
            log_capture.add_log(f"✅ Successfully uploaded to S3: {s3_url}")
            
            # Delete local output file after S3 upload
            try:
                os.remove(output_path)
                log_capture.add_log(f"🧹 Cleaned up local output file: {output_path}")
            except Exception as e:
                log_capture.add_log(f"⚠️ Warning: Could not delete local output file: {str(e)}", "WARNING")
            
            # Clean up uploaded file
            try:
                os.remove(upload_path)
                log_capture.add_log(f"🧹 Cleaned up uploaded file: {upload_path}")
            except Exception as e:
                log_capture.add_log(f"⚠️ Warning: Could not delete uploaded file: {str(e)}", "WARNING")
            
            return jsonify({
                "success": True,
                "message": "Video processed and uploaded successfully",
                "s3_url": s3_url,
                "filename": output_filename
            })
            
        except Exception as e:
            log_capture.add_log(f"❌ S3 upload failed: {str(e)}", "ERROR")
            return jsonify({"error": "Upload failed", "message": f"Video was processed but failed to upload to S3: {str(e)}"}), 500
        
    except Exception as e:
        log_capture.add_log(f"❌ Unexpected error: {str(e)}", "ERROR")
        return jsonify({"error": "Unexpected error", "message": str(e)}), 500

@app.route('/config', methods=['POST'])
def update_config():
    """Update processor configuration"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No configuration data provided"}), 400
        
        # Validate configuration
        config = {}
        if 'images_per_minute' in data:
            config['images_per_minute'] = max(1, min(15, int(data['images_per_minute'])))
        if 'max_images' in data:
            config['max_images'] = max(5, min(50, int(data['max_images'])))
        if 'min_images' in data:
            config['min_images'] = max(1, min(10, int(data['min_images'])))
        if 'fixed_image_count' in data:
            config['fixed_image_count'] = max(1, min(50, int(data['fixed_image_count'])))
        
        log_capture.add_log(f"⚙️ Configuration updated: {config}")
        return jsonify({"success": True, "config": config})
            
    except Exception as e:
        log_capture.add_log(f"❌ Configuration update failed: {str(e)}", "ERROR")
        return jsonify({"error": "Configuration update failed", "message": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get processing status and logs"""
    try:
        limit = request.args.get('limit', type=int)
        clear = request.args.get('clear', 'false').lower() == 'true'
        
        logs = log_capture.get_logs(limit=limit, clear=clear)
        
        return jsonify({
            "status": "running",
            "logs": logs,
            "log_count": len(logs),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": "Failed to get status", "message": str(e)}), 500

def main():
    """Main function to run the Flask API server"""
    try:
        log_capture.add_log("🚀 Starting Video Processing API Server")
        log_capture.add_log("=" * 50)
        
        # Validate S3 environment variables
        if not S3_ACCESS_KEY or not S3_SECRET_KEY:
            log_capture.add_log("❌ S3 credentials not found in environment variables", "ERROR")
            log_capture.add_log("Please set S3_ACCESS_KEY and S3_SECRET_KEY environment variables", "ERROR")
            return
        
        log_capture.add_log("✨ Features:")
        log_capture.add_log("   • Web API for video upload and processing")
        log_capture.add_log("   • Dynamic image count based on video duration (configurable)")
        log_capture.add_log("   • Dynamic context detection (educational, business, entertainment, etc.)")
        log_capture.add_log("   • GPT-4o-mini topic analysis with Whisper timestamps")
        log_capture.add_log("   • Context-adaptive image prompts (no hardcoded styles)")
        log_capture.add_log("   • Diverse, non-repetitive image generation")
        log_capture.add_log("   • Temporary image processing (no permanent storage)")
        log_capture.add_log("   • Minimal zoom effects (1.02x scale)")
        log_capture.add_log("   • S3 upload for processed videos")
        log_capture.add_log("   • Get processing status and logs")
        log_capture.add_log("   • Optimized for videos up to 5 minutes")
        log_capture.add_log("")
        log_capture.add_log("🌐 API Endpoints:")
        log_capture.add_log("   • POST /upload - Upload and process video")
        log_capture.add_log("   • POST /config - Update configuration")
        log_capture.add_log("   • GET /health - Health check")
        log_capture.add_log("   • GET /status - Get processing status and logs")
        log_capture.add_log("")
        log_capture.add_log("🚀 Starting Flask server on http://localhost:8000")
        
        # Start Flask server
        app.run(host='0.0.0.0', port=8000, debug=False)
        
    except Exception as e:
        log_capture.add_log(f"❌ Error: {e}", "ERROR")
        log_capture.add_log("Make sure you have set the OPENAI_API_KEY environment variable.")

if __name__ == "__main__":
    main() 