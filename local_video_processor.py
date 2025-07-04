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
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import requests
from openai import OpenAI

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
        
        print(f"Input directory: {self.input_dir.absolute()}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Image configuration: {self.images_per_minute} per minute, max {self.max_images}, min {self.min_images}")
        if self.fixed_image_count:
            print(f"Fixed image count: {self.fixed_image_count}")

    def transcribe_audio(self, audio_file_path: str) -> dict:
        """Transcribe audio using OpenAI Whisper API with timestamps"""
        try:
            print(f"Transcribing audio file: {audio_file_path}")
            
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )
            
            print("Transcription completed successfully with timestamps")
            return transcript.model_dump()
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise

    def detect_video_context(self, transcript_data: dict) -> dict:
        """Dynamically detect video context, genre, and style using GPT-4o-mini"""
        try:
            print("Detecting video context and genre...")
            
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
                print(f"✅ Detected context: {context_data.get('context', 'unknown')} - {context_data.get('genre', 'unknown')}")
                return context_data
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed for context: {e}")
                print(f"Raw response: {response.choices[0].message.content[:200]}...")
                raise Exception("Failed to parse context detection - no fallback available")
                
        except Exception as e:
            print(f"Error in context detection: {e}")
            raise Exception("Failed to detect context - no fallback available")

    def analyze_content(self, transcript_data: dict, video_duration: float = 60) -> tuple:
        """Analyze transcript content and create scenes with topic-based segmentation"""
        try:
            print("Analyzing transcript content with topic-based segmentation and improved context...")
            
            # Detect video context and genre
            video_context = self.detect_video_context(transcript_data)
            
            # Calculate number of images based on configuration
            if self.fixed_image_count:
                num_images = self.fixed_image_count
                print(f"Using fixed image count: {num_images} images")
            else:
                # Dynamic calculation based on video duration and configuration
                calculated_images = int(video_duration / 60 * self.images_per_minute)
                num_images = max(self.min_images, min(calculated_images, self.max_images))
                print(f"Video duration: {video_duration:.1f} seconds ({video_duration/60:.1f} minutes) → Generating {num_images} images")
                print(f"Calculation: {video_duration/60:.1f} min × {self.images_per_minute} per min = {calculated_images}, bounded by {self.min_images}-{self.max_images}")
            
            # Use topic-based segmentation for improved visual accuracy
            print("Using topic-based segmentation for improved visual accuracy...")
            scenes = self._create_topic_based_scenes(transcript_data, video_duration, num_images, video_context)
            
            return scenes, video_context
            
        except Exception as e:
            print(f"Error in content analysis: {e}")
            raise

    def _create_topic_based_scenes(self, transcript_data: dict, video_duration: float, num_images: int, video_context: dict) -> list:
        """Create scenes based on topic analysis of the entire transcript"""
        try:
            print("Analyzing entire script for important topics and word-level timestamps...")
            
            # Get transcript text and segments
            transcript_text = transcript_data.get('text', '')
            segments = transcript_data.get('segments', [])
            
            # Analyze the entire transcript to extract main topics and context
            analysis_prompt = f"""
            Analyze this video transcript and extract the most important topics, concepts, and context for creating synchronized visual content with enhanced brand emphasis.
            
            TRANSCRIPT: {transcript_text}
            
            INSTRUCTIONS:
            1. Identify the 3-5 most important topics/concepts discussed in the video
            2. Extract the brand/company name and any related brand information (products, services, industry)
            3. Determine the overall theme and target audience
            4. Focus on topics that would benefit from visual representation
            5. Ensure topics are unique and represent different aspects of the content
            6. Pay special attention to brand mentions, company culture, and corporate identity
            7. Identify any specific products, services, or facilities mentioned
            8. Note the industry sector and business context
            
            OUTPUT FORMAT (JSON only):
            {{
                "main_topics": ["topic1", "topic2", "topic3", "topic4"],
                "key_concepts": ["concept1", "concept2"],
                "brand_company": "company name or empty string",
                "brand_industry": "industry sector (e.g., manufacturing, technology, healthcare)",
                "brand_products": ["product1", "product2"],
                "brand_facilities": ["facility1", "facility2"],
                "video_theme": "overall theme",
                "target_audience": "target audience",
                "corporate_culture": "company culture or values mentioned"
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
                print(f"✅ Script analysis completed - Found {len(analysis_data.get('main_topics', []))} main topics")
                print(f"📋 Content: {transcript_text[:200]}...")
                print(f"🏢 Brand: {analysis_data.get('brand_company', 'Not specified')}")
                
                # Create word-level timestamped scenes
                scenes = self._create_word_level_scenes(segments, analysis_data, num_images, video_context, video_duration)
                
                print(f"Topic-based content analysis completed - {len(scenes)} scenes created with natural topic alignment")
                return scenes
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed for analysis: {e}")
                print(f"Raw response: {response.choices[0].message.content[:200]}...")
                raise Exception("Failed to parse content analysis - no fallback available")
            
        except Exception as e:
            print(f"Error in topic-based scene creation: {e}")
            raise

    def _create_word_level_scenes(self, segments: list, analysis_data: dict, num_images: int, video_context: dict, video_duration: float = 60) -> list:
        """Create scenes based on word-level timestamps and topic analysis"""
        try:
            print("Creating word-level timestamped scenes with topic-based prompts...")
            
            # Extract analysis data
            main_topics = analysis_data.get('main_topics', [])
            key_concepts = analysis_data.get('key_concepts', [])
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
                # Create clean image prompt without text
                image_prompt = self._create_clean_image_prompt(
                    moment['topic'], 
                    moment['context'], 
                    brand_company, 
                    video_theme, 
                    target_audience,
                    brand_industry,
                    brand_products,
                    brand_facilities,
                    corporate_culture
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
                
                image_prompt = self._create_clean_image_prompt(
                    topic, 
                    f"Visual representation of {topic}", 
                    brand_company, 
                    video_theme, 
                    target_audience,
                    brand_industry,
                    brand_products,
                    brand_facilities,
                    corporate_culture
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
            
            return scenes
                
        except Exception as e:
            print(f"Error creating word-level scenes: {e}")
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

    def _create_clean_image_prompt(self, topic: str, context: str, brand_company: str, video_theme: str, target_audience: str, brand_industry: str, brand_products: list, brand_facilities: list, corporate_culture: str) -> str:
        """Create realistic image prompt based on actual transcript content with enhanced brand emphasis"""
        try:
            # Enhanced brand integration prompt
            brand_emphasis = ""
            brand_details = ""
            
            if brand_company and brand_company.strip():
                # Build comprehensive brand details
                brand_details_parts = []
                if brand_industry:
                    brand_details_parts.append(f"Industry: {brand_industry}")
                if brand_products:
                    brand_details_parts.append(f"Products: {', '.join(brand_products)}")
                if brand_facilities:
                    brand_details_parts.append(f"Facilities: {', '.join(brand_facilities)}")
                if corporate_culture:
                    brand_details_parts.append(f"Culture: {corporate_culture}")
                
                brand_details = "\n".join(brand_details_parts) if brand_details_parts else ""
                
                brand_emphasis = f"""
                BRAND INTEGRATION REQUIREMENTS:
                - Prominently feature {brand_company} branding in the scene
                - Include {brand_company} logos on uniforms, equipment, or signage
                - Show {brand_company} products, facilities, or corporate identity
                - Create an authentic {brand_company} workplace environment
                - Ensure {brand_company} is clearly recognizable and well-represented
                - Reflect the company's industry: {brand_industry if brand_industry else 'Professional business'}
                - Showcase company products: {', '.join(brand_products) if brand_products else 'Professional services'}
                - Highlight company facilities: {', '.join(brand_facilities) if brand_facilities else 'Modern workplace'}
                - Embody company culture: {corporate_culture if corporate_culture else 'Professional excellence'}
                """
            else:
                brand_emphasis = """
                BRAND INTEGRATION REQUIREMENTS:
                - Focus on professional workplace environments
                - Show modern business settings and equipment
                - Include subtle corporate branding elements
                - Create authentic business atmosphere
                """
            
            prompt = f"""
            Create a realistic visual representation based on the actual transcript content with enhanced brand emphasis.
            
            TOPIC: {topic}
            TRANSCRIPT CONTEXT: {context}
            BRAND/COMPANY: {brand_company}
            BRAND DETAILS: {brand_details}
            THEME: {video_theme}
            AUDIENCE: {target_audience}
            
            {brand_emphasis}
            
            Generate a DALL-E image prompt that:
            - Is PHOTOGRAPHIC and REALISTIC, not abstract or symbolic
            - Shows actual scenes, people, or objects mentioned in the transcript
            - Represents the specific content being discussed at that moment
            - Uses real-world settings and situations
            - Has natural lighting and realistic colors
            - Shows authentic business or workplace environments
            - No text overlays, no diagrams, no charts - just realistic scenes
            - Emphasizes the brand/company identity throughout the scene
            - Includes brand elements like logos, uniforms, products, or facilities
            - Creates a sense of brand pride and corporate culture
            - Shows employees engaged in brand-related activities
            - Highlights the company's values and mission through visual elements
            
            VISUAL ELEMENTS TO INCLUDE:
            - Professional workplace environment
            - Brand-consistent color schemes and design
            - Company logos and branding materials
            - Employees in branded uniforms or attire
            - Company products, equipment, or facilities
            - Modern office or manufacturing settings
            - Team collaboration and brand-focused activities
            - Quality and professionalism that reflects the brand
            
            Return only the image prompt, no explanations.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert visual designer specializing in creating brand-focused, text-free image prompts for corporate video content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"Error creating clean image prompt: {e}")
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
            num_images = len(scenes)
            print(f"Generating {num_images} images based on enhanced scene analysis...")
            
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
                
                print(f"Generating image {i+1}/{num_images} for scene: {concept[:30]}...")
                
                # Use the image_prompt directly from structured data
                image_prompt = scene.get('image_prompt', description)
                
                # Ensure the prompt specifies no text
                if "no text" not in image_prompt.lower() and "text-free" not in image_prompt.lower():
                    image_prompt += ", no text, clean design"
                
                # Log the prompt for debugging
                print(f"📝 Image Prompt for Scene {scene_id}: {image_prompt}")
                
                # Save prompt to log file for debugging
                try:
                    log_entry = f"Scene {scene_id} - {concept}: {image_prompt}\n"
                    with open(os.path.join(self.output_dir, "image_prompts.log"), "a", encoding="utf-8") as log_file:
                        log_file.write(log_entry)
                except Exception as e:
                    print(f"Warning: Could not write to log file: {e}")
                
                try:
                    response = self.openai_client.images.generate(
                        model="gpt-image-1",
                        prompt=image_prompt,
                        size="1024x1536",
                        quality="medium",
                        n=1
                    )
                    
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
                        
                        # Save image temporarily for video creation (will be cleaned up)
                        img_filename = f"{video_name}_scene_{scene_id}.jpg"
                        img_path = os.path.join(self.temp_dir, img_filename)
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        # Store image data with scene information
                        all_images[f"scene_{scene_id}"] = {
                            "path": img_path,
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
                        
                        print(f"✅ Generated scene {scene_id} ({concept})")
                        
                    else:
                        print(f"❌ No image data received for scene {scene_id}")
                        all_images[f"scene_{scene_id}"] = {"error": "No image data received"}
                    
                except Exception as e:
                    print(f"❌ Error generating image for scene {scene_id}: {e}")
                    all_images[f"scene_{scene_id}"] = {"error": str(e)}
            
            return all_images
            
        except Exception as e:
            print(f"Error in image generation: {e}")
            raise

    def extract_audio_ffmpeg(self, video_path: str, audio_path: str):
        """Extract audio from video using FFmpeg"""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'mp3', audio_path
            ]
            print(f"Running FFmpeg for audio extraction: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            raise

    def create_video_with_images(self, video_path: str, images_data: dict, output_path: str) -> str:
        """Create output video with synchronized image overlays"""
        try:
            print("Creating output video with synchronized image overlays and zoom-in effects...")
            
            # Get video duration
            video_duration = self.get_video_duration(video_path)
            print(f"Video duration: {video_duration:.2f} seconds")

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
                print("No valid images to overlay, copying original video...")
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
                scale = f"[{img_idx}:v]scale=1102:1958:force_original_aspect_ratio=increase,crop=1080:1920[imgz{img_idx}]"
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

            print(f"Running FFmpeg with {len(overlays)} overlays and simple zoom effect...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Video with {len(overlays)} image overlays created: {output_path}")
                for o in overlays:
                    print(f"🎬 Image overlaid at {o['timestamp']:.1f}s for {o['duration']:.1f}s (minimal zoom 1.02x)")
                
                # Clean up temporary image files
                self._cleanup_temp_images(temp_image_paths)
                
                return output_path
            else:
                print(f"❌ FFmpeg failed: {result.stderr}")
                print("Falling back to copying original video...")
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
        """Clean up temporary image files after video creation"""
        try:
            cleaned_count = 0
            for img_path in image_paths:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    cleaned_count += 1
            if cleaned_count > 0:
                print(f"🧹 Cleaned up {cleaned_count} temporary image files")
        except Exception as e:
            print(f"Warning: Could not clean up some temporary files: {e}")

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting video duration: {e}")
            return 0.0

    def process_video_file(self, video_path: str):
        """Process a single video file with enhanced analysis and optimized for up to 5 minutes"""
        try:
            video_name = Path(video_path).stem
            print(f"\n{'='*60}")
            print(f"🎬 PROCESSING: {video_name}")
            print(f"{'='*60}")
            
            # Get video duration first for optimization
            video_duration = self.get_video_duration(video_path)
            print(f"📏 Video duration: {video_duration:.1f} seconds ({video_duration/60:.1f} minutes)")
            
            # Estimate processing time based on duration
            estimated_images = min(int(video_duration / 60 * self.images_per_minute), self.max_images)
            estimated_time = estimated_images * 0.75  # ~45 seconds per image
            print(f"⏱️  Estimated processing time: ~{estimated_time:.0f} minutes")
            
            # Step 1: Extract audio from video using FFmpeg
            print("📺 Step 1: Extracting audio from video...")
            temp_audio_path = os.path.join(self.temp_dir, f"{video_name}_audio.mp3")
            self.extract_audio_ffmpeg(video_path, temp_audio_path)
            
            # Step 2: Transcribe audio using Whisper API with timestamps
            print("🎤 Step 2: Transcribing audio with Whisper (with timestamps)...")
            transcript_data = self.transcribe_audio(temp_audio_path)
            print(f"📝 Transcript preview: {transcript_data.get('text', '')[:150]}...")
            print(f"📊 Found {len(transcript_data.get('segments', []))} transcript segments")
            
            # Step 3: Topic detection and content analysis with audio synchronization
            print("🧠 Step 3: Topic detection and content analysis with audio synchronization...")
            scenes, video_context = self.analyze_content(transcript_data, video_duration)
            
            # Step 4: Generate high-quality images based on video duration
            num_images = len(scenes)
            print(f"🎨 Step 4: Generating {num_images} high-quality images...")
            all_images = self.generate_images(scenes, video_context, video_name=video_name)
            
            # Step 5: Create output video with synchronized audio-image overlays and fade-in effects
            print("📹 Step 5: Creating output video with synchronized audio-image overlays and fade-in effects...")
            output_video_path = os.path.join(self.output_dir, f"{video_name}_with_images.mp4")
            result = self.create_video_with_images(video_path, all_images, output_video_path)
            
            # Count successful images
            successful_images = len([k for k in all_images.keys() if 'error' not in all_images[k]])
            
            # Step 6: Skip JSON results file (as requested)
            print("📊 Step 6: Processing complete (no JSON output)")
            
            # Clean up temporary files
            try:
                os.remove(temp_audio_path)
            except:
                pass
            
            print(f"✅ SUCCESS: {video_name} ({successful_images}/{num_images} images generated)")
            print(f"{'='*60}")
            return True
            
        except Exception as e:
            print(f"❌ ERROR processing {video_path}: {e}")
            print(f"{'='*60}")
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
        print(f"✨ NEW: Minimal zoom effects (1.02x scale)")
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

def main():
    """Main function to run the video processor"""
    try:
        print("📋 Dynamic Context-Aware Video Processing Options:")
        print("1. 🔄 Watch input folder continuously (recommended)")
        print("2. ⚡ Process existing videos once and exit")
        print()
        print("✨ Features:")
        print("   • Dynamic image count based on video duration (configurable)")
        print("   • Dynamic context detection (educational, business, entertainment, etc.)")
        print("   • GPT-4o-mini topic analysis with Whisper timestamps")
        print("   • Context-adaptive image prompts (no hardcoded styles)")
        print("   • Minimal zoom effects (1.02x scale)")
        print("   • Bulk processing support")
        print("   • Optimized for videos up to 5 minutes")
        print()
        
        # Configuration options
        print("🎛️  Image Configuration Options:")
        print("   • Default: 5 images per minute, max 35, min 3")
        print("   • You can customize these settings")
        print()
        print("📋 Recommendations for 3-5 minute videos:")
        print("   • 5-7 images per minute (15-35 total images)")
        print("   • Maximum 35-40 images for good coverage")
        print("   • Logarithmic distribution for better spread")
        print("   • Optimized for up to 5-minute content")
        print()
        
        # Ask for configuration
        use_custom_config = input("Use custom image configuration? (y/n) [n]: ").strip().lower() == 'y'
        
        config = {}
        if use_custom_config:
            print("\n⚙️  Custom Configuration:")
            
            # Fixed image count option
            use_fixed = input("Use fixed image count for all videos? (y/n) [n]: ").strip().lower() == 'y'
            if use_fixed:
                try:
                    fixed_count = int(input("Enter fixed image count (1-50): "))
                    config['fixed_image_count'] = max(1, min(50, fixed_count))
                    print(f"✅ Fixed image count set to: {config['fixed_image_count']}")
                except ValueError:
                    print("❌ Invalid input, using default configuration")
        else:
                # Dynamic configuration
                try:
                    images_per_min = int(input("Images per minute (1-15) [5]: ") or "5")
                    config['images_per_minute'] = max(1, min(15, images_per_min))
                    
                    max_images = int(input("Maximum images (5-50) [35]: ") or "35")
                    config['max_images'] = max(5, min(50, max_images))
                    
                    min_images = int(input("Minimum images (1-10) [3]: ") or "3")
                    config['min_images'] = max(1, min(10, min_images))
                    
                    print(f"✅ Configuration: {config['images_per_minute']} per min, max {config['max_images']}, min {config['min_images']}")
                except ValueError:
                    print("❌ Invalid input, using default configuration")
        
        choice = input("\nSelect mode (1 or 2) [1]: ").strip() or "1"
        
        processor = LocalVideoProcessor(config=config)
        
        if choice == "1":
            print("\n🔄 Starting continuous monitoring...")
            processor.watch_input_folder(continuous=True)
        else:
            print("\n⚡ Processing existing videos once...")
            processor.watch_input_folder(continuous=False)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have set the OPENAI_API_KEY environment variable.")

if __name__ == "__main__":
    main() 