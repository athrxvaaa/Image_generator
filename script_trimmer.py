#!/usr/bin/env python3
"""
Script Trimmer - Video processing with audio transcription and image insertion
Extracts audio from video, transcribes it using Whisper, analyzes with ChatGPT, and generates custom images using DALL-E
"""

import os
import json
import requests
import tempfile
import shutil
from typing import List, Dict, Optional
from pathlib import Path
import time
import hashlib
import argparse

import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
import whisper
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

class ScriptTrimmer:
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the ScriptTrimmer with OpenAI API key
        
        Args:
            openai_api_key: OpenAI API key (can also be set via OPENAI_API_KEY env var)
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            print("Warning: No OpenAI API key provided. Whisper and DALL-E will be disabled.")
        else:
            openai.api_key = self.openai_api_key
        
        # Initialize Whisper model
        self.whisper_model = None
        self._load_whisper_model()
        
        # Temporary directory for downloads
        self.temp_dir = tempfile.gettempdir()
        
    def _load_whisper_model(self):
        """Load Whisper model for speech recognition"""
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        print("Whisper model loaded successfully")
    
    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Path to extracted audio file
        """
        print(f"Extracting audio from {video_path}...")
        
        try:
            video = VideoFileClip(video_path)
            audio_path = tempfile.mktemp(suffix='.wav')
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            
            print(f"Audio extracted to: {audio_path}")
            return audio_path
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """
        Transcribe audio using Whisper to get word-by-word timestamps
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of transcription segments with timestamps
        """
        if not self.whisper_model:
            raise ValueError("Whisper model not loaded.")
        
        print("Transcribing audio with Whisper (word-by-word)...")
        
        try:
            # Transcribe with Whisper using word timestamps
            result = self.whisper_model.transcribe(audio_path, word_timestamps=True)
            
            # Extract words with timestamps
            words = []
            for segment in result['segments']:
                if 'words' in segment:
                    for word_info in segment['words']:
                        words.append({
                            'word': word_info['word'].strip(),
                            'start': word_info['start'],
                            'end': word_info['end']
                        })
            
            print(f"Transcription completed. Found {len(words)} words.")
            
            # Print complete transcription
            print("\nComplete Transcription (Word-by-Word):")
            print("=" * 50)
            for word in words:
                print(f"[{word['start']:.1f}s] {word['word']}")
            print("=" * 50)
            
            return words
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise
    
    def analyze_transcript_with_chatgpt(self, words: List[Dict]) -> List[Dict]:
        """
        Analyze transcript with ChatGPT to find words that need images at their exact spoken timestamps
        
        Args:
            words: List of words with timestamps
            
        Returns:
            List of image insertion points with words and search keywords
        """
        if not self.openai_api_key:
            print("No OpenAI API key provided. Skipping ChatGPT analysis.")
            return []
        
        print("Analyzing transcript with ChatGPT for image insertion points...")
        
        try:
            # Create a detailed transcript with exact timestamps
            transcript_with_timestamps = ""
            for word in words:
                timestamp = word['start']
                word_text = word['word']
                transcript_with_timestamps += f"[{timestamp:.1f}s] {word_text} "
            
            # Create a comprehensive prompt for ChatGPT tailored for Hospital/Healthcare Leave Policy content
            prompt = f"""
You are an expert healthcare HR content analyzer specializing in hospital employee policies and healthcare workplace benefits. I have a video transcript with EXACT timestamps about hospital leave policies and healthcare HR rules, and need to identify contextual moments that would benefit from visual illustration using AI image generation.

TRANSCRIPT WITH EXACT TIMESTAMPS:
{transcript_with_timestamps}

TASK:
1. Analyze the CONTEXT and FLOW of the conversation to identify meaningful moments that would benefit from visual illustration
2. Focus on complete concepts, scenarios, or explanations rather than individual words
3. Look for moments where visual aids would enhance understanding of hospital policies, procedures, or workplace scenarios
4. Use the EXACT timestamp when the key concept is introduced (from the transcript above)
5. Include context about what type of hospital/healthcare workplace image would best illustrate the concept being discussed
6. Be VERY SELECTIVE - only choose the most impactful contextual moments (aim for 3-5 images per video)

RULES:
- Focus on CONTEXTUAL MOMENTS where visual illustration would enhance understanding
- Look for complete concepts, policy explanations, or workplace scenarios being described
- Consider the conversation flow and what would be most helpful for viewers to see
- Skip individual words unless they represent a complete concept being explained
- Use the EXACT timestamp when the key concept is introduced
- Be VERY SELECTIVE - only choose moments that genuinely benefit from visual illustration
- Aim for 3-5 images maximum per video to avoid overwhelming the content

OUTPUT FORMAT:
Return a JSON array with objects containing:
- "word": the key concept or term being discussed (can be a phrase)
- "timestamp": the EXACT start time in seconds when the concept is introduced
- "search_keyword": context about what type of hospital/healthcare workplace image would best illustrate this concept

EXAMPLE:
[
  {{
    "word": "sick leave policy",
    "timestamp": 1.8,
    "search_keyword": "healthcare worker calling in sick, hospital workplace scenario, medical staff phone call, professional hospital environment"
  }},
  {{
    "word": "maternity leave benefits",
    "timestamp": 36.3,
    "search_keyword": "pregnant healthcare worker, hospital maternity leave, medical staff support, professional hospital environment"
  }},
  {{
    "word": "leave application process",
    "timestamp": 45.1,
    "search_keyword": "healthcare worker submitting leave request, hospital form submission, medical office desk, professional hospital setting"
  }}
]

Focus on contextual moments like:
- Policy explanations: when a complete policy or rule is being explained
- Process descriptions: when a workflow or procedure is being described
- Scenario examples: when a workplace situation or example is being discussed
- Benefit descriptions: when employee benefits or entitlements are being explained
- Time-related concepts: when calendar, scheduling, or time management is discussed
- Approval processes: when the request/approval workflow is explained

Look for moments where the speaker is:
- Explaining a complete concept or policy
- Describing a workplace scenario or example
- Outlining a process or procedure
- Discussing benefits or entitlements
- Explaining time-related rules or schedules

Only include moments that would genuinely benefit from hospital/healthcare workplace visual illustration. Quality over quantity - be very selective.
Use the EXACT timestamps from the transcript above.
"""
            
            print("Sending transcript to ChatGPT for hospital HR content analysis...")
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert healthcare HR content analyzer specializing in hospital employee policies and healthcare workplace benefits. Analyze the conversation context and flow to identify meaningful moments that would benefit from visual illustration. Provide only valid JSON output with exact timestamps from the transcript. Be very selective - aim for 3-5 contextual moments maximum per video."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                
                # Try to extract JSON from the response
                try:
                    # Find JSON array in the response
                    start_idx = content.find('[')
                    end_idx = content.rfind(']') + 1
                    
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        image_points = json.loads(json_str)
                        
                        print(f"ChatGPT hospital HR analysis completed. Found {len(image_points)} image insertion points.")
                        
                        # Print the analysis results
                        print("\nHospital HR Image Insertion Points:")
                        print("=" * 40)
                        for point in image_points:
                            print(f"[{point['timestamp']:.1f}s] '{point['word']}' -> {point['search_keyword']}")
                        print("=" * 40)
                        
                        return image_points
                    else:
                        print("No valid JSON array found in ChatGPT response")
                        return []
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing ChatGPT response as JSON: {e}")
                    print(f"Raw response: {content}")
                    return []
            else:
                print("No response from ChatGPT")
                return []
                
        except Exception as e:
            print(f"Error analyzing transcript with ChatGPT: {e}")
            return []
    
    def generate_image_with_dalle(self, word: str, context: str, is_portrait: bool = True) -> Optional[str]:
        """
        Generate custom image using DALL-E based on word and context for hospital/healthcare content
        
        Args:
            word: The word being spoken
            context: Context from the script analysis
            is_portrait: Whether the video is portrait orientation
            
        Returns:
            Path to generated image or None if failed
        """
        if not self.openai_api_key:
            print("No OpenAI API key provided. Skipping image generation.")
            return None
        
        print(f"  Generating hospital image for: '{word}'")
        
        try:
            # Create a detailed prompt for DALL-E based on context
            aspect_ratio = "9:16" if is_portrait else "16:9"
            
            # Use the context provided by ChatGPT to create a more specific prompt
            if context and any(keyword in context.lower() for keyword in ['hospital', 'healthcare', 'medical', 'nurse', 'doctor']):
                # Use the context directly if it's already well-formed for healthcare content
                prompt = f"{context}, professional hospital setting, clean modern medical facility, {aspect_ratio} aspect ratio, high quality, detailed"
            else:
                # Build context-aware prompt based on the word and context for hospital/healthcare content
                word_lower = word.lower()
                
                # Leave types
                if word_lower in ['sick', 'illness', 'ill']:
                    prompt = f"Healthcare worker calling in sick, hospital workplace scenario, medical staff phone call, clean modern hospital environment, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['vacation', 'holiday', 'time off']:
                    prompt = f"Healthcare worker on vacation, medical staff benefits, professional hospital setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['maternity', 'pregnancy', 'pregnant']:
                    prompt = f"Pregnant healthcare worker, hospital maternity leave, medical staff support, professional hospital environment, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['paternity', 'father', 'dad']:
                    prompt = f"New father healthcare worker, hospital paternity leave, medical staff family support, professional hospital setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['personal', 'personal leave']:
                    prompt = f"Healthcare worker taking personal time, hospital personal leave scenario, medical staff office setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['bereavement', 'funeral', 'grief']:
                    prompt = f"Healthcare worker dealing with bereavement, hospital support scenario, professional compassionate medical environment, {aspect_ratio} aspect ratio, high quality"
                
                # Hospital workplace concepts
                elif word_lower in ['policy', 'policies', 'rules']:
                    prompt = f"Hospital HR policy document, healthcare workplace rules and regulations, medical facility setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['approval', 'approve', 'approved']:
                    prompt = f"Hospital manager approving leave request, healthcare approval process, medical office meeting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['request', 'submission', 'submit']:
                    prompt = f"Healthcare worker submitting leave request, hospital form submission, medical office desk, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['denial', 'denied', 'reject']:
                    prompt = f"Hospital leave request denied scenario, healthcare communication, medical office setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['review', 'reviewing', 'evaluation']:
                    prompt = f"Hospital HR reviewing leave request, healthcare evaluation process, medical office environment, {aspect_ratio} aspect ratio, high quality"
                
                # Time-related
                elif word_lower in ['days', 'day']:
                    prompt = f"Calendar with days marked, hospital time tracking, medical office setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['weeks', 'week']:
                    prompt = f"Weekly calendar or schedule, hospital time management, medical office environment, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['months', 'month']:
                    prompt = f"Monthly calendar view, hospital planning, medical office setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['hours', 'hour']:
                    prompt = f"Clock or time tracking, hospital hours management, medical office environment, {aspect_ratio} aspect ratio, high quality"
                
                # Hospital scenarios
                elif word_lower in ['office', 'workplace', 'work']:
                    prompt = f"Modern professional hospital office environment, healthcare workplace setting, clean medical office space, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['meeting', 'conference']:
                    prompt = f"Professional hospital meeting, healthcare conference room, medical staff meeting scenario, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['desk', 'workstation']:
                    prompt = f"Professional hospital desk, healthcare workstation, clean modern medical desk setup, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['computer', 'laptop']:
                    prompt = f"Healthcare worker using computer, hospital technology, medical office setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['phone', 'call']:
                    prompt = f"Healthcare worker on phone call, hospital communication, medical office setting, {aspect_ratio} aspect ratio, high quality"
                
                # Healthcare staff concepts
                elif word_lower in ['employee', 'staff', 'worker']:
                    prompt = f"Professional healthcare worker in hospital, medical staff, professional healthcare setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['manager', 'supervisor', 'boss']:
                    prompt = f"Professional hospital manager, healthcare leadership scenario, medical office environment, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['hr', 'human resources']:
                    prompt = f"Hospital HR department, healthcare human resources office, medical facility setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['benefits', 'compensation']:
                    prompt = f"Healthcare worker benefits illustration, hospital compensation, medical office setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['nurse', 'nursing']:
                    prompt = f"Professional nurse in hospital, healthcare nursing staff, medical facility environment, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['doctor', 'physician']:
                    prompt = f"Professional doctor in hospital, healthcare physician, medical facility setting, {aspect_ratio} aspect ratio, high quality"
                
                # Process-related
                elif word_lower in ['application', 'form', 'document']:
                    prompt = f"Hospital leave application form, healthcare document, medical office desk, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['notification', 'notice', 'inform']:
                    prompt = f"Hospital notification, healthcare worker communication, medical office setting, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['confirmation', 'confirm']:
                    prompt = f"Hospital leave confirmation, healthcare approval notification, medical office environment, {aspect_ratio} aspect ratio, high quality"
                
                # Hospital-specific terms
                elif word_lower in ['hospital', 'medical', 'healthcare']:
                    prompt = f"Modern hospital facility, healthcare environment, medical building, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['department', 'unit']:
                    prompt = f"Hospital department, healthcare unit, medical facility section, {aspect_ratio} aspect ratio, high quality"
                elif word_lower in ['grade', 'level']:
                    prompt = f"Hospital staff grade levels, healthcare hierarchy, medical staff structure, {aspect_ratio} aspect ratio, high quality"
                
                else:
                    # Generic prompt for other words, incorporating context if available
                    if context:
                        prompt = f"{context}, professional hospital setting, clean modern medical facility, {aspect_ratio} aspect ratio, high quality, detailed"
                    else:
                        prompt = f"Professional hospital illustration of {word}, modern medical facility environment, {aspect_ratio} aspect ratio, high quality, detailed"
            
            print(f"    Generating with hospital prompt: '{prompt}'")
            
            # Generate image with DALL-E
            response = openai.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",  # DALL-E 3 uses square images
                quality="standard",
                n=1
            )
            
            if response.data and len(response.data) > 0:
                image_url = response.data[0].url
                
                # Download the generated image
                image_filename = f"dalle_{word.replace(' ', '_')}_{int(time.time())}.png"
                image_path = os.path.join(self.temp_dir, image_filename)
                
                print(f"    Downloading generated hospital image...")
                
                img_response = requests.get(image_url, timeout=15)
                if img_response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    print(f"    ✓ Generated hospital image for '{word}'")
                    return image_path
                else:
                    print(f"    Failed to download image: {img_response.status_code}")
            else:
                print(f"    No image generated for '{word}'")
            
            return None
            
        except Exception as e:
            print(f"    Error generating hospital image with DALL-E: {e}")
            return None
    
    def insert_images_in_video(self, video_path: str, image_points: List[Dict], 
                              output_path: str) -> str:
        """
        Insert DALL-E generated images into video at exact timestamps
        
        Args:
            video_path: Path to input video
            image_points: List of image insertion points with exact timestamps
            output_path: Path for output video
            
        Returns:
            Path to final video with inserted images
        """
        print(f"Inserting {len(image_points)} images into video...")
        
        try:
            # Load the video
            video = VideoFileClip(video_path)
            video_duration = video.duration
            video_size = video.size
            is_portrait = video_size[1] > video_size[0]
            
            print(f"Video properties: {video_size[0]}x{video_size[1]}, {video_duration:.1f}s duration")
            
            # Filter image points to ensure they're within video duration
            filtered_points = []
            for point in image_points:
                timestamp = point['timestamp']
                if 0 <= timestamp < video_duration:
                    filtered_points.append(point)
                else:
                    print(f"  Skipping image at {timestamp}s (outside video duration)")
            
            if not filtered_points:
                print("No valid image insertion points found.")
                return video_path
            
            print(f"Inserting {len(filtered_points)} images at exact timestamps...")
            
            # Create clips list starting with the original video
            clips = [video]
            
            # Insert images at exact timestamps
            for i, point in enumerate(filtered_points):
                timestamp = point['timestamp']
                word = point['word']
                
                print(f"  [{i+1}/{len(filtered_points)}] Inserting image for '{word}' at {timestamp:.1f}s")
                
                # Generate image using DALL-E
                image_path = self.generate_image_with_dalle(
                    point['word'], 
                    point['search_keyword'], 
                    is_portrait
                )
                
                if image_path and os.path.exists(image_path):
                    try:
                        # Create image clip
                        image_clip = ImageClip(image_path)
                        
                        # Resize image to fill entire screen while maintaining aspect ratio
                        if is_portrait:
                            # For portrait videos, resize to fill entire screen while maintaining aspect ratio
                            video_ratio = video_size[0] / video_size[1]  # width/height
                            image_ratio = image_clip.w / image_clip.h
                            
                            if image_ratio > video_ratio:
                                # Image is wider than video, fit to height and crop width
                                target_height = video_size[1]
                                target_width = int(image_clip.w * target_height / image_clip.h)
                                image_clip = image_clip.resize((target_width, target_height))
                                
                                # Crop from center to fit video width
                                crop_x = (target_width - video_size[0]) // 2
                                image_clip = image_clip.crop(x1=crop_x, y1=0, x2=crop_x + video_size[0], y2=target_height)
                            else:
                                # Image is taller than video, fit to width and crop height
                                target_width = video_size[0]
                                target_height = int(image_clip.h * target_width / image_clip.w)
                                image_clip = image_clip.resize((target_width, target_height))
                                
                                # Crop from center to fit video height
                                crop_y = (target_height - video_size[1]) // 2
                                image_clip = image_clip.crop(x1=0, y1=crop_y, x2=target_width, y2=crop_y + video_size[1])
                        else:
                            # For landscape videos, fit to height
                            target_height = video_size[1]
                            target_width = int(image_clip.w * target_height / image_clip.h)
                            
                            image_clip = image_clip.resize((target_width, target_height))
                            
                            # Center the image
                            x_center = (video_size[0] - target_width) // 2
                            y_center = (video_size[1] - target_height) // 2
                            image_clip = image_clip.set_position((x_center, y_center))
                        
                        # Set duration and timing with subtle transitions
                        image_duration = 3.0  # Show image for 3 seconds
                        fade_duration = 0.5  # 0.5 second fade in/out
                        
                        # Create fade-in and fade-out effects
                        image_clip = image_clip.set_duration(image_duration)
                        image_clip = image_clip.set_start(timestamp)
                        
                        # Add fade-in effect (0.5s fade in)
                        image_clip = image_clip.fadein(fade_duration)
                        
                        # Add fade-out effect (0.5s fade out at the end)
                        image_clip = image_clip.fadeout(fade_duration)
                        
                        # Add to clips list
                        clips.append(image_clip)
                        
                        print(f"    ✓ Image inserted for '{word}' at {timestamp:.1f}s")
                        
                    except Exception as e:
                        print(f"    Error creating image clip for '{word}': {e}")
                else:
                    print(f"    ✗ Failed to generate image for '{word}'")
            
            # Composite all clips
            print("Compositing video with images...")
            final_video = CompositeVideoClip(clips)
            
            # Write the final video
            print(f"Writing final video to: {output_path}")
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Clean up
            final_video.close()
            video.close()
            
            print(f"✓ Video processing completed successfully!")
            return output_path
            
        except Exception as e:
            print(f"Error inserting images into video: {e}")
            raise
    
    def cleanup_all_images(self):
        """Remove all downloaded images from cache and temporary directories"""
        print("🧹 Cleaning up all generated images...")
        
        # Clean up temporary images in temp directory
        temp_files_removed = 0
        for filename in os.listdir(self.temp_dir):
            if filename.startswith('dalle_') and filename.endswith('.png'):
                file_path = os.path.join(self.temp_dir, filename)
                try:
                    os.remove(file_path)
                    temp_files_removed += 1
                except Exception as e:
                    print(f"❌ Error removing {filename}: {e}")
        
        if temp_files_removed > 0:
            print(f"✅ Removed {temp_files_removed} generated image files")
        else:
            print("ℹ️  No generated image files found")
        
        print("✅ All generated images cleaned up successfully!")
    
    def process_video(self, input_video: str, output_video: Optional[str] = None) -> str:
        """
        Process video: extract audio, transcribe, analyze with ChatGPT, and insert DALL-E generated images
        
        Args:
            input_video: Path to input video file
            output_video: Path to output video file (auto-generated if None)
            
        Returns:
            Path to processed video file
        """
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video not found: {input_video}")
        
        # Generate output path if not provided
        if not output_video:
            input_path = Path(input_video)
            output_video = str(input_path.parent / f"{input_path.stem}_with_images{input_path.suffix}")
        
        print(f"Processing video: {input_video}")
        print(f"Output will be saved to: {output_video}")
        
        try:
            # Step 1: Extract audio
            audio_path = self.extract_audio(input_video)
            
            # Step 2: Transcribe audio with word-by-word timestamps
            words = self.transcribe_audio(audio_path)
            
            if not words:
                print("No words transcribed. Cannot proceed.")
                return input_video
            
            # Step 3: Get video properties
            video = VideoFileClip(input_video)
            video_duration = video.duration
            video_size = video.size
            is_portrait = video_size[1] > video_size[0]  # Height > Width
            video.close()
            
            print(f"Video duration: {video_duration:.1f} seconds")
            print(f"Video size: {video_size[0]}x{video_size[1]} ({'Portrait' if is_portrait else 'Landscape'})")
            
            # Step 4: Analyze with ChatGPT for image insertion points
            image_points = self.analyze_transcript_with_chatgpt(words)
            
            if not image_points:
                print("No image insertion points identified. Creating video without images.")
                return input_video
            
            # Step 5: Insert DALL-E generated images
            final_video_path = self.insert_images_in_video(input_video, image_points, output_video)
            
            # Clean up temporary audio file
            try:
                os.remove(audio_path)
            except:
                pass
            
            print(f"\n✓ Video processing completed successfully!")
            print(f"Final video: {final_video_path}")
            
            return final_video_path
            
        except Exception as e:
            print(f"Error processing video: {e}")
            raise


def main():
    """Main function to handle command line arguments and process video"""
    parser = argparse.ArgumentParser(
        description="Process video with audio transcription and DALL-E image insertion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script_trimmer.py video.mp4
  python script_trimmer.py video.mp4 -o output.mp4
  python script_trimmer.py --cleanup-images
        """
    )
    
    parser.add_argument("input_video", nargs='?', help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output video path (auto-generated if not specified)")
    parser.add_argument("--cleanup-images", action="store_true", 
                       help="Remove all downloaded images and exit")
    
    args = parser.parse_args()
    
    # Handle cleanup option
    if args.cleanup_images:
        trimmer = ScriptTrimmer()
        trimmer.cleanup_all_images()
        return
    
    # Check if input video is provided
    if not args.input_video:
        parser.error("Input video file is required (unless using --cleanup-images)")
    
    # Initialize ScriptTrimmer
    trimmer = ScriptTrimmer()
    
    # Process the video
    try:
        output_path = trimmer.process_video(args.input_video, args.output)
        print(f"\n🎉 Success! Processed video saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 