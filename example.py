#!/usr/bin/env python3
"""
Example usage of Script Trimmer
Demonstrates how to use the ScriptTrimmer class programmatically
"""

from script_trimmer import ScriptTrimmer
import os

def main():
    """Example usage of ScriptTrimmer"""
    
    # Initialize ScriptTrimmer
    # You can pass your Pexels API key here or set it as environment variable
    pexels_api_key = os.getenv('PEXELS_API_KEY')  # or "your_api_key_here"
    trimmer = ScriptTrimmer(pexels_api_key=pexels_api_key)
    
    # Example video processing
    input_video = "example_video.mp4"  # Replace with your video file
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        print("Please provide a valid video file path")
        return
    
    try:
        print("🎬 Starting video processing...")
        
        # Process the video
        output_video = trimmer.process_video(
            input_video=input_video,
            output_video="enhanced_video.mp4",  # Custom output name
            max_images=3,  # Limit to 3 images
            keep_audio=True  # Keep original audio
        )
        
        print(f"✅ Video processing completed!")
        print(f"📁 Output saved as: {output_video}")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have a Vosk model downloaded")
        print("2. Ensure FFmpeg is installed")
        print("3. Check that the input video has audio")
        print("4. Verify your Pexels API key if using image insertion")

def example_with_custom_keywords():
    """Example showing how to customize keywords"""
    
    trimmer = ScriptTrimmer()
    
    # Add custom keywords
    trimmer.keywords.update({
        'education': ['learning', 'study', 'school', 'university', 'course'],
        'health': ['medical', 'doctor', 'hospital', 'medicine', 'healthcare'],
        'entertainment': ['movie', 'film', 'show', 'entertainment', 'fun']
    })
    
    print("Custom keywords added:")
    for category, keywords in trimmer.keywords.items():
        print(f"  {category}: {', '.join(keywords)}")

if __name__ == "__main__":
    print("🎯 Script Trimmer Example")
    print("=" * 40)
    
    # Show custom keywords example
    example_with_custom_keywords()
    print()
    
    # Run main example
    main() 