#!/usr/bin/env python3
"""
Batch Video Processor - Process multiple videos with audio transcription and image insertion
Automatically processes all videos in input_videos folder and saves results to output_videos folder
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional
import argparse

# Import the ScriptTrimmer class from the main script
from script_trimmer import ScriptTrimmer

class BatchVideoProcessor:
    def __init__(self, input_folder: str = "input_videos", output_folder: str = "output_videos"):
        """
        Initialize the batch processor
        
        Args:
            input_folder: Folder containing input videos
            output_folder: Folder to save processed videos
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.trimmer = ScriptTrimmer()
        
        # Supported video formats
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        # Create folders if they don't exist
        self.input_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
    
    def get_video_files(self) -> List[Path]:
        """
        Get all supported video files from the input folder
        
        Returns:
            List of video file paths
        """
        video_files = []
        
        if not self.input_folder.exists():
            print(f"❌ Input folder '{self.input_folder}' does not exist!")
            return video_files
        
        for file_path in self.input_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                video_files.append(file_path)
        
        return sorted(video_files)
    
    def process_single_video(self, input_video: Path, video_number: int, total_videos: int) -> bool:
        """
        Process a single video file
        
        Args:
            input_video: Path to input video file
            video_number: Current video number (for progress display)
            total_videos: Total number of videos to process
            
        Returns:
            True if processing was successful, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"🎬 Processing Video {video_number}/{total_videos}: {input_video.name}")
        print(f"{'='*60}")
        
        try:
            # Generate output path
            output_video = self.output_folder / f"{input_video.stem}_with_images{input_video.suffix}"
            
            # Check if output already exists
            if output_video.exists():
                print(f"⚠️  Output file already exists: {output_video.name}")
                response = input("Do you want to overwrite it? (y/N): ").strip().lower()
                if response != 'y':
                    print(f"⏭️  Skipping {input_video.name}")
                    return True
            
            # Process the video
            start_time = time.time()
            result_path = self.trimmer.process_video(str(input_video), str(output_video))
            processing_time = time.time() - start_time
            
            if result_path and os.path.exists(result_path):
                print(f"✅ Successfully processed: {input_video.name}")
                print(f"⏱️  Processing time: {processing_time:.1f} seconds")
                print(f"📁 Output saved to: {output_video.name}")
                return True
            else:
                print(f"❌ Failed to process: {input_video.name}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing {input_video.name}: {e}")
            return False
    
    def process_all_videos(self, skip_existing: bool = False) -> dict:
        """
        Process all videos in the input folder
        
        Args:
            skip_existing: Skip videos that already have output files
            
        Returns:
            Dictionary with processing results
        """
        video_files = self.get_video_files()
        
        if not video_files:
            print(f"❌ No video files found in '{self.input_folder}'")
            print(f"Supported formats: {', '.join(self.supported_formats)}")
            return {"success": 0, "failed": 0, "skipped": 0}
        
        print(f"📁 Found {len(video_files)} video(s) in '{self.input_folder}'")
        print(f"📁 Output will be saved to '{self.output_folder}'")
        print(f"🎯 Supported formats: {', '.join(self.supported_formats)}")
        
        # Show list of videos to be processed
        print(f"\n📋 Videos to process:")
        for i, video in enumerate(video_files, 1):
            output_path = self.output_folder / f"{video.stem}_with_images{video.suffix}"
            status = "⏭️  (will skip)" if skip_existing and output_path.exists() else "🔄 (will process)"
            print(f"  {i}. {video.name} {status}")
        
        # Confirm before processing
        print(f"\n⚠️  This will process {len(video_files)} video(s).")
        response = input("Do you want to continue? (Y/n): ").strip().lower()
        if response == 'n':
            print("❌ Processing cancelled.")
            return {"success": 0, "failed": 0, "skipped": 0}
        
        # Process videos
        results = {"success": 0, "failed": 0, "skipped": 0}
        start_time = time.time()
        
        for i, video in enumerate(video_files, 1):
            output_path = self.output_folder / f"{video.stem}_with_images{video.suffix}"
            
            # Skip if output exists and skip_existing is True
            if skip_existing and output_path.exists():
                print(f"⏭️  Skipping {video.name} (output already exists)")
                results["skipped"] += 1
                continue
            
            # Process the video
            if self.process_single_video(video, i, len(video_files)):
                results["success"] += 1
            else:
                results["failed"] += 1
            
            # Add a small delay between videos to prevent resource conflicts
            if i < len(video_files):
                print("⏳ Waiting 2 seconds before next video...")
                time.sleep(2)
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successfully processed: {results['success']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"⏭️  Skipped: {results['skipped']}")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"📁 Output folder: {self.output_folder}")
        
        if results['success'] > 0:
            print(f"\n🎉 Successfully processed videos are available in: {self.output_folder}")
        
        return results
    
    def cleanup(self):
        """Clean up temporary files and images"""
        print("🧹 Cleaning up temporary files...")
        self.trimmer.cleanup_all_images()


def main():
    """Main function to handle command line arguments and batch processing"""
    parser = argparse.ArgumentParser(
        description="Batch process videos with audio transcription and image insertion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_processor.py                    # Process all videos in input_videos folder
  python batch_processor.py --skip-existing   # Skip videos that already have output files
  python batch_processor.py --input custom_input --output custom_output
  python batch_processor.py --cleanup         # Clean up temporary files
        """
    )
    
    parser.add_argument("--input", default="input_videos", 
                       help="Input folder containing videos (default: input_videos)")
    parser.add_argument("--output", default="output_videos", 
                       help="Output folder for processed videos (default: output_videos)")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip videos that already have output files")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up temporary files and exit")
    
    args = parser.parse_args()
    
    # Handle cleanup option
    if args.cleanup:
        processor = BatchVideoProcessor()
        processor.cleanup()
        return 0
    
    # Initialize batch processor
    processor = BatchVideoProcessor(args.input, args.output)
    
    # Process all videos
    try:
        results = processor.process_all_videos(args.skip_existing)
        
        if results["failed"] > 0:
            return 1
        return 0
        
    except KeyboardInterrupt:
        print("\n\n❌ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 