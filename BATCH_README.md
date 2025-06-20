# Batch Video Processing

This system allows you to process multiple videos automatically by placing them in the `input_videos` folder and running the batch processor.

## Quick Start

1. **Place your videos** in the `input_videos` folder
2. **Run the batch processor**:
   ```bash
   python batch_processor.py
   ```
3. **Find processed videos** in the `output_videos` folder

## Folder Structure

```
Script_trimmer/
├── input_videos/          # Put your videos here
│   ├── video1.mp4
│   ├── video2.avi
│   └── ...
├── output_videos/         # Processed videos appear here
│   ├── video1_with_images.mp4
│   ├── video2_with_images.avi
│   └── ...
├── script_trimmer.py      # Main processing script
├── batch_processor.py     # Batch processing script
└── ...
```

## Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)
- M4V (.m4v)

## Usage Options

### Basic Batch Processing

```bash
python batch_processor.py
```

Processes all videos in `input_videos` folder and saves results to `output_videos` folder.

### Skip Existing Files

```bash
python batch_processor.py --skip-existing
```

Skips videos that already have output files (useful for resuming interrupted processing).

### Custom Folders

```bash
python batch_processor.py --input my_videos --output my_results
```

Use custom input and output folders.

### Clean Up Temporary Files

```bash
python batch_processor.py --cleanup
```

Removes all temporary files and generated images.

## How It Works

1. **Scans Input Folder**: Finds all supported video files
2. **Shows Preview**: Lists all videos that will be processed
3. **Confirms Action**: Asks for confirmation before starting
4. **Processes Each Video**:
   - Extracts audio
   - Transcribes with Whisper (word-by-word)
   - Analyzes with ChatGPT for image insertion points
   - Generates images with DALL-E
   - Inserts images into video
   - Saves to output folder
5. **Provides Summary**: Shows success/failure statistics

## Features

- ✅ **Automatic Processing**: No manual intervention needed
- ✅ **Progress Tracking**: Shows current video and progress
- ✅ **Error Handling**: Continues processing even if one video fails
- ✅ **Skip Existing**: Option to skip already processed videos
- ✅ **Custom Folders**: Use any input/output folder names
- ✅ **Cleanup**: Remove temporary files when done
- ✅ **Multiple Formats**: Supports 8 different video formats

## Example Output

```
📁 Found 3 video(s) in 'input_videos'
📁 Output will be saved to 'output_videos'
🎯 Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v

📋 Videos to process:
  1. tutorial.mp4 🔄 (will process)
  2. lecture.avi 🔄 (will process)
  3. presentation.mov 🔄 (will process)

⚠️  This will process 3 video(s).
Do you want to continue? (Y/n): Y

============================================================
🎬 Processing Video 1/3: tutorial.mp4
============================================================
✅ Successfully processed: tutorial.mp4
⏱️  Processing time: 45.2 seconds
📁 Output saved to: tutorial_with_images.mp4

============================================================
📊 BATCH PROCESSING COMPLETE
============================================================
✅ Successfully processed: 3
❌ Failed: 0
⏭️  Skipped: 0
⏱️  Total time: 142.8 seconds
📁 Output folder: output_videos

🎉 Successfully processed videos are available in: output_videos
```

## Tips

1. **Large Videos**: Processing time depends on video length and complexity
2. **API Limits**: Be aware of OpenAI API rate limits for large batches
3. **Storage**: Ensure you have enough disk space for processed videos
4. **Interruption**: You can safely interrupt with Ctrl+C and resume later
5. **Cleanup**: Run cleanup periodically to free up disk space

## Troubleshooting

- **No videos found**: Check that videos are in the correct folder and format
- **API errors**: Verify your OpenAI API key is set correctly
- **Memory issues**: Process videos one at a time for very large files
- **Permission errors**: Ensure write permissions for output folder
