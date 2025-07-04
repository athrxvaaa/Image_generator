# Dynamic Context-Aware Video Image Generator

A Python script that intelligently processes videos and generates contextually relevant AI images with live animation effects.

## Features

- 🎬 Processes videos from input folder
- 🎨 Generates 7 AI images per minute using GPT-Image-1 (optimized for quality)
- 🧠 **Dynamic context detection** (educational, business, entertainment, news, etc.)
- 🎯 Synchronizes images with audio content using Whisper transcription with timestamps
- 🧠 GPT-4o-mini topic detection with timestamp association
- 🎨 **Context-adaptive image prompts** (no hardcoded styles - adapts to video content)
- ✨ Live animation effects (Zoom-in, Slide-in, Rotation, Scale-up)
- 🔄 Bulk processing support
- 📁 Automatic file organization

## Quick Setup

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Install FFmpeg** (required for video processing):

   - **macOS:** `brew install ffmpeg`
   - **Windows:** Download from https://ffmpeg.org/download.html
   - **Linux:** `sudo apt install ffmpeg`

3. **Set up OpenAI API key:**
   Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Add videos to process:**
   Place your video files in the `input/` folder

5. **Run the script:**
   ```bash
   python local_video_processor.py
   ```

## Usage

- **Continuous Mode (Recommended):** Watches input folder for new videos
- **One-time Mode:** Processes existing videos once and exits
- **Output:** Processed videos with animated image overlays in `output/` folder
- **Image Density:** 7 images per minute (optimized for quality over quantity)

## Supported Video Formats

- MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V

## Requirements

- Python 3.7+
- FFmpeg
- OpenAI API key with credits for GPT-4 and DALL-E 3
