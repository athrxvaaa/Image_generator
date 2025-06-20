# Script Trimmer

A Python script that processes videos by extracting audio, transcribing it using Whisper, analyzing the content with ChatGPT, and inserting custom-generated images using DALL-E.

## Features

- **Audio Extraction**: Extracts audio from video files
- **Speech Recognition**: Uses OpenAI's Whisper for accurate word-by-word transcription with timestamps
- **Content Analysis**: Leverages ChatGPT to identify educational keywords that would benefit from visual illustration
- **Custom Image Generation**: Uses DALL-E to create educational images that perfectly match the script content
- **Smart Image Insertion**: Automatically inserts relevant images at appropriate timestamps
- **Image Caching**: Caches generated images to avoid regenerating the same content
- **Portrait/Landscape Support**: Automatically detects video orientation and adjusts image generation accordingly

## Requirements

- Python 3.8+
- OpenAI API key (for Whisper, ChatGPT, and DALL-E)
- FFmpeg (for video processing)

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install FFmpeg (if not already installed):
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`
   - **Windows**: Download from https://ffmpeg.org/download.html

## Setup

1. Get an OpenAI API key from https://platform.openai.com/api-keys
2. Set your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file in the project directory:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Basic Usage

```bash
python script_trimmer.py input_video.mp4
```

### Specify Output File

```bash
python script_trimmer.py input_video.mp4 -o output_video.mp4
```

### Clean Up Generated Images

```bash
python script_trimmer.py --cleanup-images
```

## How It Works

1. **Audio Extraction**: Extracts audio from the input video
2. **Transcription**: Uses Whisper to transcribe audio with word-by-word timestamps
3. **Content Analysis**: ChatGPT analyzes the transcript to identify educational keywords that would benefit from visual illustration
4. **Image Generation**: DALL-E creates custom educational images based on the identified keywords and context
5. **Video Processing**: Inserts the generated images at appropriate timestamps in the video
6. **Output**: Saves the enhanced video with educational images

## Example

For an educational video about the nervous system:

- Transcribes words like "brain", "neurons", "signals"
- ChatGPT identifies these as educational keywords
- DALL-E generates anatomical diagrams and scientific illustrations
- Images are inserted when these words are spoken

## Notes

- The script automatically detects video orientation (portrait/landscape)
- Generated images are cached to avoid regenerating the same content
- Processing time depends on video length and number of images generated
- DALL-E image generation requires OpenAI API credits

## Troubleshooting

- **No images inserted**: Check your OpenAI API key and ensure you have sufficient credits
- **Poor transcription**: Ensure clear audio quality in the input video
- **Memory issues**: For large videos, consider processing in smaller segments

## Dependencies

- `moviepy`: Video processing and editing
- `vosk`: Offline speech recognition
- `requests`: HTTP requests for Pexels API
- `Pillow`: Image processing
- `opencv-python`: Computer vision operations
- `numpy`: Numerical operations
- `python-dotenv`: Environment variable management

## License

This project is open source. Feel free to modify and distribute.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on the repository.
