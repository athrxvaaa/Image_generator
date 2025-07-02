# Improved Video Image Generator - Segmented Approach

## How It Works Now

The improved system now divides the transcript into meaningful segments and generates specific images for each segment based on its content and keywords.

## Example Workflow

### Input: Video Transcript

```
"I'm cooking pasta in my kitchen. The water is boiling and I'm adding salt. Now I'm chopping fresh vegetables for the sauce. The tomatoes are bright red and the basil smells amazing."
```

### Step 1: Transcript Division

The system divides this into segments:

1. "I'm cooking pasta in my kitchen. The water is boiling and I'm adding salt."
2. "Now I'm chopping fresh vegetables for the sauce."
3. "The tomatoes are bright red and the basil smells amazing."

### Step 2: Segment Analysis

For each segment, the system extracts:

- **Visual description**: What the scene looks like
- **Keywords**: Key visual elements
- **Mood**: Atmosphere of the scene

**Segment 1 Analysis:**

- Description: "A person standing at a stove with a pot of boiling water, adding salt to pasta"
- Keywords: ["cooking", "pasta", "boiling water", "salt", "stove"]
- Mood: "focused, cooking atmosphere"

**Segment 2 Analysis:**

- Description: "Someone chopping colorful vegetables on a cutting board"
- Keywords: ["chopping", "vegetables", "cutting board", "knife"]
- Mood: "precise, culinary preparation"

**Segment 3 Analysis:**

- Description: "Bright red tomatoes and fresh green basil leaves"
- Keywords: ["tomatoes", "red", "basil", "fresh", "green"]
- Mood: "fresh, vibrant, aromatic"

### Step 3: Image Generation

The system generates specific images for each segment:

- **Image 1**: A person cooking pasta at a stove with boiling water
- **Image 2**: Someone chopping vegetables on a cutting board
- **Image 3**: Close-up of bright red tomatoes and fresh basil

## Benefits of This Approach

1. **Contextual Relevance**: Each image directly relates to specific content in the transcript
2. **Better Storytelling**: Images follow the narrative flow of the video
3. **More Meaningful**: Instead of 3 random variations, you get 3 different scenes
4. **Keyword-Driven**: Images are generated based on specific keywords from each segment
5. **Mood-Aware**: Each image captures the appropriate atmosphere

## API Usage

```bash
# Process video with 1 image per segment (default)
curl -X POST "http://localhost:8000/process-video" \
  -F "video_file=@your_video.mp4" \
  -F "num_images_per_segment=1"

# Process video with 2 images per segment
curl -X POST "http://localhost:8000/process-video" \
  -F "video_file=@your_video.mp4" \
  -F "num_images_per_segment=2"
```

## Output Structure

The results now include detailed segment information:

```json
{
  "task_id": "uuid-here",
  "transcript": "full transcript text",
  "segments": [
    {
      "segment_id": 1,
      "description": "visual description",
      "keywords": ["keyword1", "keyword2"],
      "mood": "mood description",
      "sentence": "original sentence"
    }
  ],
  "all_images": {
    "segment_1": {
      "images": [
        {
          "url": "s3-url-here",
          "description": "visual description",
          "keywords": ["keyword1", "keyword2"],
          "mood": "mood description"
        }
      ],
      "segment_data": {
        /* segment info */
      }
    }
  },
  "total_segments": 3,
  "total_images_generated": 3,
  "timestamp": "2024-01-01T12:00:00"
}
```

## File Naming Convention

Images are now named with segment information:

- `{task_id}_segment_1_image_1.png` - First image from segment 1
- `{task_id}_segment_1_image_2.png` - Second image from segment 1 (if multiple per segment)
- `{task_id}_segment_2_image_1.png` - First image from segment 2

This makes it easy to understand which image corresponds to which part of the transcript.
