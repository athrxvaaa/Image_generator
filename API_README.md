# Video Image Generator API Documentation

A RESTful API for processing videos: transcribes audio, analyzes content, generates contextual images, and creates enhanced videos with inserted images. Built with FastAPI.

---

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### 1. Health Check

**GET** `/health`

- **Description:** Check if the API is running and healthy.
- **Response:**
  - `200 OK`: `{ "status": "healthy", "timestamp": "2024-06-30T12:34:56.789Z" }`

---

### 2. Upload Video

**POST** `/upload-video`

- **Description:** Upload a video file for processing. Starts background processing and returns a task ID.
- **Request:**
  - `multipart/form-data`
  - **Parameters:**
    - `file` (required): Video file (e.g., `.mp4`, `.mov`)
    - `generate_images` (optional, bool): Whether to generate images (default: true)
    - `image_interval` (optional, float): Seconds between images
    - `max_images` (optional, int): Maximum number of images to generate
- **Response:**

  - `200 OK`:
    ```json
    {
      "task_id": "string",
      "status": "uploaded",
      "message": "Video uploaded and processing started",
      "progress": 0.0
    }
    ```
  - `400 Bad Request`: Invalid file type

- **Example (curl):**
  ```bash
  curl -X POST http://localhost:8000/upload-video \
    -F "file=@/path/to/video.mp4"
  ```

---

### 3. Check Task Status

**GET** `/status/{task_id}`

- **Description:** Get the processing status of a video task.
- **Path Parameter:**
  - `task_id` (string): The ID returned by `/upload-video`
- **Response:**

  - `200 OK`:
    ```json
    {
      "task_id": "string",
      "status": "processing|completed|error|uploaded",
      "message": "string",
      "progress": 0.0-1.0,
      "download_url": "/download/{task_id}",
      "s3_url": "https://..." // if S3 enabled
    }
    ```
  - `404 Not Found`: Invalid or unknown task ID

- **Example (curl):**
  ```bash
  curl http://localhost:8000/status/your-task-id
  ```

---

### 4. Download Processed Video

**GET** `/download/{task_id}`

- **Description:** Download the processed/enhanced video for a completed task.
- **Path Parameter:**
  - `task_id` (string): The ID returned by `/upload-video`
- **Response:**

  - `200 OK`: Returns the video file as an attachment
  - `404 Not Found`: If the video is not ready or task ID is invalid

- **Example (curl):**
  ```bash
  curl -O http://localhost:8000/download/your-task-id
  ```

---

## Error Responses

- `400 Bad Request`: Invalid input (e.g., non-video file)
- `404 Not Found`: Task not found or video not ready
- `500 Internal Server Error`: Unexpected server error

---

## Example Workflow

1. **Upload a video:**
   - `POST /upload-video` with your video file
   - Receive a `task_id`
2. **Check status:**
   - `GET /status/{task_id}`
   - Poll until `status` is `completed`
3. **Download video:**
   - `GET /download/{task_id}`

---

## OpenAPI/Swagger UI

- Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Alternative docs: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Environment Variables

See `.env_example.txt` for required and optional environment variables.

---

## Notes

- Only video files are accepted for upload.
- Processing is asynchronous; use the status endpoint to track progress.
- If AWS S3 is configured, processed videos are uploaded to your S3 bucket.
- For troubleshooting, check server logs.

---

## Contact

For support or questions, open an issue on the repository.
