FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including FFmpeg and other required packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verify moviepy installation
RUN python -c "import moviepy; print('moviepy version:', moviepy.__version__)"

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p output

# Test import at runtime
RUN python -c "from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip; print('moviepy.editor imports successful')"

# Make startup script executable
RUN chmod +x start_api.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application using the startup script
CMD ["python", "start_api.py"] 