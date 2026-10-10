FROM python:3.10-slim

# Install required system libraries for OpenCV (ffmpeg, libsm6, libxext6, libgl1-mesa-glx)
RUN apt-get update && \
    (apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 libgl1-mesa-glx || \
     apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 libgl1) && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files into /app working directory
COPY . /app

# Set environment variable PORT=8080 and expose port 8080
ENV PORT=8080
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]
