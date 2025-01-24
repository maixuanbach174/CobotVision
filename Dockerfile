FROM python:3.12.7-slim

# Set the working directory in the container
WORKDIR /app

# Copy your script into the container
COPY src/server/. /app
COPY requirements.txt /app
COPY models/. /app/models

# Update and install ffmpeg
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install required libraries
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the WebSocket server listens on
EXPOSE 8080

# Run the script
CMD ["python", "server.py"]