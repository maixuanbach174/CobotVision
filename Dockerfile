FROM python:3.13.1-slim

# Set the working directory in the container
WORKDIR /app

# Copy your script into the container
COPY src/server/. /app

# Install required libraries
RUN pip install websockets

# Expose the port the WebSocket server listens on
EXPOSE 8080

# Run the script
CMD ["python", "server.py"]