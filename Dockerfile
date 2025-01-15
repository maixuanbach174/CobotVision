# Step 1: Use Python base image for ARM (Raspberry Pi)
FROM python:3.10-slim-bullseye

# Step 2: Set a working directory
WORKDIR /app

# Step 3: Copy project files to the container
COPY . .

# Step 4: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Define the default command to run the app
CMD ["python", "main.py"]

