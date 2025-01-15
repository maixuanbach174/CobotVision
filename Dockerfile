# Step 1: Use Python base image for ARM (Raspberry Pi)
FROM pytorch/pytorch:latest

# Step 2: Set a working directory
WORKDIR /app

# Step 3: Copy project files to the container
COPY requirements.txt .

# Step 4: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Step 5: Define the default command to run the app
CMD ["python", "src/mycobot_classification./py"]

