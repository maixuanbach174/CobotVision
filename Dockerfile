# Use the official PyTorch image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /workspace

# Install any additional dependencies
RUN pip install --upgrade pip && \
    pip install numpy pandas matplotlib

# Set the default command to run Python
CMD ["python"]
