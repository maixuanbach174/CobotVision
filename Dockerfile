# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install pip (if not already available)
RUN python -m ensurepip --upgrade

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip install torch torchvision

# Check Python version
RUN python --version

# Check pip version
RUN pip --version


