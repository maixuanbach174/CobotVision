# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Check Python version
RUN python --version

# Check pip version
RUN pip --version

python3 -m pip install https://github.com/maxisoft/pytorch-arm/releases/download/v1.0.0/numpy-1.23.5-cp39-cp39-linux_armv7l.whl # change the url if needed
python3 -m pip install https://github.com/maxisoft/pytorch-arm/releases/download/v1.0.0/torch-1.13.0a0+git7c98e70-cp39-cp39-linux_armv7l.whl # change the url if needed

python3 -c 'import torch; print(torch.nn.Conv2d(8, 1, (3, 3))(torch.randn(4, 8, 3, 3)).squeeze_())'


