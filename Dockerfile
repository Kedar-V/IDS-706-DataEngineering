# Base image: Python 3.12 slim
FROM python:3.12-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only Week3 requirements
COPY Week3/requirements.txt ./Week3/requirements.txt

# Install Week3 dependencies
RUN pip install --upgrade pip \
    && pip install -r Week3/requirements.txt

# Copy full workspace
COPY . /app

# Default command
CMD ["bash"]
