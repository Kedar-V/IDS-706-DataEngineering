FROM python:3.12-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies including cron
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY Week4/requirements.txt ./Week4/requirements.txt

# Install Python packages
RUN pip install --upgrade pip \
    && pip install -r Week4/requirements.txt

# Copy the app code including startup.sh
COPY . /app

# Make startup.sh executable
RUN chmod +x /app/startup/startup.sh

# Run startup.sh when container starts
CMD ["/app/startup/startup.sh"]
