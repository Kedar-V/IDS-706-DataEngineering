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

# Copy the app code
COPY . /app

# Add the daily cron job (midnight)
RUN echo "0 0 * * * python3 /app/bitcoin_price_update.py >> /var/log/bitcoin_price_update.log 2>&1" > /etc/cron.d/bitcoin_cron \
    && chmod 0644 /etc/cron.d/bitcoin_cron \
    && crontab /etc/cron.d/bitcoin_cron

# Create log file
RUN touch /var/log/bitcoin_price_update.log

# Start cron in foreground
CMD ["cron", "-f"]
