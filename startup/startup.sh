#!/bin/bash

setup_cron_jobs() {
    # Daily cron job for bitcoin_price_update.py
    echo "0 0 * * * python3 /app/bitcoin_price_update.py >> /var/log/bitcoin_price_update.log 2>&1" > /etc/cron.d/bitcoin_cron
    chmod 0644 /etc/cron.d/bitcoin_cron
    crontab /etc/cron.d/bitcoin_cron

    # Hourly cron job for news_update.py
    echo "0 * * * * python3 /app/Week4/app/worker/news_update.py >> /var/log/news_update.log 2>&1" > /etc/cron.d/news_cron
    chmod 0644 /etc/cron.d/news_cron
    crontab /etc/cron.d/news_cron

    # Monthly cron job for model_update_cron.py
    echo "0 0 1 * * python3 /app/Week4/app/worker/model_update_cron.py >> /var/log/model_update_cron.log 2>&1" > /etc/cron.d/model_update_cron
    chmod 0644 /etc/cron.d/model_update_cron
    crontab /etc/cron.d/model_update_cron
}

create_log_files() {
    touch /var/log/bitcoin_price_update.log
    touch /var/log/news_update.log
    touch /var/log/model_update.log
}

run_init() {
    echo "Running bitcoin_price_update.py..."
    python3 /app/bitcoin_price_update.py >> /var/log/bitcoin_price_update.log 2>&1

    echo "Running news_update.py..."
    python3 /app/Week4/app/worker/news_update.py >> /var/log/news_update.log 2>&1
}

start_app() {
    cd /app/Week4/app/server || exit 1
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
}

start_cron() {
    cron -f
}

###
# NOTE
# Currently cron jobs have been stopped since the AWS instance is only having one gigs of Ram and one cpu instance resulting in chocking when cron jobs are run in parallel to the application
###

# start_cron
# setup_cron_jobs
# create_log_files

run_init
start_app

