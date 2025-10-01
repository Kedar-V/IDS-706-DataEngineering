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

start_streamlit() {
    cd /app/Week4/app/server || exit 1
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
}

start_cron() {
    cron -f
}

setup_cron_jobs
create_log_files
start_streamlit
start_cron