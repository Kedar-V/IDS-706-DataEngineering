# Cron Jobs

This project includes several cron jobs to automate data ingestion, model training, and news updates. These jobs ensure that the application remains up-to-date with the latest data and predictions.

## Cron Job Details

### 1. Bitcoin Price Update
- **Script**: `bitcoin_price_update.py`
- **Description**: Fetches historical Bitcoin OHLC data and stores it in the SQL database.
- **Frequency**: Runs daily at midnight.
- **Command**:
  ```bash
  python3 /app/bitcoin_price_update.py >> /var/log/bitcoin_price_update.log 2>&1
  ```

### 2. News Update
- **Script**: `news_update.py`
- **Description**: Scrapes the latest Bitcoin-related news articles and stores them in the SQL database.
- **Frequency**: Runs hourly.
- **Command**:
  ```bash
  python3 /app/Week4/app/worker/news_update.py >> /var/log/news_update.log 2>&1
  ```

### 3. Model Update
- **Script**: `model_update.py`
- **Description**: Trains a new Random Forest model using the latest Bitcoin data and saves the model for predictions.
- **Frequency**: Runs on the 1st of every month at midnight.
- **Command**:
  ```bash
  python3 /app/Week4/app/worker/model_update_cron.py >> /var/log/model_update_cron.log 2>&1
  ```

## Setting Up Cron Jobs

The cron jobs are configured in the `startup.sh` script. This script sets up the cron jobs, creates log files, and initializes the application. Below are the relevant sections from the `startup.sh` script:

### Cron Job Setup in `startup.sh`
```bash
# Daily cron job for bitcoin_price_update.py
0 0 * * * python3 /app/bitcoin_price_update.py >> /var/log/bitcoin_price_update.log 2>&1

# Hourly cron job for news_update.py
0 * * * * python3 /app/Week4/app/worker/news_update.py >> /var/log/news_update.log 2>&1

# Monthly cron job for model_update_cron.py
0 0 1 * * python3 /app/Week4/app/worker/model_update_cron.py >> /var/log/model_update_cron.log 2>&1
```

### Log File Creation
The `startup.sh` script also ensures that log files are created for each cron job:
```bash
touch /var/log/bitcoin_price_update.log
touch /var/log/news_update.log
touch /var/log/model_update.log
```

### Running Initialization Scripts
Before starting the application, the `startup.sh` script runs the `bitcoin_price_update.py` and `news_update.py` scripts to ensure the database is populated with the latest data:
```bash
python3 /app/bitcoin_price_update.py >> /var/log/bitcoin_price_update.log 2>&1
python3 /app/Week4/app/worker/news_update.py >> /var/log/news_update.log 2>&1
```

For more details, refer to the `startup.sh` script in the `startup` directory.