# Personal AI Trading Assistant

![Demo Website](images/demo1.png)

## Demo Website
This website is deployed and live via AWS
[Click here to view the demo](http://54.250.160.141:8501/)

## Features
- **OHLC Data Viewer**: Visualize Bitcoin OHLC data with filtering options.
- **Daily News**: Stay updated with the latest Bitcoin-related news.
- **AI Assistant**: Query Bitcoin data using natural language.
- **Price Prediction**: Predict the next day's Bitcoin closing price using a Random Forest model.

### CommitDiff (For HW)
![commit diff](images/commit%20diff.png)

### Build Success (For HW)
![commit diff](images/CI:CD%20success.png)

## Detailed Documentation

This repository includes additional documentation for specific aspects of the project:

1. **[EDA.md](readme/EDA.md)**: Detailed exploratory data analysis (EDA) of the Bitcoin dataset, including benchmarking, feature engineering, and insights from price and volume trends.
2. **[TEST.md](readme/TEST.md)**: Overview of the testing framework, including pytest fixtures, test categories, and instructions for running tests.
3. **[DATASET.md](readme/DATASET.md)**: Information about the Bitcoin dataset, supported DataFrame libraries, and setup instructions for data analysis.
4. **[ENV_SETUP.md](readme/ENV_SETUP.md)**: Instructions for setting up the development environment using Dev Containers or Docker for reproducible development and testing.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Kedar-V/IDS-706-DataEngineering.git
   ```
2. Navigate to the project directory:
   ```bash
   cd IDS-706-DataEngineering
   ```
3. Build and run the Docker containers:
   ```bash
   docker compose up -d
   ```
4. Access the app at [http://localhost:8501](http://localhost:8501).

## Folder Structure
```
IDS-706-DataEngineering
├── Week1
├── Week2
│   ├── Bitcoin_DataAnalysis.py
│   ├── assets
│   │   ├── benchmark.png
│   │   ├── candles.png
│   │   ├── output.png
│   │   ├── priceTrend.png
│   │   └── volumeTrend.png
├── Week3
├── Week4
│   ├── app
│   │   ├── server
│   │   ├── worker
│   │   ├── dao
│   │   ├── data
│   │   └── model
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## License
This project is licensed under the MIT License.

