[![Build & Deploy to EC2 via Docker Hub](https://github.com/Kedar-V/IDS-706-DataEngineering/actions/workflows/deploy.yml/badge.svg)](https://github.com/Kedar-V/IDS-706-DataEngineering/actions/workflows/deploy.yml)

# Personal AI Trading Assistant

[![Demo Website](images/demo1.png)](http://13.231.121.59:8501/)

## Index
1. [Demo Website](#demo-website)
2. [Features](#features)
3. [Detailed Documentation](#detailed-documentation)
4. [How to Run](#how-to-run)
5. [Multi-Container Docker Application](#multi-container-docker-application)
6. [Folder Structure](#folder-structure)
7. [License](#license)

## Demo Website
This website is deployed and live via AWS
[Click here to view the live demo website](http://13.231.121.59:8501/)

Please note that the site may occasionally lag / crash as it has been hosted on free tier of AWS with very low compute

Click [here](https://drive.google.com/file/d/1QRNeNIyvURrpFANJiaPxJmN0PLpWT1QG/view?usp=sharing) to watch the pre-recorded demo of the application.

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

1. **[EDA.md](readme/EDA.md)**: Provides insights into the exploratory data analysis (EDA) of the Bitcoin dataset. It includes benchmarking results, feature engineering techniques, and visualizations of price and volume trends.
2. **[TEST.md](readme/TEST.md)**: Describes the testing framework used in the project. It covers pytest fixtures, test categories, and instructions for running unit and system tests.
3. **[DATASET.md](readme/DATASET.md)**: Details the Bitcoin dataset used in the project, including its structure, size, and instructions for loading it into various DataFrame libraries like Pandas, Polars, and PySpark.
4. **[ENV_SETUP.md](readme/ENV_SETUP.md)**: Explains how to set up the development environment using Dev Containers or Docker. It ensures a reproducible and isolated environment for development and testing.
5. **[CODE_FEATURES.md](readme/CODE_FEATURES.md)**: Highlights the key features of the codebase, such as data loading, EDA, feature engineering, dataset preparation, and model evaluation. It also includes example usage for each module.
6. **[GIT_ACTIONS.md](readme/GIT_ACTIONS.md)**: Documents the GitHub Actions workflows for CI/CD. It explains the steps for Docker image building, deployment to AWS EC2, and code sanity checks.
7. **[CRON_JOBS.md](readme/CRON_JOBS.md)**: Highlights the automation of data ingestion, news updates, and model training using cron jobs. Details the scripts, their schedules, and how to set them up.

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

## Multi-Container Docker Application

This project is designed as a multi-container Docker application, combining a Python-based web application, backend services, and a SQL database. The architecture ensures modularity, scalability, and ease of deployment.

### Components
1. **Python Web Application**:
   - Built using Streamlit for the frontend.
   - Provides an interactive interface for visualizing Bitcoin OHLC data, querying data using natural language, and predicting Bitcoin prices.

2. **Backend Services**:
   - Handles data ingestion, processing, and predictive modeling.
   - Includes modules for scraping Bitcoin data, engineering features, and running machine learning models.

3. **SQL Database**:
   - Stores historical Bitcoin data and other relevant information.
   - Enables efficient querying and data retrieval for the application.

### Deployment on AWS
- The application is deployed on AWS using an EC2 instance.
- Docker Compose is used to orchestrate the multi-container setup, ensuring seamless communication between the web app, backend, and database.
- The deployment leverages AWS Free Tier resources, making it cost-effective for development and testing.

### Automation with GitHub Actions
- The entire build and deployment process is automated using GitHub Actions.
- **CI/CD Pipeline**:
  - Builds Docker images for the application.
  - Pushes the images to Docker Hub.
  - Deploys the application to the AWS EC2 instance.
- **Code Quality Checks**:
  - Runs linting and testing workflows to ensure code reliability and maintainability.

This architecture ensures that the application is modular, scalable, and easy to maintain, while leveraging the power of Docker and AWS for deployment.

## Folder Structure
```
IDS-706-DataEngineering
├── Week1
├── Week2
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

