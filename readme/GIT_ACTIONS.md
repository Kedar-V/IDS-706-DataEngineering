# GitHub Actions for Deployment

This repository uses GitHub Actions to automate the build and deployment process. The workflow files are located in `.github/workflows/` and handle the following tasks:

---

## Index
1. [Docker Build and AWS Deployment Workflow](#docker-build-and-aws-deployment-workflow)
2. [Code Sanity Workflow](#code-sanity-workflow)
3. [Role of `startup.sh`](#role-of-startupsh)

---

## Docker Build and AWS Deployment Workflow

### Overview
The `deploy.yml` workflow automates the process of building and deploying the application to an EC2 instance. It includes the following steps:

### 1. Build and Push Docker Image
- **Action**: `docker/build-push-action@v2`
- **Description**: Builds the Docker image for the application and pushes it to Docker Hub.
- **Secrets Required**:
  - `DOCKERHUB_USERNAME`: Docker Hub username.
  - `DOCKERHUB_TOKEN`: Docker Hub access token.

### 2. Deploy to EC2
- **Action**: `appleboy/ssh-action@v0.1.7`
- **Description**: Connects to an EC2 instance via SSH and deploys the application.
- **Steps**:
  - Ensures the `/home/ec2-user/app` directory exists.
  - Installs Git if not already installed.
  - Clones the repository or pulls the latest changes if the repository already exists.

### 3. Start Docker Containers on EC2
- **Action**: `appleboy/ssh-action@v0.1.7`
- **Description**: Starts the Docker containers on the EC2 instance using `docker-compose`.
- **Steps**:
  - Pulls the latest Docker images.
  - Stops any running containers.
  - Starts the containers in detached mode.

### 4. Run Startup Script Inside Container
- **Action**: `appleboy/ssh-action@v0.1.7`
- **Description**: Executes the `startup.sh` script inside the running container to set up cron jobs and start the application.

---

## Code Sanity Workflow

### Overview
The `main.yml` workflow is designed for continuous integration (CI) tasks, including linting, testing, and code coverage. It ensures code quality and reliability before deployment.

### Steps

#### 1. Checkout Repository
- **Action**: `actions/checkout@v4`
- **Description**: Checks out the repository to the GitHub Actions runner.

#### 2. Set Up Python
- **Action**: `actions/setup-python@v5`
- **Description**: Sets up Python 3.11 for the workflow.

#### 3. Install Dependencies
- **Command**:
  ```bash
  python -m pip install --upgrade pip
  pip install -r ./Week4/requirements.txt
  ```
- **Description**: Installs the required Python dependencies.

#### 4. Lint Code
- **Command**:
  ```bash
  flake8 --ignore=C,N,E,W503 .
  ```
- **Description**: Lints the codebase using `flake8` with specific rules ignored.

#### 5. Run Tests
- **Command**:
  ```bash
  pytest -vv --cov=Bitcoin_DataAnalysis ./Week3/test_BitcoinDataAnalysis.py
  ```
- **Description**: Runs the test suite with verbose output and code coverage.

---

## Role of `startup.sh`

The `startup.sh` script plays a critical role in ensuring clean and automated deployment of the application. It is executed as part of the deployment process and handles the following tasks:

1. **Setting Up Cron Jobs**:
   - Configures cron jobs for automating data ingestion, news updates, and model training.
   - Ensures that these jobs are scheduled correctly and logs are maintained for debugging.

2. **Creating Log Files**:
   - Initializes log files for each cron job to capture their output and errors.
   - This ensures that any issues during execution can be easily traced.

3. **Running Initialization Scripts**:
   - Executes the `bitcoin_price_update.py` and `news_update.py` scripts to populate the database with the latest data before the application starts.

4. **Starting the Application**:
   - Launches the Streamlit application, making it accessible to users.

By centralizing these tasks, the `startup.sh` script ensures that the deployment process is seamless and reduces the chances of manual errors. For more details, refer to the `startup.sh` script in the `startup` directory.

---

## How to Use

1. **Set Up Secrets** (for `deploy.yml`)
   - Add the following secrets to your GitHub repository:
     - `DOCKERHUB_USERNAME`
     - `DOCKERHUB_TOKEN`
     - `EC2_IP`
     - `EC2_SSH_KEY`

2. **Trigger the Workflows**
   - Both workflows are triggered on `push` or `pull_request` events.

3. **Monitor the Workflows**
   - Check the Actions tab in your GitHub repository to monitor the progress of the workflows.

## File Locations
- `deploy.yml`: `.github/workflows/deploy.yml`
- `main.yml`: `.github/workflows/main.yml`