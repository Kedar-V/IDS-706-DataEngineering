# Environment Setup

## Dev Container (VS Code)

This project provides a **Dev Container** configuration for reproducible development and testing using VS Code. You can run all tests and Python scripts in a containerized environment without polluting your local machine.

### Steps to Use Dev Container

1. **Install VS Code Extensions**  
   Make sure you have the following installed in VS Code:
   - **Remote - Containers**  
     (Official extension to work with Dev Containers)

2. **Open the Project in VS Code**  
   - Open the folder containing this repository in VS Code.

3. **Open in Dev Container**  
   - Press `F1` → type `Remote-Containers: Open Folder in Container` → select the project folder.  
   - VS Code will build the container using the included `devcontainer.json` and `Dockerfile`.

4. **Post-create Commands**  
   - The Dev Container automatically installs dependencies from `Week3/requirements.txt`.

5. **Run Tests Inside Container**  
   Open a terminal inside VS Code (within the container) and run:

   ```bash
   cd Week3
   pytest -vv --cov=Bitcoin_DataAnalysis test_BitcoinDataAnalysis.py
    ```
---

## Docker Container

This project can also be run using a **Docker container** without relying on VS Code Dev Containers. This ensures a reproducible Python environment on any system with Docker installed.

### Steps to Use Docker Container

1. **Install Docker**  
   Ensure Docker is installed and running:

   ```bash
   docker --version
   ```
2. Build the Docker Image from the project root:
    ```bash
    docker build -t python_week3 .
    ```
    This uses the Dockerfile to create a container with Python 3.12 and dependencies from Week3/requirements.txt.

3. Run the Container
    ```bash
    docker run -it --name python_week3 -v $(pwd):/app python_week3
    ```

- `-v $(pwd):/app` mounts the project folder into the container.  
- It opens an interactive terminal.

4. To enter the running container, use:  
    ```bash
    docker exec -it python_week3 /bin/bash
    ```

5. Alternative: Using Docker Compose

    You can also use docker-compose for easier management:
    ```bash
    docker-compose up -d
    ```
    - This builds and starts the container in detached mode.

    Access the container with:

    ```bash
    docker compose exec python /bin/bash
    ```