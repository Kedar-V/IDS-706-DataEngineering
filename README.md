# IDS-706-DataEngineering

[![Python Template for IDS706](https://github.com/Kedar-V/IDS-706-DataEngineering/actions/workflows/main.yml/badge.svg)](https://github.com/Kedar-V/IDS-706-DataEngineering/actions/workflows/main.yml)

# Hello Python

This repository contains Python functions demonstrating basic greetings, addition, and a self-introduction. 
---

## Files

- `hello.py` – Contains the following functions:
  - `say_hello(name: str) -> str`  
    Returns a greeting message for students.
  - `add(a: int, b: int) -> int`  
    Returns the sum of two numbers.
  - `hello_world(name: str, role: str, skills: List[str], education: str, timestamp: bool = False) -> str`  
    Returns a formatted self-introduction. Optionally includes a timestamp.

---

## Usage

Run the script directly:

```bash
python hello.py
```

## Running Tests and Linting

You can use the provided **Makefile** to quickly install dependencies, format code, check style, run tests, and clean up temporary files.

### Makefile Commands

- `make install` – Installs/updates Python packages listed in `requirements.txt`.  
- `make format` – Formats all Python files using `black`.  
- `make lint` – Runs `flake8` on `hello.py` to check for PEP 8 compliance.  
- `make test` – Runs all unit tests using `pytest` with coverage report.  
- `make clean` – Removes temporary files such as `__pycache__`, `.pytest_cache`, and coverage reports.  
- `make all` – Runs all of the above in sequence: install, format, lint, and test.

---

### Test Cases Explanation

The project includes unit tests in `test_hello.py`:

1. **`test_say_hello()`**
   - Verifies that `say_hello(name)` returns the correct greeting message.  
   - Tests multiple names (e.g., `"Kedar"`, `"Sam"`).  
   - Tests an edge case where the name is an empty string.

2. **`test_add()`**
   - Checks that `add(a, b)` correctly computes the sum of two numbers.  
   - Includes edge cases such as very large numbers, negative numbers, and zero.

3. **`test_hello_world_intro()`**
   - Verifies that `hello_world(...)` returns a properly formatted introduction.  
   - Checks that the output contains the correct name, role, skills, and education.  
   - Timestamp is set to `False` during this test to ensure deterministic results.

---

### How to Run

Use the Makefile to run all commands:

```bash
# Install dependencies
make install

# Format code
make format

# Check code style
make lint

# Run tests
make test

# Clean temporary files
make clean

# Run everything
make all
```

## Development Environment (Dev Container)

This project includes a **Dev Container** configuration using VS Code’s Remote - Containers extension. A dev container is a Docker-based development environment that ensures the project runs with consistent dependencies and tools, regardless of the host machine.

### Key Points

- The container uses the image:
  ```bash
  mcr.microsoft.com/devcontainers/python:1-3.12-bullseye
  ```

  which comes with Python 3.12 and a Debian-based environment. Optional features can be added via the "features" property in devcontainer.json. Ports from the container can be forwarded to the host using "forwardPorts" if needed.

- Commands to set up the environment after the container is created can be added using "postCreateCommand", e.g.:
  ```bash 
  "postCreateCommand": "pip3 install --user -r requirements.txt"
  ```

  You can also configure VS Code tools or set the user with `"customizations"` or `"remoteUser"` properties.

### Benefits

- **Consistency:** All developers use the same Python version and dependencies.  
- **Isolation:** The container is isolated from your local machine, preventing conflicts.  
- **Reproducibility:** Ensures that the project runs the same on any machine that supports Docker.  
- **Integration with VS Code:** You can develop inside the container as if it were local, with debugging, terminal, and extensions fully supported.

### How to Use

1. Install Docker and VS Code Remote - Containers extension.  
2. Open the project folder in VS Code.  
3. Click **Reopen in Container** when prompted, or open the Command Palette (`Ctrl+Shift+P`) → **Remote-Containers: Reopen in Container**.  
4. The dev container will build and launch automatically, and your environment will be ready for development.


## GitHub Actions CI Workflow

This project includes a **GitHub Actions workflow** to automatically build, lint, and test the code whenever changes are pushed or a pull request is created. The workflow ensures code quality and reduces the chance of introducing errors.

### Workflow File: `.github/workflows/python.yml`

```yaml
name: Python Template for IDS706

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Lint with flake8
      run: flake8 hello.py
    - name: Run tests
      run: pytest --cov=hello
```

### What It Does

- **Trigger:** The workflow runs automatically on every `push` or `pull_request`.  
- **Environment:** Uses the latest Ubuntu runner (`ubuntu-latest`) for consistency.  
- **Checkout:** Pulls the project repository using `actions/checkout`.  
- **Python Setup:** Installs Python 3.11 using `actions/setup-python`.  
- **Install Dependencies:** Updates `pip` and installs all packages listed in `requirements.txt`.  
- **Linting:** Runs `flake8` on `hello.py` to ensure PEP 8 compliance.  
- **Testing:** Runs `pytest` with coverage reporting on the `hello` module to verify all tests pass.  

This workflow ensures that code pushed to GitHub is **tested, linted, and ready for deployment or further development**.



