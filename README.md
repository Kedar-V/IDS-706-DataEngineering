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
