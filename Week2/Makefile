# -------------------------------
# Makefile for Bitcoin Data Analysis Project
# -------------------------------

# -------------------------------
# Target: install
# -------------------------------
# Upgrade pip and install all dependencies listed in requirements.txt
install:
	 pip install --upgrade pip &&\
	 pip install -r requirements.txt

# -------------------------------
# Target: format
# -------------------------------
# Automatically format all Python files in the project using Black
format:
	 black *.py

# -------------------------------
# Target: lint
# -------------------------------
# Run flake8 on all Python files to check for PEP8 compliance
lint:
	 flake8 --ignore=C,N *.py

# -------------------------------
# Target: test
# -------------------------------
# Run pytest for all test files (if any exist)
test:
	 python -m pytest -vv --cov=.

# -------------------------------
# Target: run
# -------------------------------
# Run the main Bitcoin data analysis script
run:
	 python Bitcoin_DataAnalysis.py

# -------------------------------
# Target: clean
# -------------------------------
# Remove Python cache files and pytest coverage data
clean:
	 rm -rf __pycache__ .pytest_cache .coverage

# -------------------------------
# Target: all
# -------------------------------
# Execute install, format, lint, test, and run targets in order
all: install format lint test run
