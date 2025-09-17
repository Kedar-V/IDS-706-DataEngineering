# -------------------------------
# Makefile for Bitcoin Data Analysis Project
# -------------------------------

# -------------------------------
# Target: install
# -------------------------------
# Upgrade pip and install all dependencies listed in requirements.txt
install:
	 python3 -m pip install --upgrade pip &&\
	 python3 -m pip install -r requirements.txt

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
	 python3 -m pytest -vv --cov=Bitcoin_DataAnalysis test_BitcoinDataAnalysis.py

# -------------------------------
# Target: run
# -------------------------------
# Run the main Bitcoin data analysis script
run:
	 python3 Bitcoin_DataAnalysis.py

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
