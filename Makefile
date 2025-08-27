# -------------------------------
# Makefile for Python Project
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
# Run flake8 on hello.py to check for PEP8 compliance
lint:
	flake8 --ignore=C,N hello.py

# -------------------------------
# Target: test
# -------------------------------
# Run all tests with pytest, verbose output, and generate coverage report for hello.py
test:
	python -m pytest -vv --cov=hello test_hello.py

# -------------------------------
# Target: clean
# -------------------------------
# Remove Python cache files and pytest coverage data
clean:
	rm -rf __pycache__ .pytest_cache .coverage

# -------------------------------
# Target: all
# -------------------------------
# Execute install, format, lint, and test targets in order
all: install format lint test
