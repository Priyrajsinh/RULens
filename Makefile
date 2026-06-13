# RULens developer tasks.
# PY defaults to the active interpreter; on Windows use:
#   make gates PY=venv/Scripts/python.exe
# On Linux/macOS after activating the venv, plain `make gates` works.
PY ?= python

.PHONY: setup lint test gates run-api run-ui

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	# Lightweight runtime needed for the gates. The heavy GPU stack lives in
	# requirements.txt and is installed on experiment machines:
	#   $(PY) -m pip install -r requirements.txt
	$(PY) -m pip install numpy==2.4.6 pyyaml==6.0.3 pydantic==2.13.4

lint:
	$(PY) -m black --check src tests
	$(PY) -m isort --check-only --profile black src tests
	$(PY) -m flake8 src tests

test:
	$(PY) -m pytest tests/ -v --tb=short --cov=src --cov-fail-under=70

gates:
	$(PY) -m black --check src tests
	$(PY) -m isort --check-only --profile black src tests
	$(PY) -m flake8 src tests
	$(PY) -m mypy src
	$(PY) -m bandit -r src -ll -ii
	$(PY) -m interrogate src
	git ls-files | xargs $(PY) -m detect_secrets.pre_commit_hook --baseline .secrets.baseline
	$(PY) -m radon cc src -a -nc
	$(PY) -m pytest tests/ -v --tb=short --cov=src --cov-fail-under=70
	$(PY) -m pip_audit

run-api:
	$(PY) -m uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000

run-ui:
	$(PY) -m streamlit run src/ui/dashboard.py
