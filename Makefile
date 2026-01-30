# Makefile: quick helpers
.PHONY: dev

dev:
	# Launch using venv Python
	.venv/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
