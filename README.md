# Trine.capstone.sentiment_analysis

### Environment

This repository contains a minimal FastAPI app and a Nix `shell.nix` dev environment.

Run using Makefile shortcut

```bash
# Shortcut to start the dev server
make dev
```

Run using Nix dev shell

```bash
# Start a temporary dev shell and run uvicorn immediately
nix-shell --run 'uvicorn app.main:app --reload --host 127.0.0.1 --port 8000'
```

Open these URLs:
- API root: http://127.0.0.1:8000/
- Interactive docs (Swagger): http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### Presentation

[PPT](./report/Presentation.pptx)

[Essay](./report/Use%20Sentiment%20Analysis%20on%20Public%20Momentum%20to%20Predict%20Stock%20Trends.pdf)

### Contributor

Qiyun Lu

### FinBERT Notice

This project uses the **ProsusAI/finBERT** model, which is licensed under the **Apache License 2.0**.  
Copyright © 2020 Prosus AI.

A copy of the original finBERT license is included in `finbert/LICENSE`.  
All use of the finBERT model, weights, and associated materials must comply with the terms of the Apache 2.0 license.
