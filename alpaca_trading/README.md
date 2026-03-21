# Alpaca Paper Trading (separate folder)

This folder contains a minimal **train → save → paper-trade once** workflow.

## Install

- Install the repo dependencies (`scikit-learn`, `yfinance`, `joblib`, etc.) from the repo root `requirements.txt`.
- Ensure `alpaca-py` is installed (it’s optional in the repo root `requirements.txt`).

## 1) Set credentials as environment variables

Recommended (shell):

```bash
export ALPACA_API_KEY_ID="your_key_id_here"
export ALPACA_API_SECRET_KEY="your_secret_key_here"
```

If you really want to set them in Python (as requested):

```python
import os
os.environ["ALPACA_API_KEY_ID"] = "your_key_id_here"
os.environ["ALPACA_API_SECRET_KEY"] = "your_secret_key_here"
```

## 2) Train and save the model

```python
from alpaca_trading.ml_model import train_and_persist_model

model_path = train_and_persist_model("AAPL", out_path="ridge_AAPL.joblib")
print(model_path)
```

## 3) Run a paper trade (one-shot)

```python
from alpaca_trading.ml_model import alpaca_paper_trade_once

result = alpaca_paper_trade_once("AAPL", model_path=model_path, qty=1, threshold=0.0)
print(result)
```

## CLI usage (optional)

```bash
python alpaca_trading/train_model.py --ticker AAPL --out ridge_AAPL.joblib
python alpaca_trading/paper_trade_once.py --ticker AAPL --model ridge_AAPL.joblib --qty 1 --threshold 0.0
```
