import argparse

from ml_model import alpaca_paper_trade_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    result = alpaca_paper_trade_once(
        args.ticker,
        model_path=args.model,
        qty=args.qty,
        threshold=args.threshold,
    )
    print(result)


if __name__ == "__main__":
    main()

