import argparse

from ml_model import train_and_persist_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--out", default="model.joblib")
    args = parser.parse_args()

    out_path = train_and_persist_model(args.ticker, out_path=args.out)
    print(out_path)


if __name__ == "__main__":
    main()

