# data_processing.py
import os

def load_tickers_from_file(file_path: str) -> list:
    """Load tickers from a file and return them as a list."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return []

def save_ticker_to_file(file_path: str, ticker: str) -> None:
    """Save a ticker to a file."""
    with open(file_path, 'a') as f:
        f.write(f"{ticker}\n")
