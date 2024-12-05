# main.py
import concurrent.futures
from config import PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME
from stock_processing import process_stock
from data_processing import load_tickers_from_file
from pinecone import Pinecone, ServerlessSpec
from config import SUCCESSFUL_TICKERS_FILE, UNSUCCESSFUL_TICKERS_FILE
from company_tickers import get_company_tickers

# Initialize Pinecone instance
pinecone = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [index.name for index in pinecone.list_indexes().indexes]:
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=768,  # Ensure this matches your embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
index = pinecone.Index(INDEX_NAME)

def parallel_process_stocks(tickers: list, max_workers: int = 10) -> None:
    """Process stocks in parallel."""
    successful_tickers = load_tickers_from_file(SUCCESSFUL_TICKERS_FILE)
    unsuccessful_tickers = load_tickers_from_file(UNSUCCESSFUL_TICKERS_FILE)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_stock, ticker, successful_tickers, unsuccessful_tickers): ticker
            for ticker in tickers
        }

        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                print(result)

                if result.startswith("ERROR"):
                    print(f"Stopping program due to error in {ticker}")
                    executor.shutdown(wait=False)
                    raise SystemExit(1)

            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')
                print("Stopping program due to exception")
                executor.shutdown(wait=False)
                raise SystemExit(1)

# Example to fetch tickers and start the processing
company_tickers = get_company_tickers()
tickers_to_process = [company_tickers[num]['ticker'] for num in company_tickers.keys()]

# Process the tickers
parallel_process_stocks(tickers_to_process, max_workers=10)
