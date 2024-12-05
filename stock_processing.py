# stock_processing.py
from data_processing import save_ticker_to_file
from stock_info import get_stock_info
from vector_store import store_stock_description
from config import SUCCESSFUL_TICKERS_FILE, UNSUCCESSFUL_TICKERS_FILE

def process_stock(stock_ticker: str, successful_tickers: list, unsuccessful_tickers: list) -> str:
    """Process the stock ticker and store its description."""
    if stock_ticker in successful_tickers:
        return f"Already processed {stock_ticker}"

    try:
        # Get stock data
        stock_data = get_stock_info(stock_ticker)
        stock_description = stock_data['Business Summary']

        # Store stock description in Pinecone
        store_stock_description(stock_description, stock_data)

        # Track success
        save_ticker_to_file(SUCCESSFUL_TICKERS_FILE, stock_ticker)
        successful_tickers.append(stock_ticker)

        return f"Processed {stock_ticker} successfully"
    
    except Exception as e:
        # Track failure
        save_ticker_to_file(UNSUCCESSFUL_TICKERS_FILE, stock_ticker)
        unsuccessful_tickers.append(stock_ticker)

        return f"ERROR processing {stock_ticker}: {e}"
