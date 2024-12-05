from stock_info import get_stock_info
from embeddings import cosine_similarity_between_sentences
from company_tickers import get_company_tickers
import json
import os

def load_company_tickers_from_file(file_path="company_tickers.json"):
    """
    Loads company tickers from a local JSON file.

    Args:
        file_path (str): Path to the company tickers JSON file.

    Returns:
        list: A list of tickers extracted from the JSON file.
    """
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist. Fetching data from the server...")
        get_company_tickers()  # Download the file if it doesn't exist
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Extract tickers from the JSON structure
    tickers = [data[key]["ticker"] for key in data]
    return tickers


def fetch_data_for_tickers(tickers, max_companies=5):
    """
    Fetches and prints stock information for a list of tickers.

    Args:
        tickers (list): List of stock tickers to fetch information for.
        max_companies (int): Maximum number of companies to fetch data for.
                             Default is 5 for demonstration purposes.
    """
    for i, ticker in enumerate(tickers[:max_companies], start=1):
        print(f"\nFetching data for company {i}/{max_companies}: {ticker}")
        try:
            company_info = get_stock_info(ticker)
            print(json.dumps(company_info, indent=4))
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")


# Main execution
if __name__ == "__main__":
    # Load tickers from the JSON file
    tickers = load_company_tickers_from_file()

    # Fetch and print stock data for the first 5 companies
    fetch_data_for_tickers(tickers)

    # Example: Compare business description to a custom query
    company_description = "I want to find companies that make food and are headquartered in California"
    if tickers:
        aapl_info = get_stock_info(tickers[1])  # Use the second company (AAPL in the example JSON)
        aapl_description = aapl_info.get('Business Summary', "")
        similarity = cosine_similarity_between_sentences(aapl_description, company_description)
        print(f"\nSimilarity between descriptions: {similarity:.4f}")
