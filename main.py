from stock_info import get_stock_info
from company_tickers import get_company_tickers
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import concurrent.futures
import os
import json

# Load environment variables
load_dotenv()

# Constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX")
NAMESPACE = "stock-descriptions"
SUCCESSFUL_TICKERS_FILE = "successful_tickers.txt"
UNSUCCESSFUL_TICKERS_FILE = "unsuccessful_tickers.txt"

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

# Initialize embeddings
hf_embeddings = HuggingFaceEmbeddings()

# Load company tickers
company_tickers = get_company_tickers()

# Tracking lists for tickers
successful_tickers = []
unsuccessful_tickers = []

# Load existing progress
if os.path.exists(SUCCESSFUL_TICKERS_FILE):
    with open(SUCCESSFUL_TICKERS_FILE, "r") as f:
        successful_tickers = [line.strip() for line in f if line.strip()]

if os.path.exists(UNSUCCESSFUL_TICKERS_FILE):
    with open(UNSUCCESSFUL_TICKERS_FILE, "r") as f:
        unsuccessful_tickers = [line.strip() for line in f if line.strip()]

# Initialize Pinecone vector store
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=hf_embeddings,
    index=index
)

def process_stock(stock_ticker: str) -> str:
    # Skip if already processed
    if stock_ticker in successful_tickers:
        return f"Already processed {stock_ticker}"

    try:
        # Get and store stock data
        stock_data = get_stock_info(stock_ticker)
        stock_description = stock_data['Business Summary']

        # Store stock description in Pinecone
        vectorstore_from_texts = PineconeVectorStore.from_documents(
            documents=[Document(page_content=stock_description, metadata=stock_data)],
            embedding=hf_embeddings,
            index_name=INDEX_NAME,
            namespace=NAMESPACE
        )

        # Track success
        with open(SUCCESSFUL_TICKERS_FILE, 'a') as f:
            f.write(f"{stock_ticker}\n")
        successful_tickers.append(stock_ticker)

        return f"Processed {stock_ticker} successfully"

    except Exception as e:
        # Track failure
        with open(UNSUCCESSFUL_TICKERS_FILE, 'a') as f:
            f.write(f"{stock_ticker}\n")
        unsuccessful_tickers.append(stock_ticker)

        return f"ERROR processing {stock_ticker}: {e}"

def parallel_process_stocks(tickers: list, max_workers: int = 10) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_stock, ticker): ticker
            for ticker in tickers
        }

        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                print(result)

                # Stop on error
                if result.startswith("ERROR"):
                    print(f"Stopping program due to error in {ticker}")
                    executor.shutdown(wait=False)
                    raise SystemExit(1)

            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')
                print("Stopping program due to exception")
                executor.shutdown(wait=False)
                raise SystemExit(1)

# Prepare your tickers
tickers_to_process = [company_tickers[num]['ticker'] for num in company_tickers.keys()]

# Process them
parallel_process_stocks(tickers_to_process, max_workers=10)