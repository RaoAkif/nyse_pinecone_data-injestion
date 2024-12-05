from stock_info import get_stock_info
from company_tickers import get_company_tickers
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Replace with your API key or use a .env file
PINECONE_ENV = os.getenv("PINECONE_ENV")  # Replace with your Pinecone environment
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

# Process each ticker
for idx, stock in company_tickers.items():
    ticker = stock["ticker"]

    # Skip already processed tickers
    if ticker in successful_tickers or ticker in unsuccessful_tickers:
        print(f"Skipping already processed ticker: {ticker}")
        continue

    try:
        # Fetch stock data
        stock_data = get_stock_info(ticker)
        stock_description = stock_data.get("Business Summary", "")

        # Skip if no business description
        if not stock_description:
            print(f"No description available for ticker: {ticker}")
            unsuccessful_tickers.append(ticker)
            continue

        # Save embeddings to Pinecone
        document = Document(page_content=stock_description, metadata=stock_data)
        vectorstore_from_documents = PineconeVectorStore.from_documents(
            documents=[document],
            embedding=hf_embeddings,
            index_name=INDEX_NAME,
            namespace=NAMESPACE
        )

        print(f"Successfully processed and saved embeddings for ticker: {ticker}")
        successful_tickers.append(ticker)

    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
        unsuccessful_tickers.append(ticker)

    # Save progress to files
    with open(SUCCESSFUL_TICKERS_FILE, "w") as f:
        f.write("\n".join(successful_tickers))

    with open(UNSUCCESSFUL_TICKERS_FILE, "w") as f:
        f.write("\n".join(unsuccessful_tickers))

print("\nProcessing complete.")
print(f"Successfully processed tickers: {len(successful_tickers)}")
print(f"Failed tickers: {len(unsuccessful_tickers)}")
