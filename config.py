# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX")
NAMESPACE = "stock-descriptions"
SUCCESSFUL_TICKERS_FILE = "successful_tickers.txt"
UNSUCCESSFUL_TICKERS_FILE = "unsuccessful_tickers.txt"
