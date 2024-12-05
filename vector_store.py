# vector_store.py
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import INDEX_NAME, NAMESPACE

# Initialize embeddings and vector store
hf_embeddings = HuggingFaceEmbeddings()

def store_stock_description(stock_description: str, stock_data: dict) -> None:
    """Store the stock description in Pinecone vector store."""
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=hf_embeddings,
        index=None  # Should be initialized in the main script
    )

    vectorstore_from_texts = PineconeVectorStore.from_documents(
        documents=[Document(page_content=stock_description, metadata=stock_data)],
        embedding=hf_embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE
    )
