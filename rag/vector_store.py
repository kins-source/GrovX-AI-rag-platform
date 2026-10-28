import os
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils.logger import logger

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
STATE_FILE = os.path.join(DB_PATH, "upload_state.json")

def get_latest_source() -> str:
    """Retrieves the filename of the most recently uploaded document."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                return data.get("latest_source")
        except Exception as e:
            logger.warning(f"Could not read upload state: {e}")
    return None

def set_latest_source(source: str):
    """Sets the given source filename as the latest uploaded document context."""
    os.makedirs(DB_PATH, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"latest_source": source}, f)
        
def get_vector_store():
    """
    Initializes and returns the Chroma Vector Store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    os.makedirs(DB_PATH, exist_ok=True)
    vector_store = Chroma(
        collection_name="enterprise_knowledge",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    return vector_store

def add_documents_to_store(chunks):
    """
    Adds document chunks to the Chroma DB and scopes the global knowledge to the new document.
    """
    if not chunks:
        return

    from rag.cache import query_cache
    query_cache.clear()
        
    # Extract source metadata (added in document_loader) and persist state
    source = chunks[0].metadata.get("source")
    if source:
        set_latest_source(source)
        logger.info(f"Updated global RAG scope to only retrieve from: {source}")

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    logger.info(f"Persisted {len(chunks)} chunks into Chroma Vector Store.")

def retrieve_similar_documents(query: str, k: int = 5):
    """
    Retrieves the top-k most similar document chunks for a query.
    Filters specifically by the latest uploaded document to prevent legacy context leakage.
    """
    vector_store = get_vector_store()
    latest_source = get_latest_source()
    
    # Optional metadata filtering dict
    filter_dict = None
    if latest_source:
        filter_dict = {"source": latest_source}
        logger.info(f"Vector search active filter: {filter_dict}")
        
    results = vector_store.similarity_search(query, k=k, filter=filter_dict)
    return results
