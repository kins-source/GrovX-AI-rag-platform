from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import time
import os

from utils.security import get_api_key
from utils.logger import logger
from utils.telemetry import track_latency_async
from rag.document_loader import load_document
from rag.chunking import chunk_documents
from rag.vector_store import add_documents_to_store
from agents.orchestrator import process_query

app = FastAPI(title="Enterprise AI Knowledge Assistant", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    provider: str | None = None
    api_key: str | None = None
    model: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    latency_seconds: float

@app.post("/upload", dependencies=[Depends(get_api_key)])
@track_latency_async
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint for uploading PDF/TXT documents into the Knowledge Base.
    """
    logger.info(f"Received file upload request: {file.filename}")
    try:
        content = await file.read()
        docs = load_document(content, file.filename)
        chunks = chunk_documents(docs)
        add_documents_to_store(chunks)
        extracted_text = "\n\n".join(doc.page_content.strip() for doc in docs if doc.page_content.strip())
        return {
            "message": f"Successfully processed and stored {file.filename}.",
            "filename": file.filename,
            "pages": len(docs),
            "characters": len(extracted_text),
            "preview": extracted_text[:5000],
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        logger.error(f"Upload error: {e}\n{trace}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(get_api_key)])
@track_latency_async
async def query_assistant(request: QueryRequest):
    """
    Endpoint for querying the agentic assistant.
    """
    start_time = time.time()
    try:
        result = process_query(request.query, request.provider, request.api_key, request.model)
        latency = round(time.time() - start_time, 4)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            latency_seconds=latency
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        if isinstance(e, (ValueError, KeyError)):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=f"System encountered an error processing query: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
