from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os
import threading
from dotenv import load_dotenv
load_dotenv()
from app.rag_engine import rag_engine

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        if rag_engine.retriever is None:

            print("Loading documents during startup...")

            rag_engine.load_documents(
                docs_path="./data/indian-penal-code.pdf",
                collection_name="test"
            )

            print("Documents loaded successfully.")

    except Exception as e:
        print(f"Startup loading failed: {e}")

    yield


app = FastAPI(
    title="Enterprise RAG",
    version="1.0.0",
    lifespan=lifespan
)

load_lock = threading.Lock()

# Request Validation Schemas
class LoadRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the target PDF file", example="/app/data/indian-penal-code.pdf")
    collection: str = Field(default="default", description="The ChromaDB partition collection name")

class QueryRequest(BaseModel):
    query: str
    collection: str = "default"
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    retrieval_time_ms: float
    generation_time_ms: float
    cache_hit: bool = False

@app.get("/health")
async def health():
    #return {"status": "healthy"}
    # Added health check for ChromaDB connection
    return rag_engine.health_check()

@app.post("/load")
async def load_documents(request: LoadRequest):
    global load_lock
    with load_lock:  # Only one /load at a time
        if rag_engine.retriever is not None:
            return {"status": "already_loaded", "chunks": len(rag_engine.chunks)}
        
        try:
            chunks = rag_engine.load_documents(request.file_path, request.collection)
            return {"status": "loaded", "chunks": chunks}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        answer, sources, confidence, ret_time, gen_time, cache_hit = rag_engine.query(request.query)
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            retrieval_time_ms=ret_time,
            generation_time_ms=gen_time,
            cache_hit=cache_hit
        )
    except Exception as e:
        import traceback
        print(f"ERROR in /query: {str(e)}")
        print(traceback.format_exc())  # Print full stack
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)