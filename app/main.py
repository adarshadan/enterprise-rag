from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()
from app.rag_engine import rag_engine

app = FastAPI(title="Enterprise RAG", version="1.0.0")

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
    # Added validation block
    if not request.file_path.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF files are accepted.")
    
    try:
        rag_engine.load_documents(request.file_path, request.collection)
        return {"status": "loaded", "chunks": len(rag_engine.chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        # Captured cache_hit status dynamically from the engine array destruction
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
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)