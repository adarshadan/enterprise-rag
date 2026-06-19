import os
import time
import logging
import re
import json
import threading  # Added for thread-safe file operations
import numpy as np
from typing import List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from langsmith import traceable

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure Production Logging Layout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAGEngine")

LOCAL_MODEL_PATH = "/app/models/all-mpnet-base-v2"
CACHE_FILE = "./data/custom_semantic_cache.json"

class RAGEngine:
    """Hybrid Retrieval-Augmented Generation Engine with a thread-safe Custom Semantic Cache."""

    def __init__(self):
        # Thread safety primitive for disk I/O operations
        self.cache_lock = threading.Lock()
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.critical("GROQ_API_KEY environment variable is entirely missing.")
            raise ValueError("Production configuration requires GROQ_API_KEY.")

        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=groq_api_key,
            temperature=0.3,
            request_timeout=15  # Increased timeout from 5s to 15s for high-load reliability
        )
        
        model_path_or_name = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else os.getenv("HF_EMBEDDING_MODEL")
        logger.info(f"Initializing HuggingFaceEmbeddings using: {model_path_or_name}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_path_or_name)
        except Exception as e:
            logger.critical(f"Failed to instantiate embedding pipeline model layers: {str(e)}")
            raise e
        
        self.vectorstore = None
        self.retriever = None
        self.chunks = []
        self.sections = {}
        
        try:
            os.makedirs("./data", exist_ok=True)
            self.cache_data = self._load_cache()
            logger.info(f"Loaded semantic cache with {len(self.cache_data)} active historical pairs.")
        except Exception as e:
            logger.error(f"Failed to cleanly provision data volume maps: {str(e)}")
            self.cache_data = []

    def health_check(self) -> dict:
        """Performs system health diagnostics."""
        try:
            if not self.llm:
                return {"status": "unhealthy", "reason": "LLM not initialized"}
            if not self.embeddings:
                return {"status": "unhealthy", "reason": "Embeddings model not loaded"}
            
            test_embed = self.embeddings.embed_query("test")
            if not test_embed:
                return {"status": "unhealthy", "reason": "Embeddings inference failed"}
            if self.retriever is None:
                return {"status": "degraded", "reason": "No documents loaded. Call /load first"}
            
            return {"status": "healthy"}
        except Exception as e:
            logger.error(f"Health check encountered exception: {str(e)}")
            return {"status": "unhealthy", "reason": str(e)}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _invoke_llm_with_retry(self, chain, context: str, question: str) -> str:
        """Invoke LLM with automatic retry logic on remote endpoint dropouts."""
        return chain.invoke({"context": context, "question": question})

    def _load_cache(self) -> list:
        """Loads semantic cache records under an isolation lock."""
        with self.cache_lock:
            if os.path.exists(CACHE_FILE):
                try:
                    with open(CACHE_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data if isinstance(data, list) else []
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to read cache database, starting fresh: {e}")
            return []

    def _save_cache(self):
        """Persists custom cache memory records safely utilizing thread-locking protection."""
        with self.cache_lock:
            try:
                # Atomic-like switch pattern using a temporary tracking file to prevent corruption
                temp_file = f"{CACHE_FILE}.tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
                os.replace(temp_file, CACHE_FILE)
            except IOError as e:
                logger.error(f"Disk IO error writing updates to semantic cache: {e}")

    def reconstruct_sections(self, chunks: List) -> dict:
        """Reconstructs complete sections from fragmented document pieces."""
        sections = {}
        current_section = None
        current_text = ""
        
        for chunk in chunks:
            match = re.search(r'Section (\d+)', chunk.page_content)
            if match:
                section_num = match.group(1)
                if current_section:
                    sections[current_section] = current_text
                current_section = section_num
                current_text = chunk.page_content
            else:
                if current_section:
                    current_text += "\n" + chunk.page_content
        
        if current_section:
            sections[current_section] = current_text
        return sections

    def rerank_docs(self, docs: List, query: str) -> List:
        """Reranks retrieved context documents by term frequency matching."""
        scored = []
        words = [w.lower() for w in query.split()]
        for doc in docs:
            content_lower = doc.page_content.lower()
            score = sum(1 for word in words if word in content_lower)
            scored.append((score, doc))
        return [doc for _, doc in sorted(scored, key=lambda x: x[0], reverse=True)]

    def load_documents(self, docs_path: str, collection_name: str) -> int:
        """Loads, chunks, indexes, and builds hybrid ensemble retrieval engines."""
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Source file not found at: {docs_path}")

        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        loader = PyPDFLoader(docs_path)
        pages = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=['Section ', '\n\n', '\n', '.', ' ', '']
        )
        
        chunks = splitter.split_documents(pages)
        sections = self.reconstruct_sections(chunks)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=f"./data/{collection_name}"
        )
        
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 10
        
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.6, 0.4]
        )

        # Apply state atomic switch safely at the end of execution process
        self.chunks = chunks
        self.sections = sections
        self.vectorstore = vectorstore
        self.retriever = retriever
        
        return len(chunks)

    @traceable(name="rag_query", tags=["query"])
    def query(self, *args, **kwargs) -> Tuple[str, List[str], float, float, float, bool]:
        """Runs the semantic cache lookups, falling back to hybrid RAG if cache misses."""
        
        # Extract the question whether it's passed positionally or via a keyword argument
        if "question" in kwargs:
            question = kwargs["question"]
        elif args:
            question = args[0]
        else:
            raise ValueError("No question provided to the query engine.")

        if not self.retriever:
            raise ValueError("No collection loaded. Please ingest a document via /load first.")
        
        start_cache_check = time.time()
        
        try:
            query_vector = np.array(self.embeddings.embed_query(question))
            query_norm = np.linalg.norm(query_vector)
        except Exception as e:
            logger.error(f"Embedding computation failure: {str(e)}")
            query_vector, query_norm = None, 0.0

        best_match_answer = None
        highest_similarity = -1.0
        
        # Read matching cache iteration arrays safely
        current_cache = self._load_cache()
        if query_vector is not None and query_norm > 0:
            for item in current_cache:
                cached_vector = np.array(item["embedding"])
                cached_norm = np.linalg.norm(cached_vector)
                
                if cached_norm == 0:
                    continue
                    
                similarity = np.dot(query_vector, cached_vector) / (query_norm * cached_norm)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_answer = item["answer"]
        
        cache_check_time = (time.time() - start_cache_check) * 1000
        
        if highest_similarity >= 0.95:
            logger.info(f"🚀 Cache Hit! Semantic match similarity score: {highest_similarity:.4f}")
            return best_match_answer, ["Cached Context"], 1.0, cache_check_time, 0.0, True
            
        logger.info(f"⚡ Cache Miss ({highest_similarity:.4f}). Executing Hybrid Retrieval...")
        
        # --- RETRIEVAL PHASE ---
        start_retrieval = time.time()
        try:
            raw_docs = self.retriever.invoke(question)
            reranked_docs = self.rerank_docs(raw_docs, question)
            docs = reranked_docs[:3]
        except Exception as e:
            logger.error(f"Ensemble engine search error: {str(e)}")
            docs = []
        retrieval_time = (time.time() - start_retrieval) * 1000
        
        # --- CONTEXT STRATEGY PHASE ---
        match = re.search(r'(\d+)', question)
        if match and (match.group(1) in self.sections):
            section_num = match.group(1)
            context = self.sections[section_num]
        else:
            context = '\n\n'.join([d.page_content for d in docs])
        
        sources = list(set([d.metadata.get('source', 'unknown') for d in docs])) if docs else ["No Context Source"]
        
        # --- GENERATION PHASE ---
        start_generation = time.time()
        prompt = ChatPromptTemplate.from_template(
            "Answer based on context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            answer = self._invoke_llm_with_retry(chain, context, question)
        except Exception as ex:
            logger.error(f"❌ LLM inference failed after execution retries: {str(ex)}")
            raise RuntimeError(f"LLM service downstream unavailable: {str(ex)}")
            
        generation_time = (time.time() - start_generation) * 1000
        confidence = 0.9 if len(context) > 800 else 0.6
        
        # Append and write back to memory maps safely under isolation checks
        if query_vector is not None and answer:
            try:
                self.cache_data.append({
                    "question": question,
                    "answer": answer,
                    "embedding": query_vector.tolist()
                })
                self._save_cache()
            except Exception as cache_ex:
                logger.warning(f"Failed to record entry to database layout layers: {str(cache_ex)}")
        
        return answer, sources, confidence, retrieval_time, generation_time, False

rag_engine = RAGEngine()