import os
import time
import logging
import re
import json
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
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAGEngine")

# Absolute paths inside Docker runtime environment
LOCAL_MODEL_PATH = "/app/models/all-mpnet-base-v2"
CACHE_FILE = "./data/custom_semantic_cache.json"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")

# Verify env vars are set
print(f"LANGSMITH_API_KEY: {os.getenv('LANGSMITH_API_KEY')}")
print(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
print(f"LANGCHAIN_TRACING_V2: {os.environ.get('LANGCHAIN_TRACING_V2')}")

class RAGEngine:
    """Hybrid Retrieval-Augmented Generation Engine with a custom Semantic Cache.
    
    This class orchestrates vector document ingestions, text restructuring, hybrid multi-stage 
    retrieval (BM25 + Chroma Vector DB), semantic cache lookups using custom Cosine Similarity, 
    and context-aware completions through Groq Cloud API LLMs..
    """

    def __init__(self):
        """Initializes components and verifies local data environments."""
        # 1. Initialize LLM Component with tight temperature boundary for consistency
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.critical("GROQ_API_KEY environment variable is entirely missing.")
            raise ValueError("Production configuration requires GROQ_API_KEY.")

        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=groq_api_key,
            temperature=0.3,
            request_timeout=5
        )
        
        # 2. Wire Offline Embedding Layer with local-first disk fallback mapping
        model_path_or_name = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else os.getenv("HF_EMBEDDING_MODEL")
        logger.info(f"Initializing HuggingFaceEmbeddings using: {model_path_or_name}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_path_or_name)
        except Exception as e:
            logger.critical(f"Failed to instantiate embedding pipeline model layers: {str(e)}")
            raise e
        
        # Internal state placeholders for document parsing structures
        self.vectorstore = None
        self.retriever = None
        self.chunks = []
        self.sections = {}
        
        # 3. Safely initialize directory bindings and local cache memory state
        try:
            os.makedirs("./data", exist_ok=True)
            self.cache_data = self._load_cache()
            logger.info(f"Loaded semantic cache with {len(self.cache_data)} active historical pairs.")
        except Exception as e:
            logger.error(f"Failed to cleanly provision data volume maps: {str(e)}")
            self.cache_data = []

    def health_check(self) -> dict:
        """Performs comprehensive system health diagnostics.
        
        Returns:
            dict: Health status with detailed failure reasons if applicable.
        """
        try:
            # Check 1: LLM connectivity
            if not self.llm:
                return {"status": "unhealthy", "reason": "LLM not initialized"}
            
            # Check 2: Embeddings model loaded
            if not self.embeddings:
                return {"status": "unhealthy", "reason": "Embeddings model not loaded"}
            
            test_embed = self.embeddings.embed_query("test")
            if not test_embed:
                return {"status": "unhealthy", "reason": "Embeddings inference failed"}
            
            # Check 3: Retriever state
            if self.retriever is None:
                return {"status": "degraded", "reason": "No documents loaded. Call /load first"}
            
            # Check 4: Vector store accessible
            if self.vectorstore is None:
                return {"status": "degraded", "reason": "Vector store not initialized"}
            
            logger.info("✅ All health checks passed")
            return {"status": "healthy"}
        
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "reason": str(e)}
    
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )

    def _invoke_llm_with_retry(self, chain, context: str, question: str) -> str:
        """Invoke LLM with automatic retry logic.
        
        Retries up to 3 times with exponential backoff (2s, 4s, 8s).
        """
        try:
            result = chain.invoke({"context": context, "question": question})
            logger.info("✅ LLM inference succeeded")
            return result
        except Exception as e:
            logger.warning(f"⚠️ LLM call failed, retrying... Error: {str(e)}")
            raise
    

    def _load_cache(self) -> list:
        """Loads semantic cache database records from disk safely.
        
        Returns:
            list: Array of dictionary objects containing queries, answers, and vector lists.
        """
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    logger.warning("Cache data structure corrupted. Overwriting layout map.")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read cache database, starting fresh: {e}")
        return []

    def _save_cache(self):
        """Persists custom semantic cache memory tracking arrays safely to JSON disk layout."""
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"Disk IO error writing updates to semantic cache system layout: {e}")

    def reconstruct_sections(self, chunks: List) -> dict:
        """Reconstructs complete, unstructured logical sections from split document fragments.
        
        Args:
            chunks (List[Document]): LangChain core Document pieces extracted from splitter.
            
        Returns:
            dict: Text mapping layout categorized under specific legal section strings.
        """
        sections = {}
        current_section = None
        current_text = ""
        
        for chunk in chunks:
            # Look specifically for structural regular expressions matching statutory indices
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
        
        # Catch lingering trailing section block entries
        if current_section:
            sections[current_section] = current_text
        return sections

    def rerank_docs(self, docs: List, query: str) -> List:
        """Reranks retrieved context documents by local term frequency relevance algorithms.
        
        Args:
            docs (List[Document]): Aggregated context materials from search pipelines.
            query (str): The initial conversational query string parameters.
            
        Returns:
            List[Document]: Reranked collections sorted in descending score alignments.
        """
        scored = []
        words = [w.lower() for w in query.split()]
        
        for doc in docs:
            content_lower = doc.page_content.lower()
            # Absolute term match calculation across token fragments
            score = sum(1 for word in words if word in content_lower)
            scored.append((score, doc))
            
        # Sort collections by absolute score counts descending
        return [doc for _, doc in sorted(scored, key=lambda x: x[0], reverse=True)]


    def load_documents(self, docs_path: str, collection_name: str):
        """Loads, chunks, indexes, and instantiates hybrid ensemble retrieval modules.
        
        Args:
            docs_path (str): The absolute disk system path routing to the source PDF asset.
            collection_name (str): Unique identifying key mapping destination database names.
            
        Raises:
            FileNotFoundError: If the document asset target path does not resolve correctly.
        """
        if not os.path.exists(docs_path):
            logger.error(f"Document insertion targeted an invalid asset string path: {docs_path}")
            raise FileNotFoundError(f"Source legal text artifact not found at: {docs_path}")

        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        try:
            logger.info(f"Loading document from path: {docs_path}")
            loader = PyPDFLoader(docs_path)
            pages = loader.load()
            
            # Use deliberate layout delimiters mirroring legislative paragraphing strategies
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150,
                separators=['Section ', '\n\n', '\n', '.', ' ', '']
            )
            
            self.chunks = splitter.split_documents(pages)
            self.sections = self.reconstruct_sections(self.chunks)
            logger.info(f"Split document into {len(self.chunks)} chunks and parsed {len(self.sections)} sections.")
            
            # Persist and initialize native vector collection spaces
            self.vectorstore = Chroma.from_documents(
                documents=self.chunks,
                embedding=self.embeddings,
                persist_directory=f"./data/{collection_name}"
            )
            
            # Extract retrievers explicitly from vector db structures
            chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            bm25_retriever = BM25Retriever.from_documents(self.chunks)
            bm25_retriever.k = 10
            
            # Bind both pipelines together into a balanced hybrid ensemble search layout
            logger.info("Configuring EnsembleRetriever weights [BM25: 0.6, Semantic: 0.4]")
            self.retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever],
                weights=[0.6, 0.4]
            )
        except Exception as e:
            logger.error(f"Critical execution error tracking document ingestion workloads: {str(e)}")
            raise e

    @traceable(name="rag_query", tags=["query"])
    def query(self, question: str) -> Tuple[str, List[str], float, float, float, bool]:
        """Runs the semantic cache pipeline, falling back to hybrid RAG if cache misses.
        
        Args:
            question (str): Plain text prompt passed upstream via endpoint connections.
            
        Returns:
            Tuple[str, List[str], float, float, float, bool]: Complete collection mapping containing 
                (Answer, Source List, Confidence, Cache Check Time, Generation Time, Cache Hit Boolean).
        """
        if not self.retriever:
            logger.warning("Query received before any context database was loaded.")
            raise ValueError("No collection loaded. Please ingest a document via /load first.")
        
        start_cache_check = time.time()
        
        try:
            # Compute query vector array structures
            query_vector = np.array(self.embeddings.embed_query(question))
            query_norm = np.linalg.norm(query_vector)
        except Exception as e:
            logger.error(f"Local text embedding computation failed at query stage: {str(e)}")
            query_vector, query_norm = None, 0.0

        best_match_answer = None
        highest_similarity = -1.0
        
        # Execute custom cosine semantic tracking scan over active persistent memory cache pairs
        if query_vector is not None and query_norm > 0:
            for item in self.cache_data:
                cached_vector = np.array(item["embedding"])
                cached_norm = np.linalg.norm(cached_vector)
                
                if cached_norm == 0:
                    continue
                    
                # Cosine Similarity Vector Calculation: (A • B) / (||A|| * ||B||)
                similarity = np.dot(query_vector, cached_vector) / (query_norm * cached_norm)
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_answer = item["answer"]
        
        cache_check_time = (time.time() - start_cache_check) * 1000
        logger.info(f"🔍 Evaluated top cache similarity score: {highest_similarity:.4f}")
        
        # Strict 0.95 semantic threshold match constraint execution block
        if highest_similarity >= 0.95:
            logger.info(f"🚀 Cache Hit! Reusing response context via semantic similarity ({highest_similarity:.4f})")
            return best_match_answer, ["Cached Context"], 1.0, cache_check_time, 0.0, True
            
        logger.info(f"⚡ Cache Miss. Executing Hybrid RAG pipeline for: '{question}'")
        
        # --- RETRIEVAL PHASE ---
        start_retrieval = time.time()
        try:
            raw_docs = self.retriever.invoke(question)
            reranked_docs = self.rerank_docs(raw_docs, question)
            docs = reranked_docs[:3]
        except Exception as e:
            logger.error(f"Ensemble engine failed querying information matrices: {str(e)}")
            docs = []
        retrieval_time = (time.time() - start_retrieval) * 1000
        
        # --- CONTEXT STRATEGY PHASE ---
        # Direct statutory code injection mechanism routing context explicitly to known sections
        match = re.search(r'(\d+)', question)
        if match and (match.group(1) in self.sections):
            section_num = match.group(1)
            context = self.sections[section_num]
            logger.info(f"🎯 Context Router: Direct statutory text bound targeting Section {section_num}")
        else:
            context = '\n\n'.join([d.page_content for d in docs])
            logger.info("🎯 Context Router: Relying on aggregated hybrid search chunks")
        
        sources = list(set([d.metadata.get('source', 'unknown') for d in docs])) if docs else ["No Context Source Found"]
        
        # --- GENERATION PHASE ---
        start_generation = time.time()
        prompt = ChatPromptTemplate.from_template(
            "Answer based on context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            answer = self._invoke_llm_with_retry(chain, context, question)
        except Exception as ex:
            logger.error(f"❌ LLM inference failed after 3 retries: {str(ex)}")
            raise RuntimeError(f"LLM service unavailable after retries: {str(ex)}")
            
        generation_time = (time.time() - start_generation) * 1000
        confidence = 0.9 if len(context) > 800 else 0.6
        
        # Update custom local semantic cache memory maps with fresh computation outputs
        if query_vector is not None and answer:
            try:
                self.cache_data.append({
                    "question": question,
                    "answer": answer,
                    "embedding": query_vector.tolist()
                })
                self._save_cache()
                logger.info("Successfully saved query pair directly to cache database.")
            except Exception as cache_ex:
                logger.warning(f"Failed to record entry to database layer layout: {str(cache_ex)}")
        
        return answer, sources, confidence, retrieval_time, generation_time, False


# Instantiate globally for simple imports across operational runtime worker wrappers
rag_engine = RAGEngine()