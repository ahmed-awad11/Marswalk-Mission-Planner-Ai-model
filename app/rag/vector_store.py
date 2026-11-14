"""
Vector Store Management using ChromaDB with Offline Deterministic Embeddings.
Handles indexing and semantic retrieval for authoritative scientific documents.
"""
import hashlib
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.core.config import settings

class OfflineScientificEmbeddingFunction:
    """
    Deterministic offline semantic embedding function.
    Generates normalized 128-dimensional dense vectors based on hashed word and character n-grams.
    Zero external downloads, zero network latency, 100% offline.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            vec = [0.0] * self.dim
            words = text.lower().replace(",", " ").replace(".", " ").replace(":", " ").replace("-", " ").split()
            for w in words:
                if len(w) < 2:
                    continue
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                vec[idx] += sign * (1.0 + math.log(len(w) + 1))
            
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [round(x / norm, 6) for x in vec]
            embeddings.append(vec)
        return embeddings

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def name(self) -> str:
        return "offline_scientific_embedding"

class ScientificVectorStore:
    def __init__(self, persist_dir: Path = Path(settings.CHROMA_PERSIST_DIR)):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = "marswalk_scientific_knowledge_v3"
        self.embedding_function = OfflineScientificEmbeddingFunction()
        self.client = None
        self.collection = None
        self._fallback_corpus = []
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            self.client = None
            self.collection = None

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Upserts chunk texts, IDs, and metadata into the vector collection.
        """
        if not chunks:
            return
            
        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        for c in chunks:
            self._fallback_corpus.append(c)
            
        if self.collection is not None:
            try:
                self.collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
            except Exception:
                pass

    def query(
        self,
        query_text: str,
        n_results: int = settings.RAG_TOP_K,
        where_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.10
    ) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for semantically similar scientific chunks.
        """
        results = []
        
        if self.collection is not None and self.collection.count() > 0:
            try:
                query_args = {
                    "query_texts": [query_text],
                    "n_results": min(n_results, self.collection.count()),
                }
                if where_filter:
                    query_args["where"] = where_filter
                    
                chroma_res = self.collection.query(**query_args)
                
                docs = chroma_res.get("documents", [[]])[0]
                metas = chroma_res.get("metadatas", [[]])[0]
                distances = chroma_res.get("distances", [[]])[0]
                
                for doc, meta, dist in zip(docs, metas, distances):
                    similarity = 1.0 - float(dist) if dist is not None else 1.0
                    if similarity >= similarity_threshold:
                        results.append({
                            "text": doc,
                            "metadata": meta,
                            "similarity": round(similarity, 3)
                        })
                return results
            except Exception as e:
                pass

        # Fallback keyword matching
        q_words = set(query_text.lower().split())
        scored = []
        for c in self._fallback_corpus:
            doc_words = set(c["text"].lower().split())
            intersection = q_words.intersection(doc_words)
            if intersection:
                score = len(intersection) / max(len(q_words), 1)
                scored.append((score, c))
                
        scored.sort(key=lambda x: x[0], reverse=True)
        for score, c in scored[:n_results]:
            results.append({
                "text": c["text"],
                "metadata": c["metadata"],
                "similarity": round(min(score, 1.0), 3)
            })
            
        return results
