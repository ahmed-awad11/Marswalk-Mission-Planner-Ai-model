"""
Offline Ingestion Script for Marswalk Authoritative Scientific Knowledge Base.
Parses NASA documents, extracts metadata, chunks text, and indexes vectors into ChromaDB.
"""
from pathlib import Path
import sys

# Add parent directory to sys.path so app imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import DOCS_DIR, CHROMA_DIR
from app.rag.chunker import DocumentChunker
from app.rag.vector_store import ScientificVectorStore

def main():
    print(f"[*] Starting Knowledge Base Ingestion...")
    print(f"[*] Reading documents from: {DOCS_DIR}")
    print(f"[*] ChromaDB output path: {CHROMA_DIR}")
    
    chunker = DocumentChunker(target_chunk_size=1200, overlap_size=200)
    store = ScientificVectorStore(persist_dir=CHROMA_DIR)
    
    doc_files = list(DOCS_DIR.glob("*.md"))
    if not doc_files:
        print("[!] No markdown documents found in data/documents.")
        return
        
    total_chunks = 0
    for doc_file in doc_files:
        chunks = chunker.process_document(doc_file)
        store.upsert_chunks(chunks)
        print(f"  [+] Ingested {len(chunks)} chunks from: {doc_file.name}")
        total_chunks += len(chunks)
        
    print(f"[SUCCESS] Ingested {total_chunks} authoritative chunks across {len(doc_files)} scientific papers.")

if __name__ == "__main__":
    main()
