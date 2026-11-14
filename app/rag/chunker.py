"""
Document Chunker and Markdown Frontmatter Parser.
Extracts structured scientific metadata and splits documents into overlapping context windows.
"""
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

class DocumentChunker:
    def __init__(self, target_chunk_size: int = 1500, overlap_size: int = 250):
        self.target_chunk_size = target_chunk_size  # characters (~300-400 words)
        self.overlap_size = overlap_size

    def parse_markdown_file(self, file_path: Path) -> Tuple[Dict[str, Any], str]:
        """
        Parses YAML frontmatter and body from a markdown document.
        """
        text = file_path.read_text(encoding="utf-8")
        frontmatter = {}
        body = text
        
        # Match YAML frontmatter between --- and ---
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
        if fm_match:
            raw_fm, body = fm_match.groups()
            for line in raw_fm.strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    frontmatter[key] = val
                    
        return frontmatter, body.strip()

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text by markdown paragraphs with sliding overlap.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            current_chunk.append(para)
            current_length += len(para) + 2
            
            if current_length >= self.target_chunk_size:
                joined = "\n\n".join(current_chunk)
                chunks.append(joined)
                
                # Keep last paragraph for overlap continuity
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0
                    
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks

    def process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Processes a file into a list of chunk dicts ready for embedding and vector indexing.
        """
        frontmatter, body = self.parse_markdown_file(file_path)
        chunks = self.chunk_text(body)
        
        doc_id = frontmatter.get("document_id", file_path.stem)
        processed = []
        
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}-CHK-{idx+1:03d}"
            metadata = dict(frontmatter)
            metadata["chunk_id"] = chunk_id
            metadata["chunk_index"] = idx
            metadata["source_file"] = file_path.name
            
            processed.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": metadata
            })
            
        return processed
