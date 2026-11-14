"""
Scientific Evidence Retriever.
Formats vector store search results into structured citations and LLM context blocks.
"""
from typing import List, Dict, Any, Optional
from app.rag.vector_store import ScientificVectorStore
from app.models.contracts import EvidenceItem

class ScientificRetriever:
    def __init__(self, vector_store: ScientificVectorStore):
        self.vector_store = vector_store

    def retrieve_evidence(
        self,
        query: str,
        topic: Optional[str] = None,
        top_k: int = 3,
        threshold: float = 0.0
    ) -> List[EvidenceItem]:
        """
        Retrieves top-k authoritative chunks and formats them as EvidenceItem objects.
        """
        where_filter = {"scientific_topic": topic} if topic else None
        raw_results = self.vector_store.query(
            query_text=query,
            n_results=top_k,
            where_filter=where_filter,
            similarity_threshold=threshold
        )
        
        evidence_items = []
        for idx, res in enumerate(raw_results):
            meta = res.get("metadata", {})
            text = res.get("text", "")
            
            # Extract first ~250 chars of relevant snippet
            snippet = text.split("\n")[0] if "\n" in text else text[:250]
            
            item = EvidenceItem(
                citation_key=f"REF-{idx+1}",
                title=meta.get("title", "NASA Scientific Document"),
                source_url=meta.get("source_url", "https://pds.nasa.gov"),
                organization=meta.get("organization", "NASA / JPL / USGS"),
                relevant_text=snippet
            )
            evidence_items.append(item)
            
        return evidence_items

    def format_evidence_block(self, evidence: List[EvidenceItem]) -> str:
        """
        Formats evidence items into an unambiguous text block for LLM prompt injection.
        """
        if not evidence:
            return "No high-confidence authoritative documentation found for this query."
            
        blocks = []
        for item in evidence:
            blocks.append(
                f"[{item.citation_key}] {item.title} ({item.organization})\n"
                f"Source: {item.source_url}\n"
                f"Evidence: {item.relevant_text}\n"
            )
        return "\n".join(blocks)
