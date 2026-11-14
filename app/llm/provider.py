"""
Abstract LLM Provider Protocol.
Enables transparent swapping between Gemini, OpenAI, and offline Mock providers.
"""
from typing import Protocol, Type, TypeVar, Dict, Any, List
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMProvider(Protocol):
    async def generate_interpretation(
        self,
        analysis_type: str,
        measurements: Dict[str, Any],
        calculations: Dict[str, Any],
        classification: Dict[str, Any],
        evidence_text: str
    ) -> Dict[str, Any]:
        """
        Generates grounded scientific interpretation, limitations, and uncertainties.
        Must NOT alter measurements or calculations.
        """
        ...
