"""
Google Gemini LLM Provider with structured JSON output and mock fallback.
"""
import json
from typing import Dict, Any, Optional
from app.core.config import settings
from app.llm.mock_provider import MockLLMProvider
from app.llm.prompt_builder import SYSTEM_PROMPT, build_interpretation_prompt

class GeminiLLMProvider:
    def __init__(self, api_key: Optional[str] = None, model: str = settings.GEMINI_MODEL):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model
        self.mock_provider = MockLLMProvider()
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    async def generate_interpretation(
        self,
        analysis_type: str,
        measurements: Dict[str, Any],
        calculations: Dict[str, Any],
        classification: Dict[str, Any],
        evidence_text: str
    ) -> Dict[str, Any]:
        """
        Generates grounded scientific interpretation using Gemini API or offline fallback.
        """
        # If no client or API key, use verified deterministic mock provider
        if not self.client:
            return await self.mock_provider.generate_interpretation(
                analysis_type, measurements, calculations, classification, evidence_text
            )

        prompt = build_interpretation_prompt(
            analysis_type, measurements, calculations, classification, evidence_text
        )

        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.2,  # Low temperature for scientific fidelity
                )
            )
            parsed = json.loads(response.text)
            return {
                "interpretation": parsed.get("interpretation", ""),
                "limitations": parsed.get("limitations", []),
                "uncertainty": parsed.get("uncertainty", []),
            }
        except Exception as e:
            # Fall back safely on API error
            res = await self.mock_provider.generate_interpretation(
                analysis_type, measurements, calculations, classification, evidence_text
            )
            res["uncertainty"].append(f"LLM live API call fell back to offline engine: {str(e)}")
            return res
