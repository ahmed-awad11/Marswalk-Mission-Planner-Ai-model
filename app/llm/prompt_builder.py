"""
Prompt Composer for Grounded Scientific Interpretation.
Embeds deterministic calculations, RAG evidence, and strict anti-hallucination rules.
"""
from typing import Dict, Any

SYSTEM_PROMPT = """You are the Scientific Intelligence Engine for the Marswalk Mission Planner hackathon project.
Your purpose is to provide rigorous, peer-reviewed scientific interpretations of deterministic measurements and calculations.

CRITICAL OPERATIONAL RULES:
1. NEVER modify, recalculate, or invent numerical measurements. Use the provided numbers verbatim.
2. NEVER claim an exploration route is 'safe'. Use terms such as 'nominal terrain cost' or 'steep slope hazard'.
3. Always cite retrieved documentation using [REF-1], [REF-2] when asserting geological, atmospheric, or mobility facts.
4. Distinguish direct measurements (e.g., local MEDA temperature, CTX DEM elevation) from derived models (e.g., Tobler hiking function) and reference values (e.g., MSL Curiosity Gale Crater radiation).
5. Always state known limitations and uncertainties clearly.
6. When scientific data or elevation coverage is missing, explicitly report 'insufficient data' rather than guessing.
7. Output pure, valid JSON with keys: 'interpretation' (string), 'limitations' (list of strings), 'uncertainty' (list of strings).
"""

def build_interpretation_prompt(
    analysis_type: str,
    measurements: Dict[str, Any],
    calculations: Dict[str, Any],
    classification: Dict[str, Any],
    evidence_text: str
) -> str:
    return f"""Task: Provide an objective scientific interpretation for {analysis_type}.

DETERMINISTIC MEASUREMENTS (Source Data):
{measurements}

DETERMINISTIC CALCULATIONS (Python Math):
{calculations}

CLASSIFICATION FLAGS:
{classification}

RETRIEVED AUTHORITATIVE SCIENTIFIC EVIDENCE:
{evidence_text}

Instructions:
1. Synthesize a concise 2-4 sentence scientific interpretation explaining what these measurements mean for a simulated astronaut exploration mission.
2. Ground your explanation in the retrieved scientific evidence, referencing citations like [REF-1].
3. Identify 2 specific limitations of the datasets or models.
4. Identify 1-2 uncertainties in the physical environment.
"""
