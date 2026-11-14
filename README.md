# Marswalk Mission Planner — Scientific AI & RAG Prediction Engine

**Component A: Headless Scientific Intelligence Engine**  
*48-Hour NASA-Themed Hackathon Project — Jezero Crater Extravehicular Activity (EVA) Simulation*

---

## 1. Overview & Architecture Scope

The **Marswalk Scientific AI & RAG Prediction Engine** is a dedicated analytical microservice built for the Marswalk Mission Planner. It evaluates simulated astronaut surface exploration across Jezero Crater on Mars using genuine NASA/USGS datasets.

### Architectural Boundary: Component A vs. Component B
- **Component A (This Engine):** Headless, API-driven analytical intelligence system. Operates deterministically via Python, NumPy, Rasterio, pyproj, and scikit-image, enriched by a ChromaDB vector store of authoritative NASA literature. It emits strongly typed, schema-validated JSON payloads. **Zero conversational UI, zero chat memory.**
- **Component B (Marswalk Chatbot - Out of Scope):** Downstream conversational interface that consumes Component A's structured JSON outputs to explain findings to judges and users.

```mermaid
flowchart TD
    Client["Marswalk Application / Leaflet UI"] -->|POST /api/intelligence/*| API["FastAPI Endpoints"]
    Chatbot["Component B: Marswalk Chatbot"] -.->|Reads Structured Analysis| API
    
    subgraph IntelligenceEngine ["Component A: Scientific Intelligence Engine"]
        API --> EngineCore["Calculation & Validation Router"]
        
        subgraph DeterministicCalc ["Deterministic Calculation Layer (Zero Hallucination)"]
            TerrainService["Terrain Service (Horn's Slope & Profiling)"]
            RouteService["Route Service (scikit-image route_through_array)"]
            EVAService["EVA Service (Modified Tobler Hiking Model)"]
            RadiationService["Radiation Service (MSL RAD Gale Scaling)"]
            EnvService["Environmental Service (Perseverance MEDA)"]
            SpaceWeatherService["Space Weather Service (CCMC DONKI Rules)"]
        end
        
        subgraph RAGKnowledge ["RAG Retrieval Layer (ChromaDB)"]
            ChromaDB[("ChromaDB Vector Store (15 Curated Chunks)")]
            Retriever["Scientific Retriever (Metadata Filtering)"]
        end
        
        subgraph LLMSynthesis ["Grounded LLM Layer"]
            PromptComposer["Strict Anti-Hallucination Prompt Composer"]
            LLMProvider["Gemini 2.5 Flash / Resilient Offline Provider"]
        end
        
        EngineCore --> DeterministicCalc
        DeterministicCalc -->|Measurements + Calculations| PromptComposer
        EngineCore --> Retriever
        ChromaDB --> Retriever
        Retriever -->|Citations [REF-X]| PromptComposer
        PromptComposer --> LLMProvider
        LLMProvider -->|Structured JSON| API
    end
```

---

## 2. Scientific Knowledge Base & Datasets

### Curated Knowledge Base (`data/documents/`)
Pre-indexed and embedded in ChromaDB with full provenance metadata:
1. `01_jezero_geology_farley2020.md`: Jezero delta front stratigraphy and geomorphology (*Science*, 2021).
2. `02_perseverance_delta_mangold2021.md`: Crater floor lithologies (Máaz & Séítah formations) and sand ripple hazards (*JGR Planets*, 2020).
3. `03_usgs_ctx_hirise_topography.md`: USGS planetary topography standards, CTX 20m & HiRISE 2m error margins and 15° slope boundaries.
4. `04_meda_instrument_manfredi2021.md`: Mars 2020 MEDA sensor baseline calibration and local boundary layer limitations (*Space Science Reviews*, 2021).
5. `05_msl_rad_surface_radiation_hassler2014.md`: Curiosity RAD surface dose rates (0.21 mGy/day reference) and Gale-to-Jezero transfer limits (*Science*, 2014).
6. `06_donki_space_weather_nasa_sp2021.md`: NASA CCMC DONKI event thresholds (SEPs, CMEs, Flares) and storm shelter protocols (*NASA SP-20210014389*).
7. `07_tobler_hiking_suit_mobility.md`: Human extraterrestrial locomotion, Tobler's hiking adaptation, and spacesuit penalty factor $k_{\text{suit}} = 0.60$.

---

## 3. Core API Endpoints

All endpoints are hosted under `/api/intelligence`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/intelligence/terrain-analysis` | Computes elevation delta, average slope, max slope, and steep segment detection. |
| `POST` | `/api/intelligence/route-evaluation` | Computes traversability cost using $1 + (\theta/15.0)^2$ and scikit-image pathfinding. |
| `POST` | `/api/intelligence/eva-estimation` | Calculates walk velocities via Tobler's function, stop allocations, and 20% safety margin. |
| `POST` | `/api/intelligence/environmental-analysis` | Parses Perseverance MEDA atmospheric readings for temperature, pressure, and wind gusts. |
| `POST` | `/api/intelligence/radiation-estimation` | Calculates cumulative absorbed (mGy) and equivalent (mSv) doses from MSL RAD reference. |
| `POST` | `/api/intelligence/space-weather-analysis` | Checks recent 7-day DONKI solar flares/SEPs and emits `"Caution: shelter-ready"` or `"Nominal"`. |
| `POST` | `/api/intelligence/mission-analysis` | Master endpoint providing an executive mission briefing aggregating all modules. |

---

## 4. Verification and Testing

The engine includes 11 automated test suites in `tests/test_intelligence.py` covering all required mission edge cases:

```bash
# Run the test suite
python -m pytest -v
```

### Verified Test Scenarios:
1. **TEST-01 (Flat Terrain):** Verifies slope $< 2^\circ$ classifies as `"Flat / Easy"` with zero steep segments.
2. **TEST-02 (Steep Terrain):** Verifies slope $> 15^\circ$ flags `"Severe / Exceeds Slope Limit"` and `is_within_limits: false`.
3. **TEST-03 (Invalid Profile):** Verifies exception handling on malformed or single-point coordinate arrays.
4. **TEST-04 (EVA Budget Exceeded):** Verifies warning trigger when mission duration exceeds life-support safety margin.
5. **TEST-05 (Route Comparison):** Verifies that a gentle $3^\circ$ detour has lower terrain cost than a direct $12^\circ$ traverse.
6. **TEST-06 (Missing MEDA Data):** Verifies unrecorded Sols return `null` measurements with an explicit refusal to fabricate weather.
7. **TEST-07 (DONKI Cache Resilience):** Verifies offline disk snapshot fallback during NASA API outages.
8. **TEST-08 (Radiation Exact Math):** Verifies exact numerical equality for $D = 0.21 \times (t/24)$.
9. **TEST-09 (API Terrain Endpoint):** Verifies end-to-end FastAPI integration, Pydantic contracts, and RAG citations.
10. **TEST-10 (Radiation Provenance):** Verifies that Gale Crater is explicitly identified as a transferred reference, preventing Jezero confusion.
11. **TEST-11 (Master Mission Endpoint):** Verifies aggregated multi-module analysis and `"GO / HOLD"` recommendation logic.

---

## 5. Quickstart & Deployment

### Virtual Environment Setup
```powershell
cd C:\Users\ahm06\.gemini\antigravity\scratch\marswalk-intelligence
.venv\Scripts\activate
```

### Start the Server
```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is live at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **Health Check:** `http://localhost:8000/api/health`
