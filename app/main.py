"""
Marswalk Mission Planner — Scientific AI & RAG Prediction Engine (Component A).
Main FastAPI application entrypoint.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.intelligence import router as intelligence_router
from app.rag.vector_store import ScientificVectorStore

app = FastAPI(
    title="Marswalk Scientific AI & RAG Prediction Engine",
    description="Dedicated scientific analysis and prediction engine for simulated astronaut EVA missions across Jezero Crater on Mars.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration (Leaflet UI / Vite / React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Intelligence Router
app.include_router(intelligence_router)

@app.get("/")
async def root():
    return {
        "engine": "Marswalk Scientific Intelligence Engine",
        "role": "Component A (Scientific Analysis & Prediction)",
        "mission_target": "Jezero Crater, Mars (18.38°N, 77.58°E)",
        "documentation": "/docs",
        "status": "operational"
    }

@app.get("/api/health")
async def health_check():
    store = ScientificVectorStore()
    chunk_count = store.collection.count() if store.collection else len(store._fallback_corpus)
    return {
        "status": "healthy",
        "chromadb_loaded_chunks": chunk_count,
        "available_services": [
            "terrain_service",
            "route_service",
            "eva_budget_service",
            "environmental_meda_service",
            "radiation_rad_service",
            "space_weather_donki_service",
            "rag_evidence_retriever"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
