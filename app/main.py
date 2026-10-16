from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routes import router, init_mongodb_indexes
from app.database import check_mongodb


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CEFM Backend",
    description=(
        "Cross-modal Explainable Framework "
        "for Melanoma Assessment"
    ),
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ROUTES
# ============================================================

app.include_router(
    router,
    prefix="/api/v1",
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    if check_mongodb():
        print("MongoDB connected successfully.")
        init_mongodb_indexes()
    else:
        print("WARNING: MongoDB connection failed.")


# ============================================================
# ROOT
# ============================================================

@app.get("/", include_in_schema=False)
def root():

    index = Path(__file__).parent / "static" / "index.html"

    if index.exists():
        return FileResponse(index)

    return {
        "message": "Welcome to CEFM Backend",
        "version": "2.0.0",
        "database": "MongoDB",
        "docs": "/docs",
        "health": "/api/v1/health",
    }