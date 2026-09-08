import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
try:
    from core.config import settings
    from api import upload_router, analysis_router, results_router
except ImportError:
    from backend.core.config import settings
    from backend.api import upload_router, analysis_router, results_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="SIGMA — Signal Intelligence & Guided Modulation Analysis API",
    version="0.1.0"
)

# CORS middleware configuration
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Register routers with prefix
app.include_router(upload_router, prefix=f"{settings.API_V1_STR}/upload", tags=["Upload"])
app.include_router(analysis_router, prefix=f"{settings.API_V1_STR}/analysis", tags=["Analysis"])
app.include_router(results_router, prefix=f"{settings.API_V1_STR}/results", tags=["Results"])

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint returning system status.
    """
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
