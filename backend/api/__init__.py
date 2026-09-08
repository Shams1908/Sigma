# API router package initialization
from .upload import router as upload_router
from .analysis import router as analysis_router
from .results import router as results_router
from .visualizations import router as visualizations_router

__all__ = [
    "upload_router",
    "analysis_router",
    "results_router",
    "visualizations_router",
]
