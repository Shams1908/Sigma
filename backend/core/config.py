from typing import List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class HypothesisScoringSettings(BaseModel):
    """
    Configuration for hypothesis candidate scoring and confidence ranking.
    All weights and softmax temperature are fully configurable.
    """
    weight_ml: float = Field(default=0.30, description="Weight w1 for ML classification confidence")
    weight_symbol_rate: float = Field(default=0.20, description="Weight w2 for symbol-rate parameter agreement")
    weight_snr: float = Field(default=0.10, description="Weight w3 for SNR evidence/feasibility")
    weight_constellation: float = Field(default=0.15, description="Weight w4 for constellation agreement")
    weight_timing: float = Field(default=0.10, description="Weight w5 for timing/synchronization quality")
    weight_fec: float = Field(default=0.10, description="Weight w6 for FEC validation")
    weight_bitstream: float = Field(default=0.05, description="Weight w7 for bitstream correlation")
    weight_interleaver: float = Field(default=0.0, description="Weight for interleaver provenance (strictly 0.0 in P5.5)")
    temperature: float = Field(default=0.25, gt=0.0, description="Softmax temperature for confidence normalization")
    confidence_tolerance: float = Field(default=1e-5, description="Numerical tolerance for confidence sum validation")

class HypothesisSearchSettings(BaseModel):
    """
    Configuration for candidate hypothesis search space generation.
    Controls ML pruning, symbol-rate candidate expansion, and total candidate bounds.
    """
    default_top_k_modulations: int = Field(default=5, ge=1, description="Default top-K modulations to retain from ML")
    min_ml_probability: float = Field(default=0.01, ge=0.0, le=1.0, description="Minimum ML probability threshold to consider")
    max_symbol_rate_candidates: int = Field(default=3, ge=1, description="Maximum number of symbol-rate candidates per modulation")
    fallback_uncertainty_fraction: float = Field(default=0.05, gt=0.0, description="Fraction of Baud rate to use as step when uncertainty is unavailable")
    max_candidates: int = Field(default=24, ge=1, description="Maximum total candidate hypotheses generated across all dimensions")

class Settings(BaseSettings):
    PROJECT_NAME: str = "SIGMA"
    API_V1_STR: str = "/api/v1"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
    ]

    # Hypothesis Scoring Configuration
    HYPOTHESIS_SCORING: HypothesisScoringSettings = HypothesisScoringSettings()

    # Hypothesis Generation & Search Configuration
    HYPOTHESIS_SEARCH: HypothesisSearchSettings = HypothesisSearchSettings()

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore",
    )

settings = Settings()


