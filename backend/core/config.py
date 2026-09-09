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
    temperature: float = Field(default=0.25, gt=0.0, description="Softmax temperature for confidence normalization")
    confidence_tolerance: float = Field(default=1e-5, description="Numerical tolerance for confidence sum validation")

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

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore",
    )

settings = Settings()

