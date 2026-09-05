"""
Hypothesis engine package.

Entry point: run_hypothesis_pipeline(storage_path) → GeneratorResult
"""
from hypothesis.generator import GeneratorResult, run_hypothesis_pipeline

__all__ = ["run_hypothesis_pipeline", "GeneratorResult"]
