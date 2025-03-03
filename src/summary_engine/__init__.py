"""SummaryEngine package for document processing.

This package provides a refactored implementation of the SummaryEngine
for document processing, using pydantic-ai for LLM operations and the
Strategy pattern for different document processing scenarios.
"""

from src.summary_engine.engine import SummaryEngine

from src.summary_engine.processing_strategies import (
    ProcessingStrategy,
    PrimaryAndSupportingStrategy,
    PrimaryOnlyStrategy,
    MedicalOnlyStrategy,
)
from src.summary_engine.error_handling import (
    handle_llm_errors,
    log_execution_time,
    friendly_error,
)

__all__ = [
    # Main engine
    "SummaryEngine",
    # Models
    "DocumentChunkInput",
    "DocumentSummaryOutput",
    "PrimarySummaryInput",
    "PrimarySummaryOutput",
    "MedicalRecordsSummaryInput",
    "MedicalRecordsSummaryOutput",
    "ShortVersionInput",
    "ShortVersionOutput",
    "SummaryResult",
    # Processing strategies
    "ProcessingStrategy",
    "PrimaryAndSupportingStrategy",
    "PrimaryOnlyStrategy",
    "MedicalOnlyStrategy",
    # Error handling
    "handle_llm_errors",
    "retry_on_error",
    "log_execution_time",
    "friendly_error",
]
