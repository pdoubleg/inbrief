"""Refactored SummaryEngine for document processing.

This module contains the refactored SummaryEngine class that coordinates
document processing using the strategy pattern and pydantic-ai models.
"""

import time
from typing import List

from src.models import ConversionResult
from src.summary_engine.error_handling import log_execution_time
from src.models import SummaryResult
from src.summary_engine.processing_strategies import (
    ProcessingStrategy,
    PrimaryAndSupportingStrategy,
    PrimaryOnlyStrategy,
    MedicalOnlyStrategy,
    SupportingOnlyStrategy,
)
from ..utils import log_exception


class SummaryEngine:
    """Engine for processing and summarizing documents.

    This class coordinates the processing of different document types
    (primary, supporting, medical) using the strategy pattern and
    pydantic-ai models for LLM interactions.

    Attributes:
        primary_docs: Primary documents to be processed
        supporting_docs: Supporting documents to be processed
        has_medical_records: Whether medical records are included
        job_id: Unique identifier for the current job
        include_exhibits_research: Whether to include exhibits research
        _cost: Running total of the cost of LLM calls
    """

    def __init__(
        self,
        primary_docs: List[ConversionResult] | None = None,
        supporting_docs: List[ConversionResult] | None = None,
        model_info_path: str = "./models.json",
        has_medical_records: bool = False,
        job_id: str | None = None,
        include_exhibits_research: bool = False,
    ):
        """Initialize the SummaryEngine.

        Args:
            primary_docs: Primary documents to be processed
            supporting_docs: Supporting documents to be processed
            model_info_path: Path to the model information file
            has_medical_records: Whether medical records are included
            job_id: Unique identifier for the current job
            include_exhibits_research: Whether to include exhibits research
        """

        # Store configuration
        self.primary_docs = primary_docs or None
        self.supporting_docs = supporting_docs or None
        self.has_medical_records = has_medical_records
        self.job_id = job_id
        self.include_exhibits_research = include_exhibits_research
        self._cost = 0

    @log_execution_time()
    def run_summarization_process(self, medical_only: bool = False) -> SummaryResult:
        """Main function to run the inBrief summary process.

        This method determines the appropriate processing strategy based on
        the available documents and medical_only flag, then executes that strategy.
        It handles the high-level flow control and error management.

        Args:
            medical_only: Whether to process only medical records

        Returns:
            A SummaryResult object

        Raises:
            ValueError: If required documents are missing
            RuntimeError: If processing fails
        """
        start_time = time.perf_counter()

        try:
            # Get the appropriate strategy based on document availability
            strategy = self._get_strategy(medical_only)

            # Execute the strategy
            result = strategy.process(self)

            # Print cost information
            self._print_cost()

            # Print total run time
            elapsed_time = time.perf_counter() - start_time
            print(f"Total run time: {round(elapsed_time, 2)} seconds")

            return result

        except Exception as e:
            # Handle unexpected errors
            error_message = "Unexpected error during document processing"
            from src.summary_engine.error_handling import friendly_error

            error_traceback = friendly_error(e, error_message)

            log_exception(
                job_id=self.job_id,
                model="OpenAIModel.GPT_4O.value",
                error_message=error_message,
                error_category="document_processing",
                error_traceback=error_traceback,
            )

            raise RuntimeError(f"Document processing failed: {str(e)}") from e

    def _get_strategy(self, medical_only: bool) -> ProcessingStrategy:
        """Get the appropriate processing strategy.

        Args:
            medical_only: Whether to process only medical records

        Returns:
            The appropriate processing strategy

        Raises:
            ValueError: If the required documents for the strategy are missing
        """
        if medical_only:
            if not self.supporting_docs:
                raise ValueError(
                    "Supporting documents are required for medical-only processing."
                )
            return MedicalOnlyStrategy()

        if self.primary_docs is not None and self.supporting_docs is not None:
            return PrimaryAndSupportingStrategy()

        if self.primary_docs is not None and self.supporting_docs is None:
            return PrimaryOnlyStrategy()

        if self.primary_docs is None and self.supporting_docs is not None:
            return SupportingOnlyStrategy()

        raise ValueError(
            "No documents provided. Please provide primary and/or supporting documents."
        )

    def _print_cost(self):
        """Print the total cost of the summarization process."""
        print(f"Total cost: ${self._cost:.4f}")

    def log_task_completion(self, task_name: str, start_time: float):
        """Log the completion of a task.

        Args:
            task_name: Name of the task
            start_time: Start time of the task in seconds
        """
        elapsed_time = time.perf_counter() - start_time
        print(f"Task '{task_name}' completed in {round(elapsed_time, 2)} seconds")
