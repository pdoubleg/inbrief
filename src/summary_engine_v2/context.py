"""Context for the Strategy Pattern implementation.

This module contains the DocumentProcessor class that serves as the context
for the Strategy Pattern, allowing strategies to be switched at runtime.
"""

import time
from typing import Dict, List, Optional, Any, Type

from pydantic_ai.usage import Usage

from src.models import (
    ConversionResult,
    SummaryResult,
)
from .base import ProcessingStrategy


class DocumentProcessor:
    """Context class for the Strategy Pattern that processes documents.
    
    This class maintains a reference to a strategy object and delegates
    the actual processing work to the strategy. Clients can swap strategies
    at runtime to change how documents are processed.
    
    Attributes:
        strategy: The current processing strategy
        primary_docs: List of primary documents to process
        supporting_docs: List of supporting documents to process
        has_medical_records: Flag indicating if medical records are present
        job_id: Unique identifier for the current job
        include_exhibits_research: Flag for including exhibits research
        usage: Tracks LLM usage and costs
    
    Example:
        ```python
        # Initialize with a specific strategy
        processor = DocumentProcessor(
            strategy=PrimaryAndSupportingStrategy(),
            primary_docs=primary_documents,
            supporting_docs=supporting_documents
        )
        
        # Process documents
        result = processor.process()
        
        # Switch strategy and process again
        processor.set_strategy(MedicalOnlyStrategy())
        result = processor.process()
        ```
    """
    
    def __init__(
        self,
        strategy: Optional[ProcessingStrategy] = None,
        primary_docs: Optional[List[ConversionResult]] = None,
        supporting_docs: Optional[List[ConversionResult]] = None,
        has_medical_records: bool = False,
        job_id: Optional[str] = None,
        include_exhibits_research: bool = False,
        model_info_path: str = "./models.json",
    ):
        """Initialize the DocumentProcessor.
        
        Args:
            strategy: The processing strategy to use
            primary_docs: List of primary documents to process
            supporting_docs: List of supporting documents to process
            has_medical_records: Flag indicating if medical records are present
            job_id: Unique identifier for the current job
            include_exhibits_research: Flag for including exhibits research
            model_info_path: Path to the model information file
        """
        self.strategy = strategy
        self.primary_docs = primary_docs or []
        self.supporting_docs = supporting_docs or []
        self.has_medical_records = has_medical_records
        self.job_id = job_id
        self.include_exhibits_research = include_exhibits_research
        
        # Initialize usage tracking
        self.usage = Usage(model_info_path)
        
        # Task timing storage
        self._task_times: Dict[str, float] = {}
    
    def set_strategy(self, strategy: ProcessingStrategy) -> None:
        """Set the processing strategy.
        
        This method allows changing the processing strategy at runtime.
        
        Args:
            strategy: The new processing strategy to use
        """
        self.strategy = strategy
    
    def process(self) -> SummaryResult:
        """Process documents using the current strategy.
        
        Returns:
            SummaryResult: The result of document processing
            
        Raises:
            ValueError: If no strategy is set or if required documents are missing
        """
        if not self.strategy:
            raise ValueError("No processing strategy set")
        
        # Prepare document data for the strategy
        document_data = {
            "primary_docs": self.primary_docs,
            "supporting_docs": self.supporting_docs,
            "has_medical_records": self.has_medical_records,
            "job_id": self.job_id,
            "include_exhibits_research": self.include_exhibits_research,
            "usage": self.usage,
        }
        
        # Delegate processing to the strategy
        return self.strategy.process(document_data)
    
    def log_task_start(self, task_name: str) -> None:
        """Record the start time of a task.
        
        Args:
            task_name: Name of the task being started
        """
        self._task_times[task_name] = time.time()
    
    def log_task_completion(self, task_name: str) -> float:
        """Record the completion of a task and return its duration.
        
        Args:
            task_name: Name of the completed task
            
        Returns:
            float: Duration of the task in seconds
            
        Raises:
            KeyError: If the task wasn't started
        """
        if task_name not in self._task_times:
            raise KeyError(f"Task {task_name} was not started")
        
        start_time = self._task_times[task_name]
        duration = time.time() - start_time
        
        print(f"Task '{task_name}' completed in {duration:.2f} seconds")
        return duration
    
    def print_cost(self) -> None:
        """Print the total cost of LLM calls made during processing."""
        total_cost = self.usage.get_cost()
        print(f"Total cost: ${total_cost:.4f}") 