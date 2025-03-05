"""Context for the Strategy Pattern implementation.

This module contains the DocumentProcessor class that serves as the context
for the Strategy Pattern, allowing strategies to be switched at runtime.
"""

from typing_extensions import Self
from typing import Dict, List, Optional

from pydantic import BaseModel, ValidationError, Field, computed_field, model_validator
from pydantic_ai.usage import Usage

from src.models import (
    ConversionResult,
    SummaryResult,
)
from .strategies.factory import StrategyFactory
from .base import ProcessingStrategy
from src.models import ProcessingType


class ProcessingInput(BaseModel):
    """Unified input model for all processing strategies."""
    # Common fields
    job_id: Optional[str] = None
    usage: Optional[Usage] = None
    
    # Strategy-specific fields (all optional)
    primary_docs: Optional[List[ConversionResult]] = Field(default=None, description="Primary documents to process")
    supporting_docs: Optional[List[ConversionResult]] = Field(default=None, description="Supporting documents for context")
    include_exhibits_research: bool = Field(default=False, description="Whether to include exhibits research")
    has_medical_records: bool = Field(default=False, description="Whether to process as medical records")
    
    _processing_type: Optional[ProcessingType] = None
    
    @model_validator(mode='after')
    def validate_processing_type(self) -> Self:
        """Validates that input documents can be associated with at least one valid processing strategy.
        
        This validator ensures the input data contains enough information to determine
        a valid processing strategy. It confirms at least one of these conditions is met:
        1. Medical records with primary documents
        2. Primary and supporting documents
        3. Primary documents only
        4. Supporting documents only
        
        Returns:
            Self: The validated model instance if validation passes
            
        Raises:
            ValueError: If no valid processing strategy can be determined from the input
            
        Example:
            ```python
            # Valid - contains primary docs
            input_data = ProcessingInput(primary_docs=[doc1, doc2])
            
            # Invalid - contains no documents
            try:
                input_data = ProcessingInput()  # Will raise ValueError
            except ValueError as e:
                print(f"Validation failed: {e}")
            ```
        """
        if self._processing_type is not None and isinstance(self._processing_type, ProcessingType):
            return self
        
        has_primary = self.primary_docs is not None and len(self.primary_docs) > 0
        has_supporting = self.supporting_docs is not None and len(self.supporting_docs) > 0
        is_medical = self.has_medical_records
        
        if is_medical and has_primary:
            return self
        elif has_primary and has_supporting:
            return self
        elif has_primary:
            return self
        elif has_supporting:
            return self
        else:
            raise ValueError("Cannot determine processing type strategy based on provided input")

    @computed_field
    @property
    def processing_type(self) -> ProcessingType:
        """Determines the appropriate processing strategy type based on the input configuration.
        
        This property analyzes the provided documents and flags to select the
        appropriate processing strategy. The logic prioritizes in this order:
        1. Medical records with supporting docs → MEDICAL_AND_PRIMARY
        2. Both primary and supporting docs → PRIMARY_AND_SUPPORTING
        3. Only primary docs → PRIMARY_ONLY
        4. Only supporting docs → SUPPORTING_ONLY
        
        Note that this property is only called after validation, so one of these
        conditions is guaranteed to be true.
        
        Returns:
            ProcessingType: The enum value representing the appropriate processing strategy
            
        Example:
            ```python
            # Will return ProcessingType.PRIMARY_ONLY
            input_data = ProcessingInput(primary_docs=[doc1])
            strategy_type = input_data.processing_type
            
            # Will return ProcessingType.PRIMARY_AND_SUPPORTING
            input_data = ProcessingInput(
                primary_docs=[doc1], 
                supporting_docs=[doc2]
            )
            strategy_type = input_data.processing_type
            ```
        """
        if self._processing_type is not None and isinstance(self._processing_type, ProcessingType):
            return self._processing_type
        
        has_primary = self.primary_docs is not None and len(self.primary_docs) > 0
        has_supporting = self.supporting_docs is not None and len(self.supporting_docs) > 0
        is_medical = self.has_medical_records
        
        if is_medical:
            return ProcessingType.MEDICAL_ONLY
        elif has_primary and has_supporting:
            return ProcessingType.PRIMARY_AND_SUPPORTING
        elif has_primary:
            return ProcessingType.PRIMARY_ONLY
        elif has_supporting:
            return ProcessingType.SUPPORTING_ONLY
        else:
            raise ValueError("Cannot determine processing type strategy based on provided input")
        



class DocumentProcessor:
    """Context class for the Strategy Pattern that processes documents.
    
    This class maintains a reference to a strategy object and delegates
    the actual processing work to the strategy. Clients can swap strategies
    at runtime to change how documents are processed.
    
    The processor works with the ProcessingInput model which handles validation
    and strategy type determination. If no strategy is explicitly provided,
    an appropriate one will be automatically selected using the StrategyFactory.
    
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
        # Initialize with documents (strategy will be auto-selected)
        processor = DocumentProcessor(
            primary_docs=primary_documents,
            supporting_docs=supporting_documents
        )
        
        # Process documents
        result = processor.process()
        
        # Or explicitly set a strategy
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
        
        # Task timing storage
        self._task_times: Dict[str, float] = {}
    
    
    def process(self, input_data: ProcessingInput) -> SummaryResult:
        """Process documents using the current strategy.
        
        This method coordinates document processing by:
        1. Ensuring a valid strategy exists (using the factory if needed)
        2. Delegating processing to the strategy
        3. Returning the results
        
        If no strategy is set, one will be automatically created using
        the StrategyFactory based on the input_data.
        
        Args:
            input_data: Validated ProcessingInput containing all documents
                and processing parameters
                
        Returns:
            SummaryResult: The complete result of document processing
            
        Raises:
            ValueError: If no strategy can be determined
            
        Example:
            ```python
            # Process with existing strategy
            input_data = ProcessingInput(primary_docs=[doc1, doc2])
            result = processor.process(input_data)
            
            # Or create a new ProcessingInput with different parameters
            new_input = ProcessingInput(
                primary_docs=[doc3],
                supporting_docs=[doc4, doc5],
                include_exhibits_research=True
            )
            result = processor.process(new_input)
            ```
        """
        if not self.strategy:        
            # Create appropriate strategy
            self.strategy = StrategyFactory.create_strategy(input_data)
        
        # Process using selected strategy
        return self.strategy.process(input_data)
    

    def set_strategy(self, strategy: ProcessingStrategy):
        """Sets the processing strategy for the DocumentProcessor.
        
        Args:
            strategy: The ProcessingStrategy to set
        """
        self.strategy = strategy
