from typing import List, Optional, Self
from pydantic import BaseModel, Field, computed_field, model_validator

from src.models import ProcessingType, ConversionResult


class ProcessingInput(BaseModel):
    """Unified input model for all processing strategies."""
    # Common
    job_id: Optional[str] = None
    
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