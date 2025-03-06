"""Factory for creating document processing strategies.

This module provides a factory for creating the appropriate strategy
based on the input document types and processing flags.
"""

from src.models import ProcessingType
from src.context.input import ProcessingInput
from src.strategies.base import ProcessingStrategy
from src.strategies.primary_only import PrimaryOnlyStrategy
from src.strategies.primary_and_supporting import PrimaryAndSupportingStrategy
from src.strategies.medical_only import MedicalOnlyStrategy
from src.strategies.supporting_only import SupportingOnlyStrategy




class StrategyFactory:
    """Factory to determine and create the appropriate processing strategy.
    
    This factory works in conjunction with the ProcessingInput model to automatically
    determine and instantiate the appropriate strategy based on the input data.
    
    The factory relies on the processing_type property of ProcessingInput,
    which is computed based on the available documents and processing flags.
    
    Example:
        ```python
        # Input is already validated by Pydantic
        input_data = ProcessingInput(primary_docs=[doc1, doc2])
        
        # Factory automatically determines the correct strategy
        strategy = StrategyFactory.create_strategy(input_data)
        
        # strategy will be an instance of PrimaryOnlyStrategy
        ```
    """
    
    @staticmethod
    def create_strategy(input_data: ProcessingInput) -> ProcessingStrategy:
        """Create the appropriate strategy based on the input data.
        
        This method uses the processing_type property from the input_data
        to determine which strategy to instantiate. The ProcessingInput validation
        ensures that a valid strategy type is always available.
        
        Args:
            input_data: The validated processing input with computed processing_type
            
        Returns:
            The appropriate concrete ProcessingStrategy instance
            
        Raises:
            ValueError: If the processing_type doesn't match any known strategy
            
        Example:
            ```python
            # Create input with primary documents only
            input_data = ProcessingInput(primary_docs=[doc1, doc2])
            
            # Get primary-only strategy
            strategy = StrategyFactory.create_strategy(input_data)
            
            # Process documents
            result = strategy.process(input_data)
            ```
        """
        if input_data.processing_type == ProcessingType.PRIMARY_AND_SUPPORTING:
            return PrimaryAndSupportingStrategy()
        elif input_data.processing_type == ProcessingType.PRIMARY_ONLY:
            return PrimaryOnlyStrategy()
        elif input_data.processing_type == ProcessingType.SUPPORTING_ONLY:
            return SupportingOnlyStrategy()
        elif input_data.processing_type == ProcessingType.MEDICAL_ONLY:
            return MedicalOnlyStrategy()
        else:
            raise ValueError(f"Unknown processing type: {input_data.processing_type}")