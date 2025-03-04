"""Examples of using the Summary Engine V2.

This module provides examples demonstrating how to use the improved
document processing architecture with the Strategy Pattern.
"""

from typing import List, Optional

from src.models import ConversionResult, SummaryResult

from .context import DocumentProcessor
from .strategies import (
    PrimaryAndSupportingStrategy,
    PrimaryOnlyStrategy,
    MedicalOnlyStrategy,
    SupportingOnlyStrategy,
)


def process_primary_and_supporting(
    primary_docs: List[ConversionResult],
    supporting_docs: List[ConversionResult],
    job_id: Optional[str] = None,
    include_exhibits_research: bool = False,
) -> SummaryResult:
    """Process primary and supporting documents together.
    
    Args:
        primary_docs: List of primary documents to process
        supporting_docs: List of supporting documents to process
        job_id: Optional job identifier
        include_exhibits_research: Whether to include exhibits research
        
    Returns:
        SummaryResult: The complete processing result
        
    Example:
        ```python
        primary_docs = [ConversionResult(...)]
        supporting_docs = [ConversionResult(...), ConversionResult(...)]
        
        result = process_primary_and_supporting(
            primary_docs=primary_docs,
            supporting_docs=supporting_docs,
            include_exhibits_research=True
        )
        ```
    """
    # Create the strategy
    strategy = PrimaryAndSupportingStrategy()
    
    # Create the document processor with the strategy
    processor = DocumentProcessor(
        strategy=strategy,
        primary_docs=primary_docs,
        supporting_docs=supporting_docs,
        job_id=job_id,
        include_exhibits_research=include_exhibits_research,
    )
    
    # Process the documents
    result = processor.process()
    
    # Print the cost (optional)
    processor.print_cost()
    
    return result


def process_primary_only(
    primary_docs: List[ConversionResult],
    job_id: Optional[str] = None,
) -> SummaryResult:
    """Process only primary documents.
    
    Args:
        primary_docs: List of primary documents to process
        job_id: Optional job identifier
        
    Returns:
        SummaryResult: The processing result
        
    Example:
        ```python
        primary_docs = [ConversionResult(...)]
        
        result = process_primary_only(
            primary_docs=primary_docs,
        )
        ```
    """
    # Create the strategy
    strategy = PrimaryOnlyStrategy()
    
    # Create the document processor with the strategy
    processor = DocumentProcessor(
        strategy=strategy,
        primary_docs=primary_docs,
        job_id=job_id,
    )
    
    # Process the documents
    result = processor.process()
    
    # Print the cost (optional)
    processor.print_cost()
    
    return result


def process_medical_only(
    medical_docs: List[ConversionResult],
    job_id: Optional[str] = None,
) -> SummaryResult:
    """Process only medical record documents.
    
    Args:
        medical_docs: List of medical documents to process
        job_id: Optional job identifier
        
    Returns:
        SummaryResult: The processing result
        
    Example:
        ```python
        medical_docs = [ConversionResult(...)]
        
        result = process_medical_only(
            medical_docs=medical_docs,
        )
        ```
    """
    # Create the strategy
    strategy = MedicalOnlyStrategy()
    
    # Create the document processor with the strategy
    processor = DocumentProcessor(
        strategy=strategy,
        primary_docs=medical_docs,
        job_id=job_id,
        has_medical_records=True, 
    )
    
    # Process the documents
    result = processor.process()
    
    # Print the cost (optional)
    processor.print_cost()
    
    return result


def process_supporting_only(
    supporting_docs: List[ConversionResult],
    job_id: Optional[str] = None,
) -> SummaryResult:
    """Process only supporting documents.
    
    Args:
        supporting_docs: List of supporting documents to process
        job_id: Optional job identifier
        
    Returns:
        SummaryResult: The processing result
        
    Example:
        ```python
        supporting_docs = [ConversionResult(...), ConversionResult(...)]
        
        result = process_supporting_only(
            supporting_docs=supporting_docs,
        )
        ```
    """
    # Create the strategy
    strategy = SupportingOnlyStrategy()
    
    # Create the document processor with the strategy
    processor = DocumentProcessor(
        strategy=strategy,
        supporting_docs=supporting_docs,
        job_id=job_id,
    )
    
    # Process the documents
    result = processor.process()
    
    # Print the cost (optional)
    processor.print_cost()
    
    return result


def demonstrate_strategy_switching() -> None:
    """Demonstrate how to switch strategies at runtime.
    
    This function shows how to create a document processor and then
    switch its strategy during processing.
    
    Example:
        ```python
        demonstrate_strategy_switching()
        ```
    """
    # Create a document processor without a strategy
    processor = DocumentProcessor()
    
    # Set up some example documents
    primary_docs = [ConversionResult(content="Primary document content", filename="primary.txt")]
    supporting_docs = [ConversionResult(content="Supporting content", filename="supporting.txt")]
    
    # Update the processor with the documents
    processor.primary_docs = primary_docs
    processor.supporting_docs = supporting_docs
    
    # Start with a primary-only strategy
    processor.set_strategy(PrimaryOnlyStrategy())
    
    # Process with the primary-only strategy
    print("Processing with primary-only strategy...")
    result1 = processor.process()
    
    # Switch to a primary-and-supporting strategy
    processor.set_strategy(PrimaryAndSupportingStrategy())
    
    # Process again with the new strategy
    print("Processing with primary-and-supporting strategy...")
    result2 = processor.process()
    
    # Print the costs
    processor.print_cost() 