"""Base module for document processing strategies.

Defines the abstract base class for all document processing strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from src.models import (
    ContextSummaries,
    ConversionResult,
    DiscoverySummaryResult,
    MedicalRecordsSummaryResult,
    SummaryResult,
    DocumentsProducedResult,
    ProviderListingResult,
)
from src.summary_engine.error_handling import handle_llm_errors


class ProcessingStrategy(ABC):
    """Abstract base class for document processing strategies.
    
    This class defines the interface that all document processing strategies
    must implement. It follows the Strategy Pattern to allow for different
    processing algorithms to be interchangeable.
    
    Each concrete strategy should implement the process method, which is the
    main entry point for document processing.
    """
    
    @abstractmethod
    def process(self, document_data: Dict[str, Any]) -> SummaryResult:
        """Process the documents according to the strategy.
        
        This is the main method that all concrete strategies must implement.
        It coordinates the overall processing flow specific to each strategy.
        
        Args:
            document_data: Dictionary containing all necessary data for processing,
                including primary documents, supporting documents, and settings.
                
        Returns:
            SummaryResult: The result of the document processing.
            
        Raises:
            ValueError: If required data is missing or invalid.
        """
        pass
    
    @handle_llm_errors("process supporting documents", "document_processing")
    def process_supporting_documents(self, documents: List[ConversionResult]) -> ContextSummaries:
        """Process supporting documents to generate context summaries.
        
        Args:
            documents: List of supporting documents to process.
            
        Returns:
            ContextSummaries: Summaries of the supporting documents.
            
        Example:
            ```python
            context_summaries = strategy.process_supporting_documents(supporting_docs)
            ```
        """
        # To be implemented by concrete strategies if needed
        raise NotImplementedError("This strategy does not support processing supporting documents")
    
    @handle_llm_errors("process medical records", "medical_processing")
    def process_medical_records(self, documents: List[ConversionResult]) -> MedicalRecordsSummaryResult:
        """Process medical records to generate summaries.
        
        Args:
            documents: List of medical record documents to process.
            
        Returns:
            MedicalRecordsSummaryResult: Summary of the medical records.
            
        Example:
            ```python
            medical_summary = strategy.process_medical_records(medical_docs)
            ```
        """
        # To be implemented by concrete strategies if needed
        raise NotImplementedError("This strategy does not support processing medical records")
    
    @handle_llm_errors("compress primary document", "document_processing")
    def compress_primary_document(self, doc: ConversionResult) -> ContextSummaries:
        """Compress the primary document for more efficient processing.
        
        Args:
            doc: The primary document to compress.
            
        Returns:
            ContextSummaries: Compressed summary of the document.
            
        Example:
            ```python
            compressed_doc = strategy.compress_primary_document(primary_doc)
            ```
        """
        # To be implemented by concrete strategies if needed
        raise NotImplementedError("This strategy does not support compressing primary documents")
    
    @handle_llm_errors("generate high level summary", "document_processing")
    def generate_high_level_summary(
        self, discovery_document: str, supporting_documents: str = ""
    ) -> DiscoverySummaryResult:
        """Generate a high-level summary of the discovery document.
        
        Args:
            discovery_document: The main document to summarize.
            supporting_documents: Optional supporting context.
            
        Returns:
            DiscoverySummaryResult: The high-level summary.
            
        Example:
            ```python
            high_level_summary = strategy.generate_high_level_summary(doc_text, context_text)
            ```
        """
        # To be implemented by concrete strategies if needed
        raise NotImplementedError("This strategy does not support generating high-level summaries")
    
    @handle_llm_errors("generate document produced string", "document_processing")
    def generate_document_produced_string(
        self, context_summaries: ContextSummaries
    ) -> DocumentsProducedResult:
        """Generate a formatted string of documents that were produced.
        
        Args:
            context_summaries: Summaries of the context documents.
            
        Returns:
            DocumentsProducedResult: Formatted string of produced documents.
            
        Example:
            ```python
            documents_produced = strategy.generate_document_produced_string(summaries)
            ```
        """
        # To be implemented by concrete strategies if needed
        raise NotImplementedError("This strategy does not support generating document produced strings")
    
    @handle_llm_errors("generate providers listing", "document_processing")
    def generate_providers_listing(
        self,
        primary_docs: Optional[List[ConversionResult]] = None,
        supporting_docs: Optional[List[ConversionResult]] = None,
    ) -> ProviderListingResult:
        """Generate a listing of all providers found in the documents.
        
        Args:
            primary_docs: Optional list of primary documents.
            supporting_docs: Optional list of supporting documents.
            
        Returns:
            ProviderListingResult: Listing of all providers found.
            
        Example:
            ```python
            providers = strategy.generate_providers_listing(primary_docs, supporting_docs)
            ```
        """
        # To be implemented by concrete strategies if needed
        raise NotImplementedError("This strategy does not support generating provider listings") 