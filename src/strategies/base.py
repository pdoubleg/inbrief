"""Base classes and interfaces for the summary engine V2.

This module defines the abstract interfaces that processing strategies
must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.models import (
    ContextSummaries,
    ConversionResult,
    DiscoverySummaryResult,
    MedicalRecordsSummaryResult,
    SummaryResult,
    DocumentsProducedResult,
    ProviderListingResult,
)
from src.context.input import ProcessingInput
from src.llm.error_handling import handle_llm_errors


class ProcessingStrategy(ABC):
    """Abstract base class for document processing strategies.
    
    This class defines the interface that all document processing strategies
    must implement. It follows the Strategy Pattern to allow for different
    processing pipelines to be interchangeable.
    
    Each concrete strategy should implement the process method, which is the
    main entry point for document processing. It should return a SummaryResult object.
    """
    
    @abstractmethod
    def process(self, input_data: ProcessingInput) -> SummaryResult:
        """Process the documents according to the strategy.
        
        This is the main method that all concrete strategies must implement.
        It should coordinate the overall processing flow specific to each strategy by
        taking in a ProcessingInput object and returning a SummaryResult object.
        
        Args:
            input_data: Validated input parameters for processing
            
        Returns:
            A SummaryResult containing the processing results
        """
        pass
    
    @handle_llm_errors("process supporting documents", "document_processing")
    def process_supporting_documents(self, documents: List[ConversionResult]) -> ContextSummaries:
        """Process supporting documents to generate context summaries.
        
        Args:
            documents: List of supporting documents to process
            
        Returns:
            ContextSummaries: Summaries of supporting documents
            
        Raises:
            ValueError: If documents is empty
        """
        from src.modules.document_summary import run_documents_summary
        
        if not documents:
            raise ValueError("No supporting documents provided")
        
        return run_documents_summary(documents)
    
    @handle_llm_errors("process medical records", "medical_processing")
    def generate_medical_records_summary(self, documents: List[ConversionResult]) -> MedicalRecordsSummaryResult:
        """Generate a summary of medical records.
        
        Args:
            documents: List of medical record documents to process
            
        Returns:
            MedicalRecordsSummaryResult: Summary of medical records
            
        Raises:
            ValueError: If documents is empty
        """
        from src.modules.medical_records_summary import run_medical_records_summary
        
        if not documents:
            raise ValueError("No medical records provided")
        
        return run_medical_records_summary(documents)
    
    @handle_llm_errors("generate high level summary", "document_processing")
    def generate_high_level_summary(
        self, discovery_document: str, supporting_documents: str = ""
    ) -> DiscoverySummaryResult:
        """Generate a high-level summary of the discovery document.
        
        Args:
            discovery_document: The main document to summarize
            supporting_documents: Optional supporting context
            
        Returns:
            DiscoverySummaryResult: The high-level summary
        """
        from src.modules.discovery_summary import run_discovery_summary
        
        return run_discovery_summary(discovery_document, supporting_documents)
    
    @handle_llm_errors("generate document produced string", "document_processing")
    def generate_document_produced_string(
        self, context_summaries: ContextSummaries
    ) -> DocumentsProducedResult:
        """Generate a summary of the documents that were produced.
        
        Args:
            context_summaries: Summaries of supporting documents
            
        Returns:
            DocumentsProducedResult: The documents produced summary
        """
        from src.modules.documents_produced import run_documents_produced_report
        
        return run_documents_produced_report(context_summaries)
    
    @handle_llm_errors("generate providers listing", "document_processing")
    def generate_providers_listing(
        self,
        primary_docs: Optional[List[ConversionResult]] = None,
        supporting_docs: Optional[List[ConversionResult]] = None,
    ) -> ProviderListingResult:
        """Generate a listing of all providers found in the documents.
        
        Args:
            primary_docs: Optional list of primary documents
            supporting_docs: Optional list of supporting documents
            
        Returns:
            ProviderListingResult: Listing of all providers found
        """
        from src.modules.providers_listing import run_provider_listings
        
        return run_provider_listings(primary_docs, supporting_docs)
    