"""Supporting Only Document Processing Strategy.

This strategy processes only supporting documents to generate
context summaries and document listings.
"""

import time
from typing import List, Optional, Dict, Any

from src.models import (
    ConversionResult,
    ContextSummaries,
    SummaryResult,
    ShortVersionResult,
    DocumentsProducedResult,
    ProviderListingResult,
)
from src.document_summary import run_documents_summary
from src.short_version_summary import run_short_version_exhibits
from src.documents_produced import run_documents_produced_report
from src.providers_listing import run_provider_listings
from src.summary_engine.error_handling import handle_llm_errors

from ..base import ProcessingStrategy


class SupportingOnlyStrategy(ProcessingStrategy):
    """Strategy for processing only supporting documents.
    
    This strategy focuses on analyzing supporting documents to generate
    context summaries and document listings without requiring primary documents.
    
    Example:
        ```python
        strategy = SupportingOnlyStrategy()
        processor = DocumentProcessor(
            strategy=strategy,
            supporting_docs=supporting_documents
        )
        result = processor.process()
        ```
    """
    
    @handle_llm_errors("process supporting only", "document_processing")
    def process(self, document_data: Dict[str, Any]) -> SummaryResult:
        """Process only supporting documents to generate summaries.
        
        This method implements the workflow for processing supporting documents:
        1. Process the supporting documents to generate context summaries
        2. Generate a document produced string
        3. Generate a short version of the summary
        4. Generate provider listings
        
        Args:
            document_data: Dictionary with all necessary data for processing:
                - supporting_docs: List of supporting documents
                - usage: Usage tracking object
                - job_id: Job identifier
        
        Returns:
            SummaryResult: The complete summary result
            
        Raises:
            ValueError: If supporting documents are missing
        """
        # Extract data from the document_data dictionary
        supporting_docs = document_data.get("supporting_docs", [])
        job_id = document_data.get("job_id")
        
        if not supporting_docs:
            raise ValueError("Supporting documents are required for this strategy")
        
        start_time = time.time()
        
        # Process the supporting documents
        context_summaries = self.process_supporting_documents(supporting_docs)
        
        # Generate the documents produced string
        documents_produced_result = self.generate_document_produced_string(context_summaries)
        
        # Generate the short version for exhibits
        draft_report = self._generate_draft_report(context_summaries)
        short_version_result = self._generate_short_version_exhibits(draft_report)
        
        # Generate the provider listing
        provider_listing_result = self.generate_providers_listing(None, supporting_docs)
        
        # Track processing time
        duration = time.time() - start_time
        
        # Construct and return the final result
        return SummaryResult(
            doc_id=job_id or "",
            provider_listing=provider_listing_result.provider_listing if provider_listing_result else "",
            long_version=draft_report,
            short_version=short_version_result.short_version if short_version_result else "",
            documents_produced=documents_produced_result.documents_produced if documents_produced_result else "",
            exhibits_research="",  # No exhibits research without primary document
            processing_time=duration,
        )
    
    @handle_llm_errors("process supporting documents", "document_processing")
    def process_supporting_documents(self, documents: List[ConversionResult]) -> ContextSummaries:
        """Process supporting documents to generate context summaries.
        
        Args:
            documents: List of supporting documents to process
            
        Returns:
            ContextSummaries: Summaries of the supporting documents
            
        Example:
            ```python
            context_summaries = strategy.process_supporting_documents(supporting_docs)
            ```
        """
        return run_documents_summary(documents)
    
    @handle_llm_errors("generate document produced string", "document_processing")
    def generate_document_produced_string(
        self, context_summaries: ContextSummaries
    ) -> DocumentsProducedResult:
        """Generate a formatted string of documents that were produced.
        
        Args:
            context_summaries: Summaries of the context documents
            
        Returns:
            DocumentsProducedResult: Formatted string of produced documents
        """
        return run_documents_produced_report(context_summaries)
    
    @handle_llm_errors("generate providers listing", "document_processing")
    def generate_providers_listing(
        self,
        primary_docs: Optional[List[ConversionResult]] = None,
        supporting_docs: Optional[List[ConversionResult]] = None,
    ) -> ProviderListingResult:
        """Generate a listing of all providers found in the supporting documents.
        
        Args:
            primary_docs: Not used in this strategy
            supporting_docs: List of supporting documents
            
        Returns:
            ProviderListingResult: Listing of all providers found
        """
        return run_provider_listings(None, supporting_docs)
    
    def _generate_draft_report(self, context_summaries: ContextSummaries) -> str:
        """Generate a draft report from the context summaries.
        
        Args:
            context_summaries: Summaries of the supporting documents
            
        Returns:
            str: A draft report based on the summaries
        """
        # Create a simple report from all summaries
        sections = []
        
        # Add a header
        sections.append("# SUPPORTING DOCUMENTS SUMMARY")
        sections.append("")
        
        # Add each document summary
        for doc in context_summaries.summaries:
            sections.append(f"## {doc.title}")
            sections.append(doc.summary)
            sections.append("")
        
        # Join all sections
        return "\n".join(sections)
    
    @handle_llm_errors("generate short version", "document_processing")
    def _generate_short_version_exhibits(self, draft_report: str) -> ShortVersionResult:
        """Generate a shorter version of the supporting documents summary.
        
        Args:
            draft_report: The draft report to condense
            
        Returns:
            ShortVersionResult: The condensed short version
        """
        return run_short_version_exhibits(draft_report) 