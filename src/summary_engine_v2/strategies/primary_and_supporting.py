"""Primary and Supporting Documents Processing Strategy.

This strategy processes both primary and supporting documents to generate
comprehensive summaries with cross-references between document types.
"""

import time
from typing import List, Optional, Dict, Any

from src.models import (
    ConversionResult,
    ContextSummaries,
    SummaryResult,
    ShortVersionResult,
    DiscoverySummaryResult,
    ExhibitsResearchResult,
    DocumentsProducedResult,
    ProviderListingResult,
)
from src.document_summary import run_documents_summary
from src.exhibits_research import run_exhibits_research
from src.discovery_summary import run_discovery_summary
from src.short_version_summary import run_short_version
from src.documents_produced import run_documents_produced_report
from src.providers_listing import run_provider_listings
from src.utils import count_tokens
from src.summary_engine.error_handling import handle_llm_errors

from ..base import ProcessingStrategy

# Constant for maximum tokens in a primary document
MAX_SINGLE_PRIMARY_TOKENS = 40000


class PrimaryAndSupportingStrategy(ProcessingStrategy):
    """Strategy for processing both primary and supporting documents.
    
    This strategy handles the complete workflow for processing both primary
    and supporting documents, generating comprehensive summaries that
    incorporate information from both document types.
    
    Example:
        ```python
        strategy = PrimaryAndSupportingStrategy()
        processor = DocumentProcessor(strategy=strategy, ...)
        result = processor.process()
        ```
    """
    
    @handle_llm_errors("process primary and supporting", "document_processing")
    def process(self, document_data: Dict[str, Any]) -> SummaryResult:
        """Process primary and supporting documents to generate a summary.
        
        This method implements the complete workflow for processing both
        primary and supporting documents:
        1. Process supporting documents
        2. Process each primary document
        3. Generate a high-level summary
        4. Process exhibits research if requested
        5. Generate a short version of the summary
        6. Generate provider listings
        7. Create the document produced string
        
        Args:
            document_data: Dictionary with all necessary data for processing:
                - primary_docs: List of primary documents
                - supporting_docs: List of supporting documents
                - include_exhibits_research: Whether to include exhibits research
                - usage: Usage tracking object
                - job_id: Job identifier
        
        Returns:
            SummaryResult: The complete summary result
            
        Raises:
            ValueError: If required documents are missing
        """
        # Extract data from the document_data dictionary
        primary_docs = document_data.get("primary_docs", [])
        supporting_docs = document_data.get("supporting_docs", [])
        include_exhibits_research = document_data.get("include_exhibits_research", False)
        usage = document_data.get("usage")
        job_id = document_data.get("job_id")
        
        if not primary_docs:
            raise ValueError("Primary documents are required for this strategy")
        
        start_time = time.time()
        
        # Process supporting documents first
        context_summaries = None
        if supporting_docs:
            context_summaries = self.process_supporting_documents(supporting_docs)
        
        # Process each primary document
        for doc in primary_docs:
            # Check if the document is too large and needs compression
            doc_tokens = count_tokens(doc.content)
            if doc_tokens > MAX_SINGLE_PRIMARY_TOKENS:
                # Compress large primary documents
                context_summaries = self.compress_primary_document(doc)
        
        # Generate the discovery summary
        discovery_summary_result = self._generate_discovery_summary(primary_docs, context_summaries)
        
        # Process exhibits research if requested
        exhibits_research_result = None
        if include_exhibits_research and primary_docs:
            exhibits_research_result = self._run_exhibits_research(primary_docs[0], context_summaries)
        
        # Generate the short version
        short_version_result = self._generate_short_version(discovery_summary_result.long_version)
        
        # Generate the provider listing
        provider_listing_result = self.generate_providers_listing(primary_docs, supporting_docs)
        
        # Generate the documents produced string
        documents_produced_result = None
        if context_summaries:
            documents_produced_result = self.generate_document_produced_string(context_summaries)
        
        # Track processing time
        duration = time.time() - start_time
        
        # Construct and return the final result
        return SummaryResult(
            doc_id=job_id or "",
            provider_listing=provider_listing_result.provider_listing if provider_listing_result else "",
            long_version=discovery_summary_result.long_version,
            short_version=short_version_result.short_version if short_version_result else "",
            documents_produced=documents_produced_result.documents_produced if documents_produced_result else "",
            exhibits_research=exhibits_research_result.exhibits_research if exhibits_research_result else "",
            processing_time=duration,
        )
    
    @handle_llm_errors("process supporting documents", "document_processing")
    def process_supporting_documents(self, documents: List[ConversionResult]) -> ContextSummaries:
        """Process supporting documents to extract relevant context summaries.
        
        Args:
            documents: List of supporting documents to process
            
        Returns:
            ContextSummaries: Summaries of the supporting documents
        """
        return run_documents_summary(documents)
    
    @handle_llm_errors("compress primary document", "document_processing")
    def compress_primary_document(self, doc: ConversionResult) -> ContextSummaries:
        """Compress a large primary document for more efficient processing.
        
        Args:
            doc: The primary document to compress
            
        Returns:
            ContextSummaries: Compressed summary of the document
        """
        return run_documents_summary([doc])
    
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
        return run_discovery_summary(discovery_document, supporting_documents)
    
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
        """Generate a listing of all providers found in the documents.
        
        Args:
            primary_docs: Optional list of primary documents
            supporting_docs: Optional list of supporting documents
            
        Returns:
            ProviderListingResult: Listing of all providers found
        """
        return run_provider_listings(primary_docs, supporting_docs)
    
    def _generate_discovery_summary(
        self, primary_docs: List[ConversionResult], context_summaries: Optional[ContextSummaries]
    ) -> DiscoverySummaryResult:
        """Generate a discovery summary from primary documents and context.
        
        Args:
            primary_docs: List of primary documents
            context_summaries: Optional supporting context summaries
            
        Returns:
            DiscoverySummaryResult: The discovery summary
        """
        # Extract the primary document content
        discovery_document = primary_docs[0].content
        
        # Extract supporting document content if available
        supporting_documents = ""
        if context_summaries and context_summaries.summaries:
            supporting_documents = "\n\n".join(
                f"{doc.title}\n{doc.summary}" for doc in context_summaries.summaries
            )
        
        # Generate the high-level summary
        return self.generate_high_level_summary(discovery_document, supporting_documents)
    
    @handle_llm_errors("run exhibits research", "exhibits_research")
    def _run_exhibits_research(
        self, doc: ConversionResult, context_summaries: Optional[ContextSummaries]
    ) -> ExhibitsResearchResult:
        """Run exhibits research on the documents.
        
        Args:
            doc: The primary document to analyze
            context_summaries: Summaries of context documents
            
        Returns:
            ExhibitsResearchResult: Results of the exhibits research
        """
        return run_exhibits_research(doc, context_summaries)
    
    @handle_llm_errors("generate short version", "document_processing")
    def _generate_short_version(self, long_version: str) -> ShortVersionResult:
        """Generate a shorter version of the summary.
        
        Args:
            long_version: The long version summary to condense
            
        Returns:
            ShortVersionResult: The condensed short version
        """
        return run_short_version(long_version) 