"""Primary Only Document Processing Strategy.

This strategy processes only primary documents without considering
supporting documents to generate summaries.
"""

import time
from typing import List, Optional, TYPE_CHECKING

from src.models import (
    ConversionResult,
    SummaryResult,
    ShortVersionResult,
    DiscoverySummaryResult,
    ProviderListingResult,
)
from src.discovery_summary import run_discovery_summary
from src.short_version_summary import run_short_version
from src.providers_listing import run_provider_listings
from src.utils import count_tokens
from src.summary_engine.error_handling import handle_llm_errors

from src.summary_engine_v2.base import ProcessingStrategy

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from src.summary_engine_v2.context import ProcessingInput

# Constant for maximum tokens in a primary document
MAX_SINGLE_PRIMARY_TOKENS = 30000


class PrimaryOnlyStrategy(ProcessingStrategy):
    """Strategy for processing only primary documents.
    
    This strategy handles the workflow for processing primary documents
    without considering supporting documents. It's useful when supporting
    documents aren't available or aren't relevant to the analysis.
    
    Example:
        ```python
        strategy = PrimaryOnlyStrategy()
        processor = DocumentProcessor(strategy=strategy, primary_docs=primary_docs)
        result = processor.process()
        ```
    """
    
    def process(self, input_data: "ProcessingInput") -> SummaryResult:
        """Process only primary documents to generate a summary.
        
        This method implements the workflow for processing only primary documents:
        1. Process each primary document
        2. Generate a high-level summary
        3. Generate a short version of the summary
        4. Generate provider listings
        
        Args:
            input_data: ProcessingInput object containing all necessary data for processing:
                - primary_docs: List of primary documents
                - job_id: Job identifier
        
        Returns:
            SummaryResult: The complete summary result
            
        Raises:
            ValueError: If primary documents are missing
        """
        
        primary_docs = input_data.primary_docs
        job_id = input_data.job_id
        
        start_time = time.time()
        
        # Generate the discovery summary from primary documents
        discovery_summary_result = self._generate_discovery_summary(primary_docs)
        
        # Generate the short version
        short_version_result = self._generate_short_version(discovery_summary_result.summary)
        
        # Generate the provider listing
        provider_listing_result = self.generate_providers_listing(primary_docs)
        
        # Track processing time
        duration = time.time() - start_time
        
        # Construct and return the final result
        return SummaryResult(
            doc_id=job_id or "",
            provider_listing=provider_listing_result.resolved_listing if provider_listing_result else "",
            long_version=discovery_summary_result.summary,
            short_version=short_version_result.summary if short_version_result else "",
            documents_produced="",  # No supporting documents to produce
            exhibits_research="",  # No exhibits research without supporting documents
            processing_time=duration,
        )
    
    @handle_llm_errors("generate high level summary", "document_processing")
    def generate_high_level_summary(
        self, discovery_document: str, supporting_documents: str = ""
    ) -> DiscoverySummaryResult:
        """Generate a high-level summary of the discovery document.
        
        Args:
            discovery_document: The main document to summarize
            supporting_documents: Optional supporting context (not used in this strategy)
            
        Returns:
            DiscoverySummaryResult: The high-level summary
        """
        # For primary only, we don't use supporting documents
        return run_discovery_summary(discovery_document, "")
    
    @handle_llm_errors("generate providers listing", "document_processing")
    def generate_providers_listing(
        self,
        primary_docs: Optional[List[ConversionResult]] = None,
    ) -> ProviderListingResult:
        """Generate a listing of all providers found in the primary documents.
    
        Args:
            primary_docs: List of primary documents
            
        Returns:
            ProviderListingResult: Listing of all providers found
        """
        # For primary only, we ignore supporting_docs parameter
        return run_provider_listings(primary_docs, None)
    
    @handle_llm_errors("generate discovery summary", "document_processing")
    def _generate_discovery_summary(
        self, primary_docs: List[ConversionResult]
    ) -> DiscoverySummaryResult:
        """Generate a discovery summary from primary documents only.
        
        Args:
            primary_docs: List of primary documents
            
        Returns:
            DiscoverySummaryResult: The discovery summary
        """
        # Check if the document is too large
        doc_tokens = count_tokens(primary_docs[0].text)
        if doc_tokens > MAX_SINGLE_PRIMARY_TOKENS:
            print(f"Warning: Primary document exceeds token limit ({doc_tokens} > {MAX_SINGLE_PRIMARY_TOKENS})")
            # Processing continues with truncation handled by the LLM
        
        # Extract the primary document content
        discovery_document = primary_docs[0].text
        
        # Generate the high-level summary (without supporting documents)
        return self.generate_high_level_summary(discovery_document)
    
    @handle_llm_errors("generate short version", "document_processing")
    def _generate_short_version(self, long_version: str) -> ShortVersionResult:
        """Generate a shorter version of the summary.
        
        Args:
            long_version: The long version summary to condense
            
        Returns:
            ShortVersionResult: The condensed short version
        """
        return run_short_version(long_version) 