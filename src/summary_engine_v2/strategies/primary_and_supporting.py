"""Primary and Supporting Documents Processing Strategy.

This strategy processes both primary and supporting documents to generate
comprehensive summaries with cross-references between document types.
"""

import time
from typing import List, Optional, Dict, Any, TYPE_CHECKING

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

from src.summary_engine_v2.base import ProcessingStrategy

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from src.summary_engine_v2.context import ProcessingInput

# Constant for maximum tokens in a primary document
MAX_SINGLE_PRIMARY_TOKENS = 40000


class PrimaryAndSupportingStrategy(ProcessingStrategy):
    """Strategy for processing both primary and supporting documents.
    
    This strategy handles the workflow for processing both primary
    and supporting documents, generating summaries that incorporate information 
    from both document types.
    
    Example:
        ```python
        strategy = PrimaryAndSupportingStrategy()
        processor = DocumentProcessor(strategy=strategy, ...)
        result = processor.process()
        ```
    """
    
    def process(self, input_data: "ProcessingInput") -> SummaryResult:
        """Process primary and supporting documents.
        
        Args:
            input_data: Validated processing input
            
        Returns:
            SummaryResult: The complete summary result
        """
        
        primary_docs = input_data.primary_docs
        supporting_docs = input_data.supporting_docs
        include_exhibits_research = input_data.include_exhibits_research
        has_medical_records = input_data.has_medical_records
        
        start_time = time.time()
        usages = []
        
        # Process supporting documents to get context
        context_summaries = self.process_supporting_documents(supporting_docs)
        
        high_level_summaries = []
        exhibits_research_results = []
        
        # Process each primary document
        for doc in primary_docs:
            # Run exhibits research
            exhibits_research_result = None
            if include_exhibits_research:
                exhibits_research_result = self._run_exhibits_research(doc, context_summaries)
            
            # If exhibits research was generated, combine with the initial summary, otherwise use the initial summary
            if exhibits_research_result:
                # Check if the primary document + its exhibits is too long, and if so, compress the primary document before combining it with exhibits
                if count_tokens(doc.content + exhibits_research_result.exhibits_research) > MAX_SINGLE_PRIMARY_TOKENS:
                    compressed_doc_context = self.compress_primary_document(doc)
                    compressed_content = compressed_doc_context.summaries[0].summary if compressed_doc_context.summaries else ""
                    if hasattr(compressed_doc_context.summaries[0], "usages"):
                        usages.extend(compressed_doc_context.summaries[0].usages)
                    
                    summary_response = self.generate_high_level_summary(compressed_content, context_summaries.to_string() if context_summaries else "")
                    high_level_summaries.append(summary_response.summary)
                    if hasattr(summary_response, "usages"):
                        usages.extend(summary_response.usages)
                    
                    exhibits_research = f"### {doc.name}\n\n{exhibits_research_result.exhibits_research}"
                    exhibits_research_results.append(exhibits_research)
                else:
                    summary_response = self.generate_high_level_summary(doc.content, context_summaries.to_string() if context_summaries else "")
                    high_level_summaries.append(summary_response.summary)
                    if hasattr(summary_response, "usages"):
                        usages.extend(summary_response.usages)
                    
                    exhibits_research = f"### {doc.name}\n\n{exhibits_research_result.exhibits_research}"
                    exhibits_research_results.append(exhibits_research)
            # If no exhibits research is needed, just use the initial summary
            else:
                # Check if the primary document is too long, and if so, compress it before summarizing
                if count_tokens(doc.content) > MAX_SINGLE_PRIMARY_TOKENS:
                    compressed_doc_context = self.compress_primary_document(doc)
                    compressed_content = compressed_doc_context.summaries[0].summary if compressed_doc_context.summaries else ""
                    if hasattr(compressed_doc_context.summaries[0], "usages"):
                        usages.extend(compressed_doc_context.summaries[0].usages)
                    
                    summary_response = self.generate_high_level_summary(compressed_content)
                    high_level_summaries.append(summary_response.summary)
                    if hasattr(summary_response, "usages"):
                        usages.extend(summary_response.usages)
                else:
                    summary_response = self.generate_high_level_summary(doc.content)
                    high_level_summaries.append(summary_response.summary)
                    if hasattr(summary_response, "usages"):
                        usages.extend(summary_response.usages)
        
        # Process medical records if included
        medical_summary = ""
        if has_medical_records:
            medical_summary_response = self.generate_medical_records_summary(primary_docs)
            if hasattr(medical_summary_response, "usages"):
                usages.extend(medical_summary_response.usages)
            medical_summary = medical_summary_response.summary
        
        # Documents produced
        documents_produced_result = None
        if context_summaries:
            documents_produced_result = self.generate_document_produced_string(context_summaries)
            if hasattr(documents_produced_result, "usages"):
                usages.extend(documents_produced_result.usages)
        
        # Generate provider listings
        provider_listing_result = self.generate_providers_listing(primary_docs, supporting_docs)
        if hasattr(provider_listing_result, "usages"):
            usages.extend(provider_listing_result.usages)
        
        # Construct the long version
        long_version = ""
        
        for i, doc in enumerate(primary_docs):
            long_version += f"{doc.name}\n\n"
            if i < len(high_level_summaries):
                long_version += f"{high_level_summaries[i]}\n\n"
        
        if has_medical_records:
            long_version += f"{medical_summary}\n\n"
        
        if documents_produced_result:
            long_version += f"Documents Produced:\n\n{documents_produced_result.documents_produced}\n\n"
        
        if provider_listing_result:
            long_version += f"{provider_listing_result.provider_listing}\n\n"
        
        # Append exhibits research if available
        if exhibits_research_results:
            for exhibit_result in exhibits_research_results:
                long_version += f"{exhibit_result}\n\n"
        
        # Generate short version
        short_version_result = self._generate_short_version(long_version)
        if hasattr(short_version_result, "usages"):
            usages.extend(short_version_result.usages)
        
        # Calculate processing time
        duration = time.time() - start_time
        
        # Construct and return the final result
        return SummaryResult(
            doc_id=input_data.job_id or "",
            provider_listing=provider_listing_result.provider_listing if provider_listing_result else "",
            long_version=long_version,
            short_version=short_version_result.short_version if short_version_result else "",
            documents_produced=documents_produced_result.documents_produced if documents_produced_result else "",
            exhibits_research="\n\n".join(exhibits_research_results) if exhibits_research_results else "",
            processing_time=duration,
            usages=usages
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
    
    def _generate_discovery_summary(
        self, primary_docs: List[ConversionResult], context_summaries: Optional[ContextSummaries]
    ) -> DiscoverySummaryResult:
        """Generate a discovery summary from primary documents and context.
        
        This is kept for backward compatibility but the main logic has been moved
        to the process method for better alignment with the v1 implementation.
        
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
    
    @handle_llm_errors("generate short version", "document_processing")
    def _generate_short_version(self, long_version: str) -> ShortVersionResult:
        """Generate a shorter version of the summary.
        
        Args:
            long_version: The long version summary to condense
            
        Returns:
            ShortVersionResult: The condensed short version
        """
        return run_short_version(long_version) 