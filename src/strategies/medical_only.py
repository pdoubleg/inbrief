"""Medical Only Document Processing Strategy.

This strategy specializes in processing medical records to generate
focused medical summaries.
"""

import time
from typing import List, Optional, TYPE_CHECKING

from src.models import (
    ConversionResult,
    SummaryResult,
    MedicalRecordsSummaryResult,
    ProviderListingResult,
)
from src.modules.medical_records_summary import run_medical_records_summary
from src.modules.providers_listing import run_provider_listings
from src.llm.error_handling import handle_llm_errors
from strategies.base import ProcessingStrategy

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from src.summary_engine.context import ProcessingInput


class MedicalOnlyStrategy(ProcessingStrategy):
    """Strategy for processing only medical records.
    
    This strategy specializes in analyzing medical records to generate
    comprehensive medical summaries without considering other document types.
    
    Example:
        ```python
        strategy = MedicalOnlyStrategy()
        processor = DocumentProcessor(
            strategy=strategy,
            primary_docs=medical_documents,
            has_medical_records=True
        )
        result = processor.process()
        ```
    """
    
    def process(self, input_data: "ProcessingInput") -> SummaryResult:
        """Process medical records to generate a medical summary.
        
        This method implements the workflow for processing medical records:
        1. Process the medical records
        2. Generate provider listings
        
        Args:
            input_data: ProcessingInput object containing all necessary data for processing:
        
        Returns:
            SummaryResult: The complete medical summary result
            
        Raises:
            ValueError: If medical records are missing or has_medical_records is False
        """
        
        supporting_docs = input_data.supporting_docs
        job_id = input_data.job_id
        
        start_time = time.time()
        
        # Process the medical records
        medical_summary_result = self.process_medical_records(supporting_docs)
        
        # Generate the provider listing
        provider_listing_result = self.generate_providers_listing(supporting_docs)
        
        # Track processing time
        duration = time.time() - start_time
        
        # Construct and return the final result
        return SummaryResult(
            doc_id=job_id or "",
            provider_listing=provider_listing_result.resolved_listing if provider_listing_result else "",
            long_version=medical_summary_result.summary,
            short_version="",  # No short version for medical only
            documents_produced="",  # No supporting documents to produce
            exhibits_research="",  # No exhibits research for medical only
            processing_time=duration,
        )
    
    @handle_llm_errors("process medical records", "medical_processing")
    def process_medical_records(self, documents: List[ConversionResult]) -> MedicalRecordsSummaryResult:
        """Process medical records to generate summaries.
        
        Args:
            documents: List of medical record documents to process
            
        Returns:
            MedicalRecordsSummaryResult: Comprehensive summary of the medical records
            
        Example:
            ```python
            medical_summary = strategy.process_medical_records(medical_docs)
            ```
        """
        return run_medical_records_summary(documents)
    
    @handle_llm_errors("generate providers listing", "document_processing")
    def generate_providers_listing(
        self,
        supporting_docs: Optional[List[ConversionResult]] = None,
    ) -> ProviderListingResult:
        """Generate a listing of all medical providers found in the documents.
        
        This method focuses on extracting medical providers from the medical records.
        
        Args:
            supporting_docs: List of medical record documents
            
        Returns:
            ProviderListingResult: Listing of all medical providers found
        """
        # For medical only, we ignore supporting_docs parameter
        return run_provider_listings(supporting_documents=supporting_docs) 