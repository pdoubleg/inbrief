"""Processing strategies for different document combinations.

This module implements the Strategy pattern for the SummaryEngine, with different
strategies for processing different combinations of primary and supporting documents.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from pydantic_ai.usage import Usage

from src.short_version_summary import run_short_version, run_short_version_exhibits

from src.providers_listing import run_provider_listings

from src.documents_produced import run_documents_produced_report

from src.discovery_summary import run_discovery_summary
from src.models import (
    ContextSummaries,
    ConversionResult,
    DiscoverySummaryResult,
    ExhibitsResearchResult,
    ProviderListingResult,
    MedicalRecordsSummaryResult,
    SummaryResult,
    ShortVersionResult,
    DocumentsProducedResult,
)
from src.summary_engine.error_handling import handle_llm_errors
from src.document_summary import run_documents_summary
from src.medical_records_summary import run_medical_records_summary
from src.exhibits_research import run_exhibits_research

from ..utils import count_tokens


PRIMARY_AND_SUPPORTING_MAX_SINGLE_PRIMARY_TOKENS = 40000
PRIMARY_ONLY_MAX_SINGLE_PRIMARY_TOKENS = 30000


class ProcessingStrategy(ABC):
    """Base abstract class for document processing strategies.

    This class defines the interface for all document processing strategies.
    Each concrete strategy implementation handles a specific combination of
    document types (primary, supporting, medical records, etc.).
    """

    @abstractmethod
    def process(self, engine: Any) -> SummaryResult:
        """Process documents and return the summary result.

        Args:
            engine: The SummaryEngine instance with the documents to process

        Returns:
            SummaryResult containing both long and short versions of the summary
        """
        pass

    @handle_llm_errors("process supporting documents", "document_processing")
    def process_supporting_documents(self, engine: Any) -> ContextSummaries:
        """Process supporting documents to generate context.

        Args:
            engine: The SummaryEngine instance

        Returns:
            ContextSummaries containing summaries of supporting documents
        """
        return run_documents_summary(engine.supporting_docs)

    @handle_llm_errors("process medical records", "medical_processing")
    def process_medical_records(self, engine: Any) -> MedicalRecordsSummaryResult:
        """Process medical records.

        Args:
            engine: The SummaryEngine instance

        Returns:
            MedicalRecordsSummaryResult containing the medical records summary
        """
        return run_medical_records_summary(engine.supporting_docs)

    @handle_llm_errors("compress primary document", "document_processing")
    def compress_primary_document(self, doc: ConversionResult) -> ContextSummaries:
        """Compress a primary document.

        Args:
            doc: The primary document to compress

        Returns:
            The compressed primary document
        """
        return run_documents_summary([doc])

    @handle_llm_errors("generate high level summary", "document_processing")
    def generate_high_level_summary(
        self, discovery_document: str, supporting_documents: str = ""
    ) -> DiscoverySummaryResult:
        """Generate a high level summary of a primary document.

        Args:
            doc: The primary document to summarize
            context_summaries: Summaries of supporting documents

        Returns:
            The high level summary
        """
        return run_discovery_summary(discovery_document, supporting_documents)

    @handle_llm_errors("generate document produced string", "document_processing")
    def generate_document_produced_string(
        self, context_summaries: ContextSummaries
    ) -> DocumentsProducedResult:
        """Generate a string of documents produced.

        Args:
            context_summaries: Summaries of supporting documents

        Returns:
            The document produced string and usage information
        """
        return run_documents_produced_report(context_summaries)

    @handle_llm_errors("generate providers listing", "document_processing")
    def generate_providers_listing(
        self,
        primary_docs: Optional[List[ConversionResult]] = None,
        supporting_docs: Optional[List[ConversionResult]] = None,
    ) -> ProviderListingResult:
        """Generate a providers listing.

        Args:
            primary_docs: List of primary documents
            supporting_docs: List of supporting documents

        Returns:
            The providers listing and usage information
        """
        return run_provider_listings(primary_docs, supporting_docs)


class PrimaryAndSupportingStrategy(ProcessingStrategy):
    """Strategy for processing both primary and supporting documents."""

    @handle_llm_errors("process primary and supporting", "document_processing")
    def process(self, engine: Any) -> SummaryResult:
        """Process primary and supporting documents.

        This strategy handles the case where both primary and supporting documents
        are available. It processes supporting documents first to provide context
        for the primary document summarization.

        Args:
            engine: The SummaryEngine instance with documents to process

        Returns:
            SummaryResult containing both long and short versions of the summary
        """
        start_time = time.perf_counter()
        usages: List[Usage] = []

        # Process supporting documents to get context
        context_summaries = self.process_supporting_documents(engine)

        high_level_summaries = []
        exhibits_research_results = []

        for doc in engine.primary_docs:
            exhibits_research_result = self._run_exhibits_research(
                doc, context_summaries
            )

            # If exhibits research was generated, combine with the initial summary, otherwise use the initial summary
            if exhibits_research_result:
                # Check if the primary document + its exhibits is too long, and if so, compress the primary document before combining it with exhibits
                # The goal here is to keep the 'high level summary' input tokens manageable
                if (
                    count_tokens(doc.text + exhibits_research_result.result_string)
                    > PRIMARY_AND_SUPPORTING_MAX_SINGLE_PRIMARY_TOKENS
                ):
                    compressed_input_response = self.compress_primary_document(doc)
                    compressed_input = compressed_input_response.summaries[0].summary
                    usages.extend(compressed_input_response.summaries[0].usages)
                    summary_response = self.generate_high_level_summary(
                        compressed_input, context_summaries
                    )
                    high_level_summaries.append(summary_response.summary)
                    exhibits_research = (
                        f"### {doc.name}\n\n{exhibits_research_result.result_string}"
                    )
                    exhibits_research_results.append(exhibits_research)
                else:
                    summary_response = self.generate_high_level_summary(
                        doc.text, context_summaries
                    )
                    high_level_summaries.append(summary_response.summary)
                    exhibits_research = (
                        f"### {doc.name}\n\n{exhibits_research_result.result_string}"
                    )
                    exhibits_research_results.append(exhibits_research)
            # If no exhibits research is needed, just use the initial summary
            else:
                # Check if the primary document is too long, and if so, compress it before summarizing
                if count_tokens(doc.text) > PRIMARY_ONLY_MAX_SINGLE_PRIMARY_TOKENS:
                    compressed_input_response = self.compress_primary_document(doc)
                    compressed_input = compressed_input_response.summaries[0].summary
                    usages.extend(compressed_input_response.summaries[0].usages)
                    summary_response = self.generate_high_level_summary(
                        compressed_input
                    )
                    high_level_summaries.append(summary_response.summary)
                    usages.extend(summary_response.usages)
                else:
                    summary_response = self.generate_high_level_summary(doc.text)
                    high_level_summaries.append(summary_response.summary)
                    usages.extend(summary_response.usages)

        if engine.has_medical_records:
            medical_summary_response = self.process_medical_records(engine)
            usages.extend(medical_summary_response.usages)
            medical_summary = medical_summary_response.summary

        # Documents produced
        documents_produced_summary, documents_produced_usage = (
            self.generate_document_produced_string(context_summaries)
        )
        usages.extend(documents_produced_usage)

        providers_listing_response = self.generate_providers_listing(
            engine.primary_docs, engine.supporting_docs
        )
        usages.extend(providers_listing_response.usages)
        providers_listing = providers_listing_response.resolved_listing

        long_version = ""

        for i, doc in enumerate(engine.primary_docs):
            long_version += f"{doc.name}\n\n"
            long_version += f"{high_level_summaries[i]}\n\n"

        if self.has_medical_records:
            long_version += f"{medical_summary}\n\n"

        long_version += f"Documents Produced:\n\n{documents_produced_summary}\n\n"
        long_version += f"{providers_listing}\n\n"

        # Generate short version
        short_version_result = self._generate_short_version(long_version)
        usages.extend(short_version_result.usages)

        # Calculate processing time
        processing_time = time.perf_counter() - start_time

        return SummaryResult(
            long_version=long_version,
            short_version=short_version_result.summary,
            cost=engine._cost,
            processing_time=processing_time,
            error=None,
            usages=usages,
        )

    @handle_llm_errors("run exhibits research", "exhibits_research")
    def _run_exhibits_research(
        self, doc: ConversionResult, context_summaries: ContextSummaries
    ) -> ExhibitsResearchResult:
        """Run exhibits research.

        Args:
            doc: The primary document to research
            context_summaries: Summaries of supporting documents

        Returns:
            ExhibitsResearchResult containing the exhibits research
        """
        return run_exhibits_research(doc, context_summaries)

    @handle_llm_errors("generate short version", "document_processing")
    def _generate_short_version(self, long_version: str) -> ShortVersionResult:
        """Generate a short version of the summary.

        Args:
            long_version: The long version summary
        """
        return run_short_version(long_version)


class PrimaryOnlyStrategy(ProcessingStrategy):
    """Strategy for processing only primary documents."""

    @handle_llm_errors("process primary only", "document_processing")
    def process(self, engine: Any) -> SummaryResult:
        """Process only primary documents.

        This strategy handles the case where only primary documents are available,
        with no supporting documents to provide context.

        Args:
            engine: The SummaryEngine instance with documents to process

        Returns:
            SummaryResult containing both long and short versions of the summary
        """
        start_time = time.perf_counter()

        high_level_summaries = []
        usages: List[Usage] = []

        for doc in engine.primary_docs:
            # Check if the primary document is too long, and if so, compress it before summarizing
            if count_tokens(doc.text) > PRIMARY_ONLY_MAX_SINGLE_PRIMARY_TOKENS:
                compressed_input_response = self.compress_primary_document(doc)
                compressed_input = compressed_input_response.summaries[0].summary
                usages.extend(compressed_input_response.summaries[0].usages)
                summary_response = self.generate_high_level_summary(compressed_input)
                high_level_summaries.append(summary_response.summary)
                usages.extend(summary_response.usages)
            else:
                summary_response = self.generate_high_level_summary(doc.text)
                high_level_summaries.append(summary_response.summary)
                usages.extend(summary_response.usages)

        # Providers listing
        providers_listing_response = self.generate_providers_listing(
            primary_docs=engine.primary_docs
        )
        usages.extend(providers_listing_response.usages)
        parties_involved = providers_listing_response.resolved_listing

        long_version = ""

        for i, doc in enumerate(engine.primary_docs):
            long_version += f"{doc.name}\n\n"
            long_version += f"{high_level_summaries[i]}\n\n"
        long_version += f"{parties_involved}\n\n"

        short_version_result = self._generate_short_version(long_version)
        usages.extend(short_version_result.usages)

        # Calculate processing time
        processing_time = time.perf_counter() - start_time

        return SummaryResult(
            long_version=long_version,
            short_version=short_version_result.summary,
            cost=engine._cost,
            processing_time=processing_time,
            error=None,
            usages=usages,
        )

    @handle_llm_errors("generate short version", "document_processing")
    def _generate_short_version(self, long_version: str) -> ShortVersionResult:
        """Generate a short version of the summary.

        Args:
            long_version: The long version summary
        """
        return run_short_version(long_version)


class MedicalOnlyStrategy(ProcessingStrategy):
    """Strategy for processing only medical records."""

    @handle_llm_errors("process medical only", "medical_processing")
    def process(self, engine: Any) -> SummaryResult:
        """Process only medical records.

        This strategy handles the case where only medical records need to be summarized,
        without primary discovery documents.

        Args:
            engine: The SummaryEngine instance with documents to process

        Returns:
            SummaryResult containing the medical records summary
        """
        start_time = time.perf_counter()
        usages: List[Usage] = []

        if not engine.supporting_docs:
            raise ValueError(
                "Supporting documents are required for medical-only processing."
            )

        # Process medical records
        medical_summary_response = self.process_medical_records(engine)
        usages.extend(medical_summary_response.usages)
        medical_summary = medical_summary_response.summary

        # Calculate processing time
        processing_time = time.perf_counter() - start_time

        return SummaryResult(
            long_version=medical_summary,
            short_version=medical_summary,
            cost=engine._cost,
            processing_time=processing_time,
            error=None,
            usages=usages,
        )


class SupportingOnlyStrategy(ProcessingStrategy):
    """Strategy for processing only supporting documents."""

    @handle_llm_errors("process supporting only", "document_processing")
    def process(self, engine: Any) -> SummaryResult:
        """Process only supporting documents.

        This strategy handles the case where only supporting documents are available,
        without primary discovery documents.

        Args:
            engine: The SummaryEngine instance with documents to process

        Returns:
            SummaryResult containing the supporting documents summary
        """
        start_time = time.perf_counter()
        usages: List[Usage] = []

        # Process supporting documents
        context_summaries = self.process_supporting_documents(engine)
        high_level_summaries = [str(doc) for doc in context_summaries.summaries]
        high_level_summaries_string = "\n\n".join(high_level_summaries)
        usages.extend(context_summaries.usages)

        if engine.has_medical_records:
            medical_summary_response = self.process_medical_records(engine)
            usages.extend(medical_summary_response.usages)
            medical_summary = medical_summary_response.summary

        documents_produced_summary, documents_produced_usage = (
            self.generate_document_produced_string(context_summaries)
        )
        usages.extend(documents_produced_usage)

        parties_involved_response = self.generate_providers_listing(
            supporting_docs=engine.supporting_docs
        )
        usages.extend(parties_involved_response.usages)
        parties_involved = parties_involved_response.resolved_listing

        long_version = high_level_summaries_string
        if self.has_medical_records:
            long_version += f"{medical_summary}\n\n"
        long_version += f"Documents Produced:\n\n{documents_produced_summary}\n\n"
        long_version += f"{parties_involved}\n\n"

        # Generate short version
        short_version_summary, short_version_usage = (
            self._generate_short_version_exhibits(long_version)
        )
        usages.extend(short_version_usage)

        # Calculate processing time
        processing_time = time.perf_counter() - start_time

        return SummaryResult(
            long_version=long_version,
            short_version=short_version_summary,
            cost=engine._cost,
            processing_time=processing_time,
            error=None,
            usages=usages,
        )

    @handle_llm_errors("generate short version", "document_processing")
    def _generate_short_version_exhibits(self, draft_report: str) -> ShortVersionResult:
        """Generate a short version of the summary.

        Args:
            draft_report: The long version summary
        """
        return run_short_version_exhibits(draft_report)
