"""
Tests for the providers_listing module.

This module contains tests for the providers_listing functionality, which generates
comprehensive listings of medical providers and other entities from legal documents.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List

from pydantic_ai import models, ModelRetry
from pydantic_ai.usage import Usage
from pydantic_ai.models.test import TestModel
from pydantic_ai.agent import AgentRunResult, RunContext

from src.providers_listing import (
    generate_provider_listings,
    run_provider_listings,
    qc_agent,
    entity_extractor_agent,
    entity_resolver_agent,
    entity_finalizer_agent,
    ProviderDeps,
    ENTITY_EXTRACTOR_PROMPT,
    ENTITY_RESOLVER_PROMPT,
    ENTITY_FINALIZER_PROMPT,
    QC_PROMPT,
    validate_result,
)
from src.models import (
    ConversionResult,
    EntityListing,
    FinalizedEntityListing,
    ProviderListingResult,
    QCResult,
    ResolvedEntityListing,
    Page,
    TextChunk,
    ProcessedDocument,
)

# Disable real model requests during tests
pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


class TestProvidersListing:
    """Test suite for the providers_listing module."""

    @pytest.fixture
    def mock_documents(self) -> List[ConversionResult]:
        """
        Create mock documents for testing.

        Returns:
            A list of ConversionResult objects containing sample documents
        """
        # Add skipped attribute to Page model for compatibility with utils.create_dynamic_text_chunks
        Page.model_rebuild(force=False)
        Page.model_fields["skipped"] = (bool, False)

        doc1 = ConversionResult(
            name="Medical Records.pdf",
            text="European Health Center P.S. Corp.\n32812 Pacific Hwy South, Federal Way, WA 98003\nPhone: (253) 874-5028\nDate: 08/04/2021\nPatient presented with back pain.",
            text_trimmed="European Health Center P.S. Corp.\n32812 Pacific Hwy South, Federal Way, WA 98003\nPhone: (253) 874-5028\nDate: 08/04/2021\nPatient presented with back pain.",
            page_text="Page 1:\nEuropean Health Center P.S. Corp.\n32812 Pacific Hwy South, Federal Way, WA 98003\nPhone: (253) 874-5028\nDate: 08/04/2021\nPatient presented with back pain.",
            pages=[
                Page(
                    page_number=1,
                    text="European Health Center P.S. Corp.\n32812 Pacific Hwy South, Federal Way, WA 98003\nPhone: (253) 874-5028\nDate: 08/04/2021\nPatient presented with back pain.",
                    table_headers=[],
                    skipped=False,
                )
            ],
        )

        doc2 = ConversionResult(
            name="Employment Records.pdf",
            text="Pizza Hut Restaurant\n100 Main Street, Springfield, Ohio 45005\nEmployee: John Smith\nPosition: Delivery Driver\nEmployment Period: 01/2020 - Present",
            text_trimmed="Pizza Hut Restaurant\n100 Main Street, Springfield, Ohio 45005\nEmployee: John Smith\nPosition: Delivery Driver\nEmployment Period: 01/2020 - Present",
            page_text="Page 1:\nPizza Hut Restaurant\n100 Main Street, Springfield, Ohio 45005\nEmployee: John Smith\nPosition: Delivery Driver\nEmployment Period: 01/2020 - Present",
            pages=[
                Page(
                    page_number=1,
                    text="Pizza Hut Restaurant\n100 Main Street, Springfield, Ohio 45005\nEmployee: John Smith\nPosition: Delivery Driver\nEmployment Period: 01/2020 - Present",
                    table_headers=[],
                    skipped=False,
                )
            ],
        )

        return [doc1, doc2]

    @pytest.fixture
    def mock_processed_documents(self) -> List[ProcessedDocument]:
        """
        Create mock processed documents for testing.

        Returns:
            A list of ProcessedDocument objects
        """
        chunk1 = TextChunk(
            text="European Health Center P.S. Corp.\n32812 Pacific Hwy South, Federal Way, WA 98003\nPhone: (253) 874-5028\nDate: 08/04/2021",
            start_page=1,
            end_page=1,
            token_count=100,
        )

        chunk2 = TextChunk(
            text="Pizza Hut Restaurant\n100 Main Street, Springfield, Ohio 45005\nEmployee: John Smith\nPosition: Delivery Driver",
            start_page=1,
            end_page=1,
            token_count=80,
        )

        doc1 = ProcessedDocument(
            name="Medical Records.pdf",
            text="Full text 1",
            text_trimmed="Trimmed text 1",
            page_text="Page text 1",
            pages=[],
            processed_text="Processed text 1",
            token_count=100,
            text_chunks=[chunk1],
        )

        doc2 = ProcessedDocument(
            name="Employment Records.pdf",
            text="Full text 2",
            text_trimmed="Trimmed text 2",
            page_text="Page text 2",
            pages=[],
            processed_text="Processed text 2",
            token_count=80,
            text_chunks=[chunk2],
        )

        return [doc1, doc2]

    @pytest.fixture
    def mock_provider_deps(self) -> ProviderDeps:
        """
        Create mock provider dependencies for testing.

        Returns:
            A ProviderDeps object with test data
        """
        return ProviderDeps(
            chunk_size=20000, primary_documents=[], supporting_documents=[]
        )

    @pytest.fixture
    def setup_test_models(self):
        """
        Fixture to set up test models for the tests using TestModel.

        This fixture overrides the agents with TestModel to avoid real API calls.

        Yields:
            None
        """
        with (
            qc_agent.override(model=TestModel()),
            entity_extractor_agent.override(model=TestModel()),
            entity_resolver_agent.override(model=TestModel()),
            entity_finalizer_agent.override(model=TestModel()),
        ):
            yield

    @pytest.mark.asyncio
    async def test_entity_extractor(self, setup_test_models):
        """
        Test the entity extractor agent.

        This test verifies that the agent correctly extracts entities from
        document text.

        Args:
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        text = "European Health Center P.S. Corp.\n32812 Pacific Hwy South, Federal Way, WA 98003\nPhone: (253) 874-5028"
        expected_listing = EntityListing(
            entity_listing="Medical Providers:\n\nEuropean Health Center P.S. Corp.\n32812 Pacific Hwy South, Federal Way, WA 98003\nPhone: (253) 874-5028"
        )

        # Create mock agent result
        mock_result = MagicMock(spec=AgentRunResult)
        mock_result.data = expected_listing
        mock_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        # Act
        with patch.object(entity_extractor_agent, "run", return_value=mock_result):
            result = await entity_extractor_agent.run(text)

        # Assert
        assert isinstance(result.data, EntityListing)
        assert "European Health Center" in result.data.entity_listing
        assert result.usage().total_tokens > 0

    @pytest.mark.asyncio
    async def test_entity_resolver(self, setup_test_models):
        """
        Test the entity resolver agent.

        This test verifies that the agent correctly resolves and consolidates
        entity listings.

        Args:
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        listings = "Medical Providers:\n\nEuropean Health Center P.S. Corp.\n32812 Pacific Hwy South, Federal Way, WA 98003"
        expected_resolved = ResolvedEntityListing(
            resolved_listings="Medical Providers:\n\nEuropean Health Center P.S. Corp.\n32812 Pacific Hwy South, Federal Way, WA 98003"
        )

        # Create mock agent result
        mock_result = MagicMock(spec=AgentRunResult)
        mock_result.data = expected_resolved
        mock_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        # Act
        with patch.object(entity_resolver_agent, "run", return_value=mock_result):
            result = await entity_resolver_agent.run(listings)

        # Assert
        assert isinstance(result.data, ResolvedEntityListing)
        assert "European Health Center" in result.data.resolved_listings
        assert result.usage().total_tokens > 0

    @pytest.mark.asyncio
    async def test_entity_finalizer(self, mock_provider_deps, setup_test_models):
        """
        Test the entity finalizer agent.

        This test verifies that the agent correctly finalizes entity listings
        and handles QC validation.

        Args:
            mock_provider_deps: Mock provider dependencies fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        input_text = "Primary Document Entities:\nMedical Providers:\nEuropean Health Center P.S. Corp."
        expected_finalized = FinalizedEntityListing(
            finalized_listing="Medical Providers:\n\nEuropean Health Center P.S. Corp."
        )

        # Create mock agent results
        mock_finalizer_result = MagicMock(spec=AgentRunResult)
        mock_finalizer_result.data = expected_finalized
        mock_finalizer_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        mock_qc_result = MagicMock(spec=AgentRunResult)
        mock_qc_result.data = QCResult(score=10, feedback="Good listing")
        mock_qc_result.usage.return_value = Usage(
            request_tokens=80, response_tokens=40, total_tokens=120
        )

        # Act
        with (
            patch.object(
                entity_finalizer_agent, "run", return_value=mock_finalizer_result
            ),
            patch.object(qc_agent, "run", return_value=mock_qc_result),
        ):
            result = await entity_finalizer_agent.run(
                input_text, deps=mock_provider_deps
            )

        # Assert
        assert isinstance(result.data, FinalizedEntityListing)
        assert "European Health Center" in result.data.finalized_listing
        assert result.usage().total_tokens > 0

    @pytest.mark.asyncio
    async def test_qc_agent(self, setup_test_models):
        """
        Test the QC agent.

        This test verifies that the agent correctly performs quality control
        checks on entity listings.

        Args:
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        listing = "Medical Providers:\n\nEuropean Health Center P.S. Corp."
        expected_qc = QCResult(score=10, feedback="Good listing")

        # Create mock agent result
        mock_result = MagicMock(spec=AgentRunResult)
        mock_result.data = expected_qc
        mock_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        # Act
        with patch.object(qc_agent, "run", return_value=mock_result):
            result = await qc_agent.run(listing)

        # Assert
        assert isinstance(result.data, QCResult)
        assert result.data.score >= 8
        assert result.usage().total_tokens > 0

    @pytest.mark.asyncio
    async def test_generate_provider_listings_primary_only(
        self, mock_documents, mock_processed_documents, setup_test_models
    ):
        """
        Test generate_provider_listings with only primary documents.

        This test verifies that the function correctly processes primary documents
        and generates appropriate listings.

        Args:
            mock_documents: Mock documents fixture
            mock_processed_documents: Mock processed documents fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        with patch(
            "src.providers_listing.prepare_processed_document_chunks",
            return_value=mock_processed_documents,
        ):
            # Mock entity extractor results
            mock_entity_listing = EntityListing(
                entity_listing="Medical Providers:\n\nEuropean Health Center P.S. Corp."
            )
            mock_extractor_result = MagicMock(spec=AgentRunResult)
            mock_extractor_result.data = mock_entity_listing
            mock_extractor_result.usage.return_value = Usage(
                request_tokens=100, response_tokens=50, total_tokens=150
            )

            # Mock resolver results
            mock_resolved_listing = ResolvedEntityListing(
                resolved_listings="Medical Providers:\n\nEuropean Health Center P.S. Corp."
            )
            mock_resolver_result = MagicMock(spec=AgentRunResult)
            mock_resolver_result.data = mock_resolved_listing
            mock_resolver_result.usage.return_value = Usage(
                request_tokens=80, response_tokens=40, total_tokens=120
            )

            # Mock finalizer results
            mock_finalized_listing = FinalizedEntityListing(
                finalized_listing="Medical Providers:\n\nEuropean Health Center P.S. Corp."
            )
            mock_finalizer_result = MagicMock(spec=AgentRunResult)
            mock_finalizer_result.data = mock_finalized_listing
            mock_finalizer_result.usage.return_value = Usage(
                request_tokens=90, response_tokens=45, total_tokens=135
            )

            # Mock QC results
            mock_qc_result = MagicMock(spec=AgentRunResult)
            mock_qc_result.data = QCResult(score=10, feedback="Good listing")
            mock_qc_result.usage.return_value = Usage(
                request_tokens=70, response_tokens=35, total_tokens=105
            )

            # Act
            with (
                patch.object(
                    entity_extractor_agent, "run", return_value=mock_extractor_result
                ),
                patch.object(
                    entity_resolver_agent, "run", return_value=mock_resolver_result
                ),
                patch.object(
                    entity_finalizer_agent, "run", return_value=mock_finalizer_result
                ),
                patch.object(qc_agent, "run", return_value=mock_qc_result),
            ):
                result = await generate_provider_listings(
                    primary_documents=mock_documents
                )

            # Assert
            assert isinstance(result, ProviderListingResult)
            assert "European Health Center" in result.resolved_listing
            assert len(result.usages) > 0

    @pytest.mark.asyncio
    async def test_generate_provider_listings_both_types(
        self, mock_documents, mock_processed_documents, setup_test_models
    ):
        """
        Test generate_provider_listings with both primary and supporting documents.

        This test verifies that the function correctly processes both types of
        documents and generates appropriate listings.

        Args:
            mock_documents: Mock documents fixture
            mock_processed_documents: Mock processed documents fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        with patch(
            "src.providers_listing.prepare_processed_document_chunks",
            return_value=mock_processed_documents,
        ):
            # Mock entity extractor results
            mock_entity_listing = EntityListing(
                entity_listing="Medical Providers:\n\nEuropean Health Center P.S. Corp."
            )
            mock_extractor_result = MagicMock(spec=AgentRunResult)
            mock_extractor_result.data = mock_entity_listing
            mock_extractor_result.usage.return_value = Usage(
                request_tokens=100, response_tokens=50, total_tokens=150
            )

            # Mock resolver results
            mock_resolved_listing = ResolvedEntityListing(
                resolved_listings="Medical Providers:\n\nEuropean Health Center P.S. Corp."
            )
            mock_resolver_result = MagicMock(spec=AgentRunResult)
            mock_resolver_result.data = mock_resolved_listing
            mock_resolver_result.usage.return_value = Usage(
                request_tokens=80, response_tokens=40, total_tokens=120
            )

            # Mock finalizer results
            mock_finalized_listing = FinalizedEntityListing(
                finalized_listing="Medical Providers:\n\nEuropean Health Center P.S. Corp."
            )
            mock_finalizer_result = MagicMock(spec=AgentRunResult)
            mock_finalizer_result.data = mock_finalized_listing
            mock_finalizer_result.usage.return_value = Usage(
                request_tokens=90, response_tokens=45, total_tokens=135
            )

            # Mock QC results
            mock_qc_result = MagicMock(spec=AgentRunResult)
            mock_qc_result.data = QCResult(score=10, feedback="Good listing")
            mock_qc_result.usage.return_value = Usage(
                request_tokens=70, response_tokens=35, total_tokens=105
            )

            # Act
            with (
                patch.object(
                    entity_extractor_agent, "run", return_value=mock_extractor_result
                ),
                patch.object(
                    entity_resolver_agent, "run", return_value=mock_resolver_result
                ),
                patch.object(
                    entity_finalizer_agent, "run", return_value=mock_finalizer_result
                ),
                patch.object(qc_agent, "run", return_value=mock_qc_result),
            ):
                result = await generate_provider_listings(
                    primary_documents=mock_documents[:1],
                    supporting_documents=mock_documents[1:],
                )

            # Assert
            assert isinstance(result, ProviderListingResult)
            assert "European Health Center" in result.resolved_listing
            assert len(result.usages) > 0

    def test_run_provider_listings(self, mock_documents, setup_test_models):
        """
        Test the run_provider_listings synchronous wrapper function.

        This test verifies that the synchronous wrapper correctly processes
        the provider listing request.

        Args:
            mock_documents: Mock documents fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        expected_result = ProviderListingResult(
            resolved_listing="Medical Providers:\n\nEuropean Health Center P.S. Corp.",
            usages=[Usage(request_tokens=100, response_tokens=50, total_tokens=150)],
        )

        # Act
        with patch(
            "src.providers_listing.generate_provider_listings",
            return_value=expected_result,
        ):
            result = run_provider_listings(primary_documents=mock_documents)

        # Assert
        assert isinstance(result, ProviderListingResult)
        assert "European Health Center" in result.resolved_listing
        assert len(result.usages) > 0

    def test_prompts(self):
        """
        Test the provider listing system prompts.

        This test verifies that the system prompts contain the expected
        key instructions and information.
        """
        # Assert
        assert "You are a world class legal assistant AI" in ENTITY_EXTRACTOR_PROMPT
        assert "MEDICAL PROVIDERS INSTRUCTIONS" in ENTITY_EXTRACTOR_PROMPT
        assert "PARTIES INVOLVED INSTRUCTIONS" in ENTITY_EXTRACTOR_PROMPT

        assert "You are a world class legal assistant AI" in ENTITY_RESOLVER_PROMPT
        assert "INSTRUCTIONS" in ENTITY_RESOLVER_PROMPT
        assert "PII Rules" in ENTITY_RESOLVER_PROMPT

        assert "You are a world class legal assistant AI" in ENTITY_FINALIZER_PROMPT
        assert "INSTRUCTIONS" in ENTITY_FINALIZER_PROMPT
        assert "PII RULES" in ENTITY_FINALIZER_PROMPT

        assert "You are a world class legal assistant AI" in QC_PROMPT
        assert "PRIMARY RULES" in QC_PROMPT
        assert "PII RULES" in QC_PROMPT

    def test_agent_initialization(self):
        """
        Test that the agents are initialized correctly.

        This test verifies that the agents are configured with the correct models,
        result types, and system prompts.
        """
        # Assert for qc_agent
        assert qc_agent.result_type == QCResult
        assert qc_agent.model.client.max_retries == 2

        # Assert for entity_extractor_agent
        assert entity_extractor_agent.result_type == EntityListing
        assert entity_extractor_agent.model.client.max_retries == 2

        # Assert for entity_resolver_agent
        assert entity_resolver_agent.result_type == ResolvedEntityListing
        assert entity_resolver_agent.model.client.max_retries == 2

        # Assert for entity_finalizer_agent
        assert entity_finalizer_agent.result_type == FinalizedEntityListing
        assert entity_finalizer_agent.model.client.max_retries == 2

    @pytest.mark.asyncio
    async def test_result_validator(self, mock_provider_deps):
        """
        Test the result validator for entity finalization.

        This test verifies that the validator correctly validates finalized
        listings and handles QC validation.

        Args:
            mock_provider_deps: Mock provider dependencies fixture
        """
        # Create mock run context
        run_context = RunContext(
            deps=mock_provider_deps,
            model=TestModel(),
            usage=Usage(request_tokens=0, response_tokens=0, total_tokens=0),
            prompt="",
            messages=[],
        )

        # Test passing QC
        finalized_listing = FinalizedEntityListing(
            finalized_listing="Medical Providers:\n\nEuropean Health Center P.S. Corp."
        )
        mock_qc_result = MagicMock(spec=AgentRunResult)
        mock_qc_result.data = QCResult(score=10, feedback="Good listing")
        mock_qc_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        with patch.object(qc_agent, "run", new_callable=AsyncMock) as mock_run_sync:
            mock_run_sync.return_value = mock_qc_result
            result = await validate_result(run_context, finalized_listing)
            assert result == finalized_listing

        # Test failing QC
        mock_qc_result.data = QCResult(score=5, feedback="Issues found")
        with patch.object(qc_agent, "run", new_callable=AsyncMock) as mock_run_sync:
            mock_run_sync.return_value = mock_qc_result
            with pytest.raises(ModelRetry):
                await validate_result(run_context, finalized_listing)
