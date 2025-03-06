"""
Tests for the document_summary module.

This module contains tests for the document_summary functionality, which generates
summaries of documents for legal cases.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from typing import List

from pydantic_ai import models
from pydantic_ai.usage import Usage
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ModelResponse
from pydantic_ai.agent import AgentRunResult
from pydantic_ai import capture_run_messages

from src.modules.document_summary import (
    process_discovery_document,
    process_single_document,
    process_documents_async,
    run_documents_summary,
    document_summary_agent,
    intermediate_summary_agent,
    consolidated_summary_agent,
    title_description_agent,
    DOCUMENT_SUMMARY_PROMPT,
    INTERMEDIATE_SUMMARY_PROMPT,
    CONSOLIDATED_SUMMARY_PROMPT,
)
from src.models import (
    ConversionResult,
    ContextSummary,
    ContextSummaries,
    TitleAndDescriptionResult,
    Page,
    TextChunk,
    ProcessedDocument,
)

# Disable real model requests during tests
pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


class TestDocumentSummary:
    """Test suite for the document_summary module."""

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
            name="Contract Agreement.pdf",
            text="AGREEMENT\nBetween ABC Corp and XYZ LLC\nDate: 01/15/2023\n\nThis agreement outlines the terms of service between the parties.",
            text_trimmed="AGREEMENT\nBetween ABC Corp and XYZ LLC\nDate: 01/15/2023\n\nThis agreement outlines the terms of service between the parties.",
            page_text="Page 1:\nAGREEMENT\nBetween ABC Corp and XYZ LLC\nDate: 01/15/2023\n\nThis agreement outlines the terms of service between the parties.",
            pages=[
                Page(
                    page_number=1,
                    text="AGREEMENT\nBetween ABC Corp and XYZ LLC\nDate: 01/15/2023\n\nThis agreement outlines the terms of service between the parties.",
                    table_headers=[],
                    skipped=False,
                )
            ],
        )

        doc2 = ConversionResult(
            name="Meeting Minutes.pdf",
            text="MEETING MINUTES\nDate: 02/01/2023\nAttendees: John Smith, Jane Doe\n\nDiscussion of quarterly objectives and project timeline.",
            text_trimmed="MEETING MINUTES\nDate: 02/01/2023\nAttendees: John Smith, Jane Doe\n\nDiscussion of quarterly objectives and project timeline.",
            page_text="Page 1:\nMEETING MINUTES\nDate: 02/01/2023\nAttendees: John Smith, Jane Doe\n\nDiscussion of quarterly objectives and project timeline.",
            pages=[
                Page(
                    page_number=1,
                    text="MEETING MINUTES\nDate: 02/01/2023\nAttendees: John Smith, Jane Doe\n\nDiscussion of quarterly objectives and project timeline.",
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
            text="AGREEMENT\nBetween ABC Corp and XYZ LLC\nDate: 01/15/2023\n\nThis agreement outlines the terms of service between the parties.",
            start_page=1,
            end_page=1,
            token_count=100,
        )

        chunk2 = TextChunk(
            text="MEETING MINUTES\nDate: 02/01/2023\nAttendees: John Smith, Jane Doe\n\nDiscussion of quarterly objectives and project timeline.",
            start_page=1,
            end_page=1,
            token_count=80,
        )

        doc1 = ProcessedDocument(
            name="Contract Agreement.pdf",
            text="Full text 1",
            text_trimmed="Trimmed text 1",
            page_text="Page text 1",
            pages=[],
            processed_text="Processed text 1",
            token_count=100,
            text_chunks=[chunk1],
        )

        doc2 = ProcessedDocument(
            name="Meeting Minutes.pdf",
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
    def setup_test_models(self):
        """
        Fixture to set up test models for the tests using TestModel.

        This fixture overrides the agents with TestModel to avoid real API calls.

        Yields:
            None
        """
        with (
            document_summary_agent.override(model=TestModel()),
            intermediate_summary_agent.override(model=TestModel()),
            consolidated_summary_agent.override(model=TestModel()),
            title_description_agent.override(model=TestModel()),
        ):
            yield

    @pytest.mark.asyncio
    async def test_process_discovery_document(
        self, mock_documents, mock_processed_documents, setup_test_models
    ):
        """
        Test the process_discovery_document function using TestModel.

        This test verifies that the function correctly processes a document
        and returns the expected summary and metadata.

        Args:
            mock_documents: Mock documents fixture
            mock_processed_documents: Mock processed documents fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange - Mock the utility functions
        with (
            patch(
                "src.modules.document_summary.prepare_processed_document_chunks",
                return_value=mock_processed_documents,
            ),
            patch("src.modules.document_summary.count_tokens", return_value=5000),
        ):
            # Create test usage
            test_usage = Usage(request_tokens=100, response_tokens=50, total_tokens=150)

            # Create mock agent results
            mock_summary_result = MagicMock(spec=AgentRunResult)
            mock_summary_result.data = "Test summary"
            mock_summary_result.usage.return_value = test_usage

            # Act
            with (
                document_summary_agent.override(model=TestModel()),
                intermediate_summary_agent.override(model=TestModel()),
                consolidated_summary_agent.override(model=TestModel()),
                title_description_agent.override(model=TestModel()),
            ):
                result = await process_discovery_document(
                    mock_documents[0], add_labels=True
                )

        # Assert
        assert isinstance(result, dict)
        assert result["document_name"] == "Contract Agreement.pdf"
        assert len(result["document_title"]) > 0
        assert len(result["document_description"]) > 0
        assert len(result["summary"]) > 0
        assert len(result["usages"]) > 0

    @pytest.mark.asyncio
    async def test_process_single_document(self, mock_documents, setup_test_models):
        """
        Test the process_single_document function using TestModel.

        This test verifies that the function correctly processes a single document
        and returns a ContextSummary object.

        Args:
            mock_documents: Mock documents fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        mock_result = {
            "document_name": "Contract Agreement.pdf",
            "document_title": "Test Title",
            "document_description": "Test Description",
            "summary": "Test summary",
            "usages": [Usage(request_tokens=100, response_tokens=50, total_tokens=150)],
        }

        # Create a proper async function to replace process_discovery_document
        async def mock_process_discovery(*args, **kwargs):
            return mock_result

        # Act
        with patch(
            "src.modules.document_summary.process_discovery_document", mock_process_discovery
        ):
            result = await process_single_document(mock_documents[0], exhibit_number=1)

        # Assert
        assert isinstance(result, ContextSummary)
        assert result.exhibit_number == 1
        assert result.file_name == "Contract Agreement.pdf"
        assert result.summary == "Test summary"
        assert result.document_title == "Test Title"
        assert result.document_description == "Test Description"
        assert len(result.usages) > 0

    @pytest.mark.asyncio
    async def test_process_documents_async(self, mock_documents, setup_test_models):
        """
        Test the process_documents_async function using TestModel.

        This test verifies that the function correctly processes multiple documents
        concurrently and returns a ContextSummaries object.

        Args:
            mock_documents: Mock documents fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        mock_summary = ContextSummary(
            exhibit_number=1,
            file_name="Test.pdf",
            summary="Test summary",
            document_title="Test Title",
            document_description="Test Description",
            usages=[Usage(request_tokens=100, response_tokens=50, total_tokens=150)],
        )
        
        # Create a proper async function to replace process_single_document
        async def mock_process_single(*args, **kwargs):
            return mock_summary

        # Act
        with patch(
            "src.modules.document_summary.process_single_document", mock_process_single
        ):
            result = await process_documents_async(mock_documents)

        # Assert
        assert isinstance(result, ContextSummaries)
        assert len(result.summaries) == len(mock_documents)
        for summary in result.summaries:
            assert isinstance(summary, ContextSummary)

    def test_run_documents_summary(self, mock_documents, setup_test_models):
        """
        Test the run_documents_summary function using TestModel.

        This test verifies that the synchronous wrapper function correctly
        processes multiple documents.

        Args:
            mock_documents: Mock documents fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        expected_result = ContextSummaries(
            summaries=[
                ContextSummary(
                    exhibit_number=1,
                    file_name="Test.pdf",
                    summary="Test summary",
                    document_title="Test Title",
                    document_description="Test Description",
                    usages=[
                        Usage(request_tokens=100, response_tokens=50, total_tokens=150)
                    ],
                )
            ]
        )

        # Act
        with patch(
            "src.modules.document_summary.process_documents_async", return_value=expected_result
        ):
            result = run_documents_summary(mock_documents)

        # Assert
        assert isinstance(result, ContextSummaries)
        assert len(result.summaries) > 0
        assert isinstance(result.summaries[0], ContextSummary)

    def test_prompts(self):
        """
        Test the document summary system prompts.

        This test verifies that the system prompts contain the expected
        key instructions and information.
        """
        # Assert
        assert "You are a world class legal assistant AI" in DOCUMENT_SUMMARY_PROMPT
        assert "Please follow these instructions" in DOCUMENT_SUMMARY_PROMPT
        assert "PII Rules" in DOCUMENT_SUMMARY_PROMPT

        assert "You are a world class legal assistant AI" in INTERMEDIATE_SUMMARY_PROMPT
        assert "Please follow these instructions" in INTERMEDIATE_SUMMARY_PROMPT

        assert "You are a world class legal assistant AI" in CONSOLIDATED_SUMMARY_PROMPT
        assert "Please follow these instructions" in CONSOLIDATED_SUMMARY_PROMPT
        assert "PII Rules" in CONSOLIDATED_SUMMARY_PROMPT

    def test_agent_initialization(self):
        """
        Test that the agents are initialized correctly.

        This test verifies that the agents are configured with the correct models,
        result types, and system prompts.
        """
        # Assert for document_summary_agent
        assert document_summary_agent.result_type == str  # noqa: E721
        assert document_summary_agent.model.client.max_retries == 2

        # Assert for intermediate_summary_agent
        assert intermediate_summary_agent.result_type == str  # noqa: E721
        assert intermediate_summary_agent.model.client.max_retries == 2

        # Assert for consolidated_summary_agent
        assert consolidated_summary_agent.result_type == str  # noqa: E721
        assert consolidated_summary_agent.model.client.max_retries == 2

        # Assert for title_description_agent
        assert title_description_agent.result_type == TitleAndDescriptionResult
        assert title_description_agent.model.client.max_retries == 2

    @pytest.mark.asyncio
    async def test_message_flow(self, setup_test_models):
        """
        Test the message flow when using the document_summary_agent.

        This test verifies that the correct sequence of messages is sent to and from
        the agent using TestModel.

        Args:
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        document = "AGREEMENT\nBetween ABC Corp and XYZ LLC\nDate: 01/15/2023\n\nThis agreement outlines the terms."

        # Act
        with document_summary_agent.override(model=TestModel()):
            with capture_run_messages() as messages:
                await document_summary_agent.run(document)

        # Assert
        assert len(messages) >= 2  # At least request and response
        assert messages[0].parts[0].content  # System prompt content exists

        # Check for user message
        user_part = next(
            (
                part
                for msg in messages
                for part in msg.parts
                if part.part_kind == "user-prompt"
            ),
            None,
        )
        assert user_part is not None
        assert "AGREEMENT" in user_part.content

        # Check for response
        response_msg = next(
            (msg for msg in messages if isinstance(msg, ModelResponse)), None
        )
        assert response_msg is not None
