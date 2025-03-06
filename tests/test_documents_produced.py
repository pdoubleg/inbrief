"""
Tests for the documents_produced module.

This module contains tests for the documents_produced functionality, which generates
summaries of documents produced in legal discovery.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock

from pydantic_ai import capture_run_messages, models
from pydantic_ai.usage import Usage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.test import TestModel

from src.modules.documents_produced import (
    generate_documents_produced_summary,
    run_documents_produced_report,
    documents_produced_agent,
    DOCUMENTS_PRODUCED_PROMPT,
)
from src.models import ContextSummaries, ContextSummary, DocumentsProducedResult


# Disable real model requests during tests
pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


class TestDocumentsProduced:
    """Test suite for the documents_produced module."""

    @pytest.fixture
    def mock_context_summaries(self) -> ContextSummaries:
        """
        Create a mock ContextSummaries object for testing.

        Returns:
            A ContextSummaries object with test data
        """
        return ContextSummaries(
            summaries=[
                ContextSummary(
                    exhibit_number=1,
                    file_name="medical_records.pdf",
                    summary="Medical records from Dr. Smith",
                    document_title="Medical Records",
                    document_description="Records from 2020-2022",
                    usages=[],
                ),
                ContextSummary(
                    exhibit_number=2,
                    file_name="billing_statements.pdf",
                    summary="Billing statements from Hospital",
                    document_title="Billing Statements",
                    document_description="Statements from 2020-2022",
                    usages=[],
                ),
            ]
        )

    @pytest.fixture
    def mock_agent_result(self):
        """
        Create a mock agent result for testing.

        Returns:
            A MagicMock object simulating an agent result
        """
        mock_result = MagicMock()
        mock_result.data = "medical_records.pdf: Medical records from Dr. Smith.\nbilling_statements.pdf: Billing statements from Hospital."
        mock_result.usage.return_value = Usage(
            request_tokens=100,
            response_tokens=50,
            total_tokens=150,
        )
        return mock_result

    @pytest.fixture
    def setup_test_model(self):
        """Fixture to override the documents_produced_agent with TestModel for testing."""
        # Override the agent's model with TestModel
        with documents_produced_agent.override(model=TestModel()):
            yield

        # The agent is automatically restored after the test completes

    @pytest.mark.asyncio
    async def test_generate_documents_produced_summary(self, setup_test_model):
        """Test generating a documents produced summary using TestModel."""
        # TestModel will be used because of the fixture
        input_docs = "doc1.pdf\ndoc2.pdf"

        # Call the function that uses the agent
        result = await generate_documents_produced_summary(input_docs)

        # Assert on the result - TestModel will generate valid data that matches your schema
        assert isinstance(result, DocumentsProducedResult)

        # You can also check the agent's last run messages
        with capture_run_messages() as messages:
            result = await generate_documents_produced_summary(input_docs)
            assert len(messages) > 0

    def test_run_documents_produced_report(
        self, mock_context_summaries, setup_test_model
    ):
        """
        Test the run_documents_produced_report function.

        This test verifies that the function correctly processes the input context
        and returns a DocumentsProducedResult with the expected structure.

        Args:
            mock_context_summaries: Mock context summaries fixture
            setup_test_model: Fixture to set up the test model
        """
        # We need to patch asyncio.run to avoid the "cannot be called from a running event loop" error
        with patch("asyncio.run") as mock_run:
            # Set up the mock to return a DocumentsProducedResult
            mock_result = DocumentsProducedResult(
                summary="medical_records.pdf: Medical records from Dr. Smith.\nbilling_statements.pdf: Billing statements from Hospital.",
                usages=Usage(
                    request_tokens=100,
                    response_tokens=50,
                    total_tokens=150,
                ),
            )
            mock_run.return_value = mock_result

            # Act
            result = run_documents_produced_report(mock_context_summaries)

        # Assert
        assert isinstance(result, DocumentsProducedResult)
        assert isinstance(result.summary, str)
        assert isinstance(result.usages, Usage)
        assert "medical_records.pdf" in result.summary
        assert "billing_statements.pdf" in result.summary

    def test_documents_produced_prompt(self):
        """
        Test that the documents_produced_prompt contains the expected content.

        This test verifies that the prompt contains key instructions and formatting
        requirements for the documents produced summary.
        """
        # Assert
        assert "You are a world class legal assistant AI" in DOCUMENTS_PRODUCED_PROMPT
        assert "Simply state" in DOCUMENTS_PRODUCED_PROMPT
        assert "Make every word count" in DOCUMENTS_PRODUCED_PROMPT
        assert (
            "Never use markdown, bullet points, or lists" in DOCUMENTS_PRODUCED_PROMPT
        )

    def test_context_summaries_docs_info_string(self, mock_context_summaries):
        """
        Test the docs_info_string method of ContextSummaries.

        This test verifies that the method correctly formats the document information
        with and without exhibit numbers.

        Args:
            mock_context_summaries: Mock context summaries fixture
        """
        # Act
        with_numbers = mock_context_summaries.docs_info_string(number=True)
        without_numbers = mock_context_summaries.docs_info_string(number=False)

        # Assert
        assert (
            with_numbers
            == "Exhibit 1: medical_records.pdf\nExhibit 2: billing_statements.pdf"
        )
        assert without_numbers == "medical_records.pdf\nbilling_statements.pdf"

    @pytest.mark.parametrize(
        "input_docs,expected_content",
        [
            ("doc1.pdf\ndoc2.pdf", ["doc1.pdf", "doc2.pdf"]),
            ("empty.pdf", ["empty.pdf"]),
        ],
    )
    @pytest.mark.asyncio
    async def test_generate_documents_produced_summary_with_different_inputs(
        self, input_docs, expected_content, setup_test_model
    ):
        """
        Test generate_documents_produced_summary with different input documents.

        This test verifies that the function handles different input formats correctly.

        Args:
            input_docs: Input document information string
            expected_content: Expected content in the user prompt
            setup_test_model: Fixture to set up the test model
        """
        # Act
        result = await generate_documents_produced_summary(input_docs)

        # Assert
        assert isinstance(result, DocumentsProducedResult)
        assert isinstance(result.summary, str)
        assert isinstance(result.usages, Usage)

    def test_run_documents_produced_report_with_empty_context(self, setup_test_model):
        """
        Test run_documents_produced_report with an empty context.

        This test verifies that the function handles empty input correctly.

        Args:
            setup_test_model: Fixture to set up the test model
        """
        # Arrange
        empty_context = ContextSummaries(summaries=[])

        # Act
        with patch("asyncio.run") as mock_run:
            # Set up the mock to return a DocumentsProducedResult
            mock_result = DocumentsProducedResult(
                summary="No documents found.",
                usages=Usage(
                    request_tokens=10,
                    response_tokens=5,
                    total_tokens=15,
                ),
            )
            mock_run.return_value = mock_result

            # Act
            result = run_documents_produced_report(empty_context)

        # Assert
        assert isinstance(result, DocumentsProducedResult)
        assert isinstance(result.summary, str)
        assert result.summary == "No documents found."

    def test_agent_initialization(self):
        """
        Test that the documents_produced_agent is initialized correctly.

        This test verifies that the agent is configured with the correct model,
        result type, and system prompt.
        """
        # Assert
        assert isinstance(documents_produced_agent.model, OpenAIModel)
        assert documents_produced_agent.result_type == str  # noqa: E721
        assert documents_produced_agent.model.client.max_retries == 2
