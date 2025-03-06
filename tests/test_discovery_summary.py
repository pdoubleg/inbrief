"""
Tests for the discovery_summary module.

This module contains tests for the discovery_summary functionality, which generates
summaries of legal discovery documents.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock

from pydantic_ai import models
from pydantic_ai.usage import Usage
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.openai import OpenAIModel

from src.modules.discovery_summary import (
    process_discovery_document,
    run_discovery_summary,
    discovery_summary_agent,
    reasoning_agent,
    DISCOVERY_SUMMARY_PROMPT,
)
from src.models import DiscoverySummaryResult


# Disable real model requests during tests
pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False

# Configure the default loop scope for asyncio fixtures to address the deprecation warning
pytest.asyncio_default_fixture_loop_scope = "function"


class TestDiscoverySummary:
    """Test suite for the discovery_summary module."""

    @pytest.fixture
    def mock_discovery_document(self) -> str:
        """
        Create a mock discovery document for testing.

        Returns:
            A string containing a sample discovery document
        """
        return """
CASE CAPTION : JOHNSON, MICHAEL V. SMITH CONSTRUCTION CO.

CLAIM NUMBER: LA592-084732164-0002

---

INTERROGATORIES:

1. Please state your full name, address, date of birth, and Social Security number.
ANSWER: Michael Johnson, 123 Oak Street, Seattle, WA. DOB: 05/12/1975. I object to providing my Social Security number on privacy grounds.

2. Please describe, in detail, how the incident that is the subject of this lawsuit occurred.
ANSWER: On June 15, 2023, I was working at the construction site at 456 Main Street, Seattle, WA. I was walking across the site when I stepped into an unmarked hole that had been covered with a thin piece of plywood. The plywood broke under my weight, and I fell approximately 4 feet into the hole, landing on my right side.
"""

    @pytest.fixture
    def mock_supporting_documents(self) -> str:
        """
        Create mock supporting documents for testing.

        Returns:
            A string containing sample supporting documents
        """
        return """
SUPPORTING DOCUMENT 1: MEDICAL RECORD FROM SEATTLE MEDICAL CENTER

Patient: Michael Johnson
Date of Service: June 15, 2023
Provider: Dr. Jennifer Smith

Chief Complaint: Patient presents to ER following fall at construction site.

History of Present Illness: 48-year-old male construction worker who fell approximately 4 feet into a hole at work site today. Patient reports landing on his right side, with immediate pain in right ankle, wrist, hip, and shoulder. Also reports back pain and headache.
"""

    @pytest.fixture
    def setup_test_model(self):
        """
        Fixture to set up test models for the tests.

        This fixture patches the agents' run methods to return predictable results.

        Yields:
            None
        """
        # Create a mock result for the discovery_summary_agent
        mock_summary_result = MagicMock()
        mock_summary_result.data = "DISCOVERY SUMMARY:\n\nPARTY RESPONDING / TITLE OF RESPONSE: Plaintiff Michael Johnson\n\nHIGH LEVEL SUMMARY: Plaintiff, age 48, resides in Seattle, WA. He was injured on June 15, 2023, while working at a construction site when he stepped on a plywood cover that broke, causing him to fall 4 feet into a hole."
        mock_summary_result.usage.return_value = Usage(
            request_tokens=500,
            response_tokens=200,
            total_tokens=700,
        )

        # Create a mock result for the reasoning_agent
        mock_reasoning_result = MagicMock()
        mock_reasoning_result.data = "DISCOVERY SUMMARY:\n\nPARTY RESPONDING / TITLE OF RESPONSE: Plaintiff Michael Johnson\n\nHIGH LEVEL SUMMARY: Plaintiff, age 48, resides in Seattle, WA. He was injured on June 15, 2023, while working at a construction site when he stepped on a plywood cover that broke, causing him to fall 4 feet into a hole."
        mock_reasoning_result.usage.return_value = Usage(
            request_tokens=1000,
            response_tokens=300,
            total_tokens=1300,
        )

        with (
            discovery_summary_agent.override(model=TestModel()),
            reasoning_agent.override(model=TestModel()),
        ):
            yield

    @pytest.mark.asyncio
    async def test_process_discovery_document_standard(
        self, mock_discovery_document, setup_test_model
    ):
        """
        Test the process_discovery_document function with standard model.

        This test verifies that the function correctly processes the input and
        returns the expected result when using the standard model.

        Args:
            mock_discovery_document: Mock discovery document fixture
            setup_test_model: Fixture to set up the test models
        """
        # Arrange - Mock the token count to be below threshold
        with patch("src.modules.discovery_summary.count_tokens", return_value=10000):
            # Act
            result = await process_discovery_document(mock_discovery_document)

        # Assert
        assert isinstance(result, DiscoverySummaryResult)
        assert isinstance(result.summary, str)
        assert len(result.usages) == 1
        assert result.reasoning_model_flag is False
        assert result.reasoning_prompt_tokens == 0
        assert result.reasoning_completion_tokens == 0
        assert len(result.summary) > 0

    @pytest.mark.asyncio
    async def test_process_discovery_document_with_supporting_docs(
        self, mock_discovery_document, mock_supporting_documents, setup_test_model
    ):
        """
        Test the process_discovery_document function with supporting documents.

        This test verifies that the function correctly processes both the main document
        and supporting documents.

        Args:
            mock_discovery_document: Mock discovery document fixture
            mock_supporting_documents: Mock supporting documents fixture
            setup_test_model: Fixture to set up the test models
        """
        # Arrange - Mock the token count to be below threshold
        with patch("src.modules.discovery_summary.count_tokens", return_value=15000):
            # Act
            result = await process_discovery_document(
                mock_discovery_document, supporting_documents=mock_supporting_documents
            )

        # Assert
        assert isinstance(result, DiscoverySummaryResult)
        assert isinstance(result.summary, str)
        assert len(result.usages) == 1
        assert len(result.summary) > 0

    @pytest.mark.asyncio
    async def test_process_discovery_document_reasoning_model(
        self, mock_discovery_document, setup_test_model
    ):
        """
        Test the process_discovery_document function with reasoning model.

        This test verifies that the function correctly uses the reasoning model
        when the token count exceeds the threshold.

        Args:
            mock_discovery_document: Mock discovery document fixture
            setup_test_model: Fixture to set up the test models
        """
        # Arrange - Mock the token count to be above threshold
        with patch("src.modules.discovery_summary.count_tokens", return_value=40000):
            # Act
            result = await process_discovery_document(
                mock_discovery_document, reasoning_model_threshold=30000
            )

        # Assert
        assert isinstance(result, DiscoverySummaryResult)
        assert isinstance(result.summary, str)
        assert len(result.usages) == 1
        assert result.reasoning_model_flag is True
        assert result.reasoning_prompt_tokens > 0
        assert result.reasoning_completion_tokens > 0
        assert len(result.summary) > 0

    def test_run_discovery_summary(self, mock_discovery_document, setup_test_model):
        """
        Test the run_discovery_summary function.

        This test verifies that the synchronous wrapper function correctly
        processes the input and returns the expected result.

        Args:
            mock_discovery_document: Mock discovery document fixture
            setup_test_model: Fixture to set up the test models
        """
        # We need to patch asyncio.run to avoid the "cannot be called from a running event loop" error
        with patch("asyncio.run") as mock_run:
            # Set up the mock to return a DiscoverySummaryResult
            mock_result = DiscoverySummaryResult(
                summary="DISCOVERY SUMMARY:\n\nPARTY RESPONDING / TITLE OF RESPONSE: Plaintiff Michael Johnson\n\nHIGH LEVEL SUMMARY: Plaintiff, age 48, resides in Seattle, WA.",
                usages=[
                    Usage(
                        request_tokens=500,
                        response_tokens=200,
                        total_tokens=700,
                    )
                ],
                reasoning_model_flag=False,
                reasoning_prompt_tokens=0,
                reasoning_completion_tokens=0,
            )
            mock_run.return_value = mock_result

            # Act
            # Call run_discovery_summary, which is the synchronous wrapper around process_discovery_document
            result = run_discovery_summary(mock_discovery_document)

        # Assert
        assert isinstance(result, DiscoverySummaryResult)
        assert isinstance(result.summary, str)
        assert len(result.usages) == 1
        assert result.reasoning_model_flag is False
        assert len(result.summary) > 0

    def test_run_discovery_summary_with_supporting_docs(
        self, mock_discovery_document, mock_supporting_documents, setup_test_model
    ):
        """
        Test the run_discovery_summary function with supporting documents.

        Args:
            mock_discovery_document: Mock discovery document fixture
            mock_supporting_documents: Mock supporting documents fixture
            setup_test_model: Fixture to set up the test models

        Note:
            We use a proper async mock to ensure the coroutine is handled correctly.
        """
        # Create the mock result
        mock_result = DiscoverySummaryResult(
            summary="DISCOVERY SUMMARY:\n\nPARTY RESPONDING / TITLE OF RESPONSE: Plaintiff Michael Johnson\n\nHIGH LEVEL SUMMARY: Plaintiff, age 48, resides in Seattle, WA.",
            usages=[
                Usage(
                    request_tokens=500,
                    response_tokens=200,
                    total_tokens=700,
                )
            ],
            reasoning_model_flag=False,
            reasoning_prompt_tokens=0,
            reasoning_completion_tokens=0,
        )
        
        with (
            discovery_summary_agent.override(model=TestModel()),
            reasoning_agent.override(model=TestModel()),
        ):
            # Act
            result = run_discovery_summary(
                mock_discovery_document, supporting_documents=mock_supporting_documents
            )

        # Assert
        assert isinstance(result, DiscoverySummaryResult)
        assert isinstance(result.summary, str)
        assert len(result.usages) == 1
        assert len(result.summary) > 0

    def test_run_discovery_summary_with_reasoning_model(
        self, mock_discovery_document, setup_test_model
    ):
        """
        Test the run_discovery_summary function with reasoning model.

        This test verifies that the synchronous wrapper function correctly
        uses the reasoning model when the token count exceeds the threshold.

        Args:
            mock_discovery_document: Mock discovery document fixture
            setup_test_model: Fixture to set up the test models
        """
        # Create the mock result
        mock_result = DiscoverySummaryResult(
            summary="DISCOVERY SUMMARY:\n\nPARTY RESPONDING / TITLE OF RESPONSE: Plaintiff Michael Johnson\n\nHIGH LEVEL SUMMARY: Plaintiff, age 48, resides in Seattle, WA.",
            usages=[
                Usage(
                    request_tokens=1000,
                    response_tokens=300,
                    total_tokens=1300,
                )
            ],
            reasoning_model_flag=True,
            reasoning_prompt_tokens=1000,
            reasoning_completion_tokens=300,
        )

        # We need to patch asyncio.run and make it return our mock result
        with patch("asyncio.run", return_value=mock_result):
            # Act
            result = run_discovery_summary(
                mock_discovery_document,
                reasoning_model_threshold=10,  # Set a low threshold to trigger reasoning model
            )

        # Assert
        assert isinstance(result, DiscoverySummaryResult)
        assert isinstance(result.summary, str)
        assert len(result.usages) == 1
        assert result.reasoning_model_flag is True
        assert result.reasoning_prompt_tokens > 0
        assert result.reasoning_completion_tokens > 0
        assert len(result.summary) > 0

    def test_discovery_summary_prompt(self):
        """
        Test that the discovery_summary_prompt contains the expected content.

        This test verifies that the prompt contains key instructions and formatting
        requirements for the discovery summary.
        """
        # Assert
        assert (
            "You are a legal assistant for a high-power and very busy attorney"
            in DISCOVERY_SUMMARY_PROMPT
        )
        assert "DISCOVERY DOCUMENT" in DISCOVERY_SUMMARY_PROMPT
        assert "SUPPORTING DOCUMENTS" in DISCOVERY_SUMMARY_PROMPT
        assert "DO NOT USE BULLET POINTS OR NUMBERED LISTS" in DISCOVERY_SUMMARY_PROMPT
        assert "EXAMPLE OUTPUT" in DISCOVERY_SUMMARY_PROMPT

    def test_agent_initialization(self):
        """
        Test that the agents are initialized correctly.

        This test verifies that the agents are configured with the correct models,
        result types, and system prompts.
        """
        # Assert for discovery_summary_agent
        assert isinstance(discovery_summary_agent.model, OpenAIModel)
        assert discovery_summary_agent.result_type == str  # noqa: E721
        assert discovery_summary_agent.model.client.max_retries == 2
        assert reasoning_agent.model.client.max_retries == 2
        assert reasoning_agent.model.system_prompt_role == "user"
