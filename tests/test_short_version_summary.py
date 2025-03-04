"""
Tests for the short_version_summary module.

This module contains tests for the short version summary functionality, which generates
concise summaries of legal documents while retaining all relevant information.
"""

import sys
import time
from pathlib import Path

# Add the project root to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock

from pydantic_ai import models
from pydantic_ai.usage import Usage
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ModelResponse
from pydantic_ai.agent import AgentRunResult
from pydantic_ai import capture_run_messages

from src.short_version_summary import (
    get_short_version_prompt,
    get_short_version_exhibits_prompt,
    short_version_agent,
    short_version_exhibits_agent,
    run_short_version,
    run_short_version_exhibits,
    reasoning_model,
)
from src.models import ShortVersionResult

# Disable real model requests during tests
pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


class TestShortVersionSummary:
    """Test suite for the short_version_summary module."""

    @pytest.fixture
    def mock_long_version(self) -> str:
        """
        Create a mock long version document for testing.

        Returns:
            A string containing a sample long version document
        """
        return """CASE CAPTION: DAVID BILYY V. CHRISTOPHER KENNEDY, ET AL
CLAIM NUMBER: 24-2-16107-4 KNT

HIGH LEVEL SUMMARY:
The plaintiff, David Bilyy, age 35, was involved in a motor vehicle accident on January 15, 2023.
He sustained injuries to his neck and back. Initial medical evaluation was performed at Valley Medical Center.
Treatment included physical therapy and chiropractic care spanning 6 months.

SUMMARY OF PLAINTIFF'S MEDICAL RECORDS:
Patient received treatment at Valley Medical Center from January 15, 2023 to July 15, 2023.
Treatment included X-rays, MRI of cervical spine, and 24 physical therapy sessions.
Chiropractic care was provided at Back to Health Clinic for 12 sessions.

SUBMITTED DOCUMENTS:
1. Medical records from Valley Medical Center (150 pages)
2. Physical therapy records (75 pages)
3. Chiropractic treatment records (25 pages)
4. Police accident report (10 pages)

PROVIDERS:
Valley Medical Center, 400 S 43rd St, Renton, WA 98055, Phone: (425) 228-3450
Back to Health Clinic, 1234 Main St, Kent, WA 98032, Phone: (253) 555-0123"""

    @pytest.fixture
    def mock_exhibits_draft(self) -> str:
        """
        Create a mock exhibits draft for testing.

        Returns:
            A string containing a sample exhibits draft
        """
        return """EXHIBITS SUMMARY:
Review of medical records shows consistent treatment for neck and back injuries.
Valley Medical Center records document initial evaluation and ongoing care.
Physical therapy records show improvement in range of motion over 24 sessions.
Chiropractic records indicate decreased pain levels after 12 sessions.

DOCUMENTS REVIEWED:
- Medical records from Valley Medical Center dated 1/15/2023 - 7/15/2023
- Physical therapy records spanning 6 months of treatment
- Chiropractic treatment records with 12 documented sessions
- Police accident report detailing the incident

PROVIDERS:
Valley Medical Center, 400 S 43rd St, Renton, WA 98055, Phone: (425) 228-3450
Back to Health Clinic, 1234 Main St, Kent, WA 98032, Phone: (253) 555-0123"""

    @pytest.fixture
    def setup_test_models(self):
        """
        Fixture to set up test models for the tests using TestModel.

        This fixture overrides the agents with TestModel to avoid real API calls.

        Yields:
            None
        """
        with (
            short_version_agent.override(model=TestModel()),
            short_version_exhibits_agent.override(model=TestModel()),
        ):
            yield

    def test_get_short_version_prompt(self, mock_long_version):
        """
        Test the short version prompt generation.

        Args:
            mock_long_version: Mock long version document fixture
        """
        prompt = get_short_version_prompt(mock_long_version)

        # Assert prompt contains key instructions and format guidelines
        assert "Your goal is to produce a summary report" in prompt
        assert "Instructions:" in prompt
        assert "PII Rules:" in prompt
        assert "FORMAT STRUCTURE GUIDELINES:" in prompt
        assert "EXAMPLE STRUCTURE TO FOLLOW:" in prompt
        assert mock_long_version in prompt
        assert time.strftime("%B %d, %Y", time.localtime()) in prompt

    def test_get_short_version_exhibits_prompt(self, mock_exhibits_draft):
        """
        Test the short version exhibits prompt generation.

        Args:
            mock_exhibits_draft: Mock exhibits draft fixture
        """
        prompt = get_short_version_exhibits_prompt(mock_exhibits_draft)

        # Assert prompt contains key instructions and format guidelines
        assert "Your goal is to produce a summary report" in prompt
        assert "Instructions:" in prompt
        assert "PII Rules:" in prompt
        assert "FORMAT STRUCTURE GUIDELINES:" in prompt
        assert mock_exhibits_draft in prompt
        assert time.strftime("%B %d, %Y", time.localtime()) in prompt

    @pytest.mark.asyncio
    async def test_short_version_agent(self, mock_long_version, setup_test_models):
        """
        Test the short version agent using TestModel.

        Args:
            mock_long_version: Mock long version document fixture
            setup_test_models: Fixture to set up the test models
        """
        # Create mock agent result
        mock_result = MagicMock(spec=AgentRunResult)
        mock_result.data = "Concise summary of the long version"
        mock_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        result = await short_version_agent.run(user_prompt=mock_long_version)
        assert isinstance(result.data, str)
        assert result.usage().total_tokens > 0

    @pytest.mark.asyncio
    async def test_short_version_exhibits_agent(
        self, mock_exhibits_draft, setup_test_models
    ):
        """
        Test the short version exhibits agent using TestModel.

        Args:
            mock_exhibits_draft: Mock exhibits draft fixture
            setup_test_models: Fixture to set up the test models
        """
        # Create mock agent result
        mock_result = MagicMock(spec=AgentRunResult)
        mock_result.data = "Concise summary of the exhibits"
        mock_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        result = await short_version_exhibits_agent.run(user_prompt=mock_exhibits_draft)
        assert isinstance(result.data, str)
        assert result.usage().total_tokens > 0

    def test_run_short_version(self, mock_long_version, setup_test_models):
        """
        Test the synchronous wrapper for short version generation using TestModel.

        Args:
            mock_long_version: Mock long version document fixture
            setup_test_models: Fixture to set up the test models
        """
        # Create mock result
        mock_result = MagicMock(spec=AgentRunResult)
        mock_result.data = "Concise summary of the long version"
        mock_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        # Test the synchronous wrapper
        with patch.object(short_version_agent, "run", return_value=mock_result):
            result = run_short_version(mock_long_version)
            assert isinstance(result, ShortVersionResult)
            assert isinstance(result.summary, str)
            assert result.usages.total_tokens > 0

    def test_run_short_version_exhibits(self, mock_exhibits_draft, setup_test_models):
        """
        Test the synchronous wrapper for short version exhibits generation using TestModel.

        Args:
            mock_exhibits_draft: Mock exhibits draft fixture
            setup_test_models: Fixture to set up the test models
        """
        # Create mock result
        mock_result = MagicMock(spec=AgentRunResult)
        mock_result.data = "Concise summary of the exhibits"
        mock_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        # Test the synchronous wrapper
        with patch.object(
            short_version_exhibits_agent, "run", return_value=mock_result
        ):
            result = run_short_version_exhibits(mock_exhibits_draft)
            assert isinstance(result, ShortVersionResult)
            assert isinstance(result.summary, str)
            assert result.usages.total_tokens > 0

    def test_agent_initialization(self):
        """
        Test that the agents are initialized correctly with proper configuration.
        """
        # Test short_version_agent configuration
        assert short_version_agent.model.model_name == "o1-mini"
        assert short_version_agent.model.system_prompt_role == "user"
        assert short_version_agent.model.client.max_retries == 2

        # Test short_version_exhibits_agent configuration
        assert short_version_exhibits_agent.model.model_name == "o1-mini"
        assert short_version_exhibits_agent.model.system_prompt_role == "user"
        assert short_version_exhibits_agent.model.client.max_retries == 2

        # Assert for reasoning_model
        assert reasoning_model.system_prompt_role == "user"

    @pytest.mark.asyncio
    async def test_message_flow(self, mock_long_version, setup_test_models):
        """
        Test the message flow when using the short_version_agent.

        This test verifies that the correct sequence of messages is sent to and from
        the agent using TestModel.

        Args:
            mock_long_version: Mock long version document fixture
            setup_test_models: Fixture to set up the test models
        """
        # Use the context manager to capture messages
        with short_version_agent.override(model=TestModel()):
            with capture_run_messages() as messages:
                await short_version_agent.run(
                    user_prompt=mock_long_version,
                )
        # Verify the message flow
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
        assert "CASE CAPTION: DAVID BILYY" in user_part.content

        # Check for response
        response_msg = next(
            (msg for msg in messages if isinstance(msg, ModelResponse)), None
        )
        assert response_msg is not None

    @pytest.mark.asyncio
    async def test_exhibits_message_flow(self, mock_exhibits_draft, setup_test_models):
        """
        Test the message flow when using the short_version_exhibits_agent.

        This test verifies that the correct sequence of messages is sent to and from
        the exhibits agent using TestModel.

        Args:
            mock_exhibits_draft: Mock exhibits draft fixture
            setup_test_models: Fixture to set up the test models
        """
        # Use the context manager to capture messages
        with short_version_exhibits_agent.override(model=TestModel()):
            with capture_run_messages() as messages:
                await short_version_exhibits_agent.run(
                    user_prompt=mock_exhibits_draft,
                )

        # Verify the message flow
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
        assert "EXHIBITS SUMMARY:" in user_part.content

        # Check for response
        response_msg = next(
            (msg for msg in messages if isinstance(msg, ModelResponse)), None
        )
        assert response_msg is not None
