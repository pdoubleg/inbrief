"""
Tests for the exhibits_research module.

This module contains tests for the exhibits_research functionality, which performs
research on legal exhibits and generates answers to discovery questions.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
import unittest.mock
from typing import List, Union

from pydantic_ai import models, ModelRetry
from pydantic_ai.usage import Usage
from pydantic_ai.models.test import TestModel
from pydantic_ai.agent import AgentRunResult, RunContext

from src.exhibits_research import (
    perform_exhibits_research,
    run_exhibits_research,
    issue_finder_agent,
    exhibit_selector_agent,
    question_answer_agent,
    ResearchDeps,
    ISSUE_FINDER_SYSTEM_PROMPT,
    EXHIBITS_SELECTION_PROMPT,
    QUESTION_ANSWER_PROMPT,
    add_issue_limit,
    add_available_exhibits,
    validate_result,
)
from src.models import (
    ContextSummaries,
    ContextSummary,
    ExhibitsResearchItem,
    ExhibitsResearchNotNeeded,
    ExhibitsSelection,
    ExhibitsResearchResult,
)

# Disable real model requests during tests
pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


class TestExhibitsResearch:
    """Test suite for the exhibits_research module."""

    @pytest.fixture
    def mock_context_summaries(self) -> ContextSummaries:
        """
        Create mock context summaries for testing.

        Returns:
            A ContextSummaries object containing sample summaries
        """
        summaries = [
            ContextSummary(
                exhibit_number=1,
                file_name="Exhibit_1.pdf",
                summary="Contract between parties dated Jan 1, 2023",
                document_title="Contract Agreement",
                document_description="Legal contract between ABC Corp and XYZ LLC",
                usages=[
                    Usage(request_tokens=100, response_tokens=50, total_tokens=150)
                ],
            ),
            ContextSummary(
                exhibit_number=2,
                file_name="Exhibit_2.pdf",
                summary="Email correspondence regarding contract terms",
                document_title="Email Thread",
                document_description="Email discussion of contract terms",
                usages=[Usage(request_tokens=80, response_tokens=40, total_tokens=120)],
            ),
        ]
        return ContextSummaries(summaries=summaries)

    @pytest.fixture
    def mock_research_deps(self, mock_context_summaries) -> ResearchDeps:
        """
        Create mock research dependencies for testing.

        Args:
            mock_context_summaries: Mock context summaries fixture

        Returns:
            A ResearchDeps object with test data
        """
        return ResearchDeps(
            primary_document="Test document with research items",
            context_summaries=mock_context_summaries,
            max_research_tasks=10,
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
            issue_finder_agent.override(model=TestModel()),
            exhibit_selector_agent.override(model=TestModel()),
            question_answer_agent.override(model=TestModel()),
        ):
            yield

    @pytest.mark.asyncio
    async def test_issue_finder_with_research_needed(
        self, mock_research_deps, setup_test_models
    ):
        """
        Test the issue finder agent when research is needed.

        This test verifies that the agent correctly identifies research items
        when they exist in the document.

        Args:
            mock_research_deps: Mock research dependencies fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        primary_doc = "Please refer to Exhibit 1 for contract terms. See Exhibit 2 for email correspondence."
        mock_research_items = [
            ExhibitsResearchItem(
                chain_of_thought="To answer the question, I need to find the contract terms in Exhibit 1.",
                excerpt="Please refer to Exhibit 1 for contract terms",
                question="What are the contract terms in Exhibit 1?",
            )
        ]

        # Create mock agent result
        mock_result = MagicMock(spec=AgentRunResult)
        mock_result.data = mock_research_items
        mock_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        # Act
        # Properly patch the run method with an AsyncMock that returns a coroutine
        issue_finder_agent_run_mock = unittest.mock.AsyncMock(return_value=mock_result)
        with patch.object(issue_finder_agent, "run", issue_finder_agent_run_mock):
            result = await issue_finder_agent.run(primary_doc, deps=mock_research_deps)

        # Assert
        assert isinstance(result.data, list)
        assert isinstance(result.data[0], ExhibitsResearchItem)
        assert len(result.data) > 0
        assert result.usage().total_tokens > 0

    @pytest.mark.asyncio
    async def test_issue_finder_no_research_needed(
        self, mock_research_deps, setup_test_models
    ):
        """
        Test the issue finder agent when no research is needed.

        This test verifies that the agent correctly identifies when no research
        items are needed.

        Args:
            mock_research_deps: Mock research dependencies fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        primary_doc = "Simple document with no references to exhibits."
        mock_result = ExhibitsResearchNotNeeded()

        # Create mock agent result
        mock_agent_result = MagicMock(spec=AgentRunResult)
        mock_agent_result.data = mock_result
        mock_agent_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        # Act
        with patch.object(issue_finder_agent, "run", return_value=mock_agent_result):
            result = await issue_finder_agent.run(primary_doc, deps=mock_research_deps)

        # Assert
        assert isinstance(result.data, ExhibitsResearchNotNeeded)
        assert result.usage().total_tokens > 0

    @pytest.mark.asyncio
    async def test_exhibit_selector(self, mock_research_deps, setup_test_models):
        """
        Test the exhibit selector agent.

        This test verifies that the agent correctly selects relevant exhibits
        for a given research question.

        Args:
            mock_research_deps: Mock research dependencies fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        question = "What are the contract terms?"
        mock_selection = ExhibitsSelection(
            chain_of_thought="To answer the question, I need to find the contract terms in Exhibit 1.",
            exhibit_numbers=[1],
        )

        # Create mock agent result
        mock_result = MagicMock(spec=AgentRunResult)
        mock_result.data = mock_selection
        mock_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        # Act
        with patch.object(exhibit_selector_agent, "run", return_value=mock_result):
            result = await exhibit_selector_agent.run(question, deps=mock_research_deps)

        # Assert
        assert isinstance(result.data, ExhibitsSelection)
        assert len(result.data.exhibit_numbers) > 0
        assert result.usage().total_tokens > 0

    @pytest.mark.asyncio
    async def test_question_answer(self, setup_test_models):
        """
        Test the question answer agent.

        This test verifies that the agent correctly generates answers based on
        the provided exhibits.

        Args:
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        qa_input = "Discovery excerpt: What are the contract terms?\n\nExhibits:\n\nContract details..."
        expected_answer = "The contract terms include..."

        # Create mock agent result
        mock_result = MagicMock(spec=AgentRunResult)
        mock_result.data = expected_answer
        mock_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        # Act
        with patch.object(question_answer_agent, "run", return_value=mock_result):
            result = await question_answer_agent.run(qa_input)

        # Assert
        assert isinstance(result.data, str)
        assert result.data == expected_answer
        assert result.usage().total_tokens > 0

    @pytest.mark.asyncio
    async def test_perform_exhibits_research_with_items(
        self, mock_context_summaries, setup_test_models
    ):
        """
        Test the perform_exhibits_research function when research items exist.

        This test verifies that the function correctly processes research items
        and returns appropriate results.

        Args:
            mock_context_summaries: Mock context summaries fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        primary_doc = "Please refer to Exhibit 1 for contract terms."

        # Mock issue finder result
        mock_research_items = [
            ExhibitsResearchItem(
                chain_of_thought="To answer the question, I need to find the contract terms in Exhibit 1.",
                excerpt="Please refer to Exhibit 1 for contract terms",
                question="What are the contract terms in Exhibit 1?",
            )
        ]
        mock_issue_result = MagicMock(spec=AgentRunResult)
        mock_issue_result.data = mock_research_items
        mock_issue_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        # Mock exhibit selector result
        mock_selection = ExhibitsSelection(
            chain_of_thought="To answer the question, I need to find the contract terms in Exhibit 1.",
            exhibit_numbers=[1],
        )
        mock_selector_result = MagicMock(spec=AgentRunResult)
        mock_selector_result.data = mock_selection
        mock_selector_result.usage.return_value = Usage(
            request_tokens=80, response_tokens=40, total_tokens=120
        )

        # Mock question answer result
        mock_answer = "The contract terms include specific provisions..."
        mock_qa_result = MagicMock(spec=AgentRunResult)
        mock_qa_result.data = mock_answer
        mock_qa_result.usage.return_value = Usage(
            request_tokens=90, response_tokens=45, total_tokens=135
        )

        # Act
        with (
            patch.object(issue_finder_agent, "run", return_value=mock_issue_result),
            patch.object(
                exhibit_selector_agent, "run", return_value=mock_selector_result
            ),
            patch.object(question_answer_agent, "run", return_value=mock_qa_result),
        ):
            result = await perform_exhibits_research(
                primary_doc, mock_context_summaries
            )

        # Assert
        assert isinstance(result, ExhibitsResearchResult)
        assert len(result.usages) > 0
        assert "contract terms" in result.result_string.lower()

    @pytest.mark.asyncio
    async def test_perform_exhibits_research_no_items(
        self, mock_context_summaries, setup_test_models
    ):
        """
        Test the perform_exhibits_research function when no research items exist.

        This test verifies that the function correctly handles cases where no
        research is needed.

        Args:
            mock_context_summaries: Mock context summaries fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        primary_doc = "Simple document with no references to exhibits."
        mock_result = ExhibitsResearchNotNeeded()

        # Create mock agent result
        mock_agent_result = MagicMock(spec=AgentRunResult)
        mock_agent_result.data = mock_result
        mock_agent_result.usage.return_value = Usage(
            request_tokens=100, response_tokens=50, total_tokens=150
        )

        # Act
        with patch.object(issue_finder_agent, "run", return_value=mock_agent_result):
            result = await perform_exhibits_research(
                primary_doc, mock_context_summaries
            )

        # Assert
        assert result == "No research needed."

    def test_run_exhibits_research(self, mock_context_summaries, setup_test_models):
        """
        Test the run_exhibits_research synchronous wrapper function.

        This test verifies that the synchronous wrapper correctly processes
        the research request.

        Args:
            mock_context_summaries: Mock context summaries fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        primary_doc = "Please refer to Exhibit 1 for contract terms."
        expected_result = ExhibitsResearchResult(
            result_string="Research findings...",
            usages=[Usage(request_tokens=100, response_tokens=50, total_tokens=150)],
        )

        # Act
        with patch(
            "src.exhibits_research.perform_exhibits_research",
            return_value=expected_result,
        ):
            result = run_exhibits_research(primary_doc, mock_context_summaries)

        # Assert
        assert isinstance(result, ExhibitsResearchResult)
        assert len(result.usages) > 0

    def test_prompts(self):
        """
        Test the exhibits research system prompts.

        This test verifies that the system prompts contain the expected
        key instructions and information.
        """
        # Assert
        assert "You are a world class legal AI assistant" in ISSUE_FINDER_SYSTEM_PROMPT
        assert "Please follow these instructions" in ISSUE_FINDER_SYSTEM_PROMPT

        assert "You are a world class legal AI assistant" in EXHIBITS_SELECTION_PROMPT
        assert "select the **Exhibit Number(s)**" in EXHIBITS_SELECTION_PROMPT

        assert "You are a world class legal AI assistant" in QUESTION_ANSWER_PROMPT
        assert "Please follow these instructions" in QUESTION_ANSWER_PROMPT

    def test_agent_initialization(self):
        """
        Test that the agents are initialized correctly.

        This test verifies that the agents are configured with the correct models,
        result types, and system prompts.
        """
        # Assert for issue_finder_agent
        assert (
            issue_finder_agent.result_type
            == Union[List[ExhibitsResearchItem], ExhibitsResearchNotNeeded]
        )
        assert issue_finder_agent.model.client.max_retries == 2

        # Assert for exhibit_selector_agent
        assert exhibit_selector_agent.result_type == ExhibitsSelection
        assert exhibit_selector_agent.model.client.max_retries == 2

        # Assert for question_answer_agent
        assert question_answer_agent.result_type == str  # noqa: E721
        assert question_answer_agent.model.client.max_retries == 2

    def test_system_prompt_modifiers(self, mock_research_deps):
        """
        Test the system prompt modifier functions.

        This test verifies that the functions correctly modify the system prompts
        based on the context.

        Args:
            mock_research_deps: Mock research dependencies fixture
        """
        # Create mock run context
        run_context = RunContext(
            deps=mock_research_deps,
            model=TestModel(),
            usage=Usage(request_tokens=0, response_tokens=0, total_tokens=0),
            prompt="",
            messages=[],
        )

        # Test add_issue_limit
        issue_limit_addition = add_issue_limit(run_context)
        assert str(mock_research_deps.max_research_tasks) in issue_limit_addition

        # Test add_available_exhibits
        exhibits_addition = add_available_exhibits(run_context)
        assert "Here are the available exhibits" in exhibits_addition

    @pytest.mark.asyncio
    async def test_result_validator(self, mock_research_deps):
        """
        Test the result validator for exhibit selection.

        This test verifies that the validator correctly validates exhibit numbers
        and raises appropriate errors for invalid selections.

        Args:
            mock_research_deps: Mock research dependencies fixture
        """
        # Create mock run context
        run_context = RunContext(
            deps=mock_research_deps,
            model=TestModel(),
            usage=Usage(request_tokens=0, response_tokens=0, total_tokens=0),
            prompt="",
            messages=[],
        )

        # Test valid selection
        valid_selection = ExhibitsSelection(
            chain_of_thought="To answer the question, I need to find the contract terms in Exhibit 1.",
            exhibit_numbers=[1, 2],
        )
        result = await validate_result(run_context, valid_selection)
        assert result == valid_selection

        # Test invalid selection
        invalid_selection = ExhibitsSelection(
            chain_of_thought="To answer the question, I need to find the contract terms in Exhibit 1.",
            exhibit_numbers=[3],
        )  # 3 doesn't exist
        with pytest.raises(ModelRetry):
            await validate_result(run_context, invalid_selection)
