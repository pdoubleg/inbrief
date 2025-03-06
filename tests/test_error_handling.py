"""
Tests for the error_handling module.

This module contains tests for error handling functionality, particularly
for handling LLM-related errors in a consistent way.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock

from pydantic_ai.usage import Usage
from pydantic_ai.exceptions import AgentRunError, UnexpectedModelBehavior, ModelRetry
from src.models import (
    ConversionResult,
    ContextSummaries,
    ExhibitsResearchResult,
    ShortVersionResult,
    ContextSummary,
)
from src.llm.error_handling import handle_llm_errors


class MockProcessingStrategy:
    """Mock class to simulate ProcessingStrategy methods for testing the decorator."""

    @handle_llm_errors("process supporting documents", "document_processing")
    def process_supporting_documents(self, engine: MagicMock) -> ContextSummaries:
        """Simulate processing supporting documents."""
        if getattr(self, "_raise_error", False):
            raise AgentRunError("Simulated processing error")

        return ContextSummaries(
            summaries=[
                ContextSummary(
                    exhibit_number=1,
                    file_name="test.pdf",
                    summary="Test summary",
                    usages=[
                        Usage(request_tokens=100, response_tokens=50, total_tokens=150)
                    ],
                    text="Test document text",
                    text_trimmed="Test document text",
                )
            ],
            usages=[Usage(request_tokens=100, response_tokens=50, total_tokens=150)],
        )

    @handle_llm_errors("run exhibits research", "exhibits_research")
    def run_exhibits_research(
        self, doc: ConversionResult, context_summaries: ContextSummaries
    ) -> ExhibitsResearchResult:
        """Simulate running exhibits research."""
        if getattr(self, "_raise_error", False):
            raise UnexpectedModelBehavior("Simulated research error", "Error details")

        return ExhibitsResearchResult(
            result_string="Test research result",
            usages=[Usage(request_tokens=100, response_tokens=50, total_tokens=150)],
        )

    @handle_llm_errors("generate short version", "document_processing")
    def generate_short_version(self, long_version: str) -> ShortVersionResult:
        """Simulate generating short version summary."""
        if getattr(self, "_raise_error", False):
            raise ModelRetry("Simulated generation error")

        return ShortVersionResult(
            summary="Test short version",
            usages=Usage(request_tokens=100, response_tokens=50, total_tokens=150),
        )


class TestErrorHandling:
    """Test suite for error handling functionality."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock processing strategy instance."""
        return MockProcessingStrategy()

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine instance."""
        engine = MagicMock()
        engine.supporting_docs = [
            ConversionResult(
                name="test.pdf",
                text="Test document",
                text_trimmed="Test document",
                page_text="Test document",
                pages=[],
            )
        ]
        return engine

    @pytest.fixture
    def mock_context_summaries(self):
        """Create mock context summaries."""
        return ContextSummaries(
            summaries=[
                ContextSummary(
                    exhibit_number=1,
                    file_name="test.pdf",
                    summary="Test summary",
                    usages=[
                        Usage(request_tokens=100, response_tokens=50, total_tokens=150)
                    ],
                    text="Test document text",
                    text_trimmed="Test document text",
                )
            ],
            usages=[Usage(request_tokens=100, response_tokens=50, total_tokens=150)],
        )

    def test_decorator_successful_processing(self, mock_strategy, mock_engine):
        """Test successful processing with the decorator."""
        result = mock_strategy.process_supporting_documents(mock_engine)
        assert isinstance(result, ContextSummaries)
        assert len(result.summaries) == 1
        assert result.summaries[0].file_name == "test.pdf"
        assert result.summaries[0].summary == "Test summary"

    def test_decorator_exhibits_research_success(
        self, mock_strategy, mock_engine, mock_context_summaries
    ):
        """Test successful exhibits research with the decorator."""
        doc = ConversionResult(
            name="test.pdf",
            text="Test document",
            text_trimmed="Test document",
            page_text="Test document",
            pages=[],
        )
        result = mock_strategy.run_exhibits_research(doc, mock_context_summaries)
        assert isinstance(result, ExhibitsResearchResult)
        assert result.result_string == "Test research result"
        assert len(result.usages) == 1

    def test_decorator_short_version_success(self, mock_strategy):
        """Test successful short version generation with the decorator."""
        result = mock_strategy.generate_short_version("Test long version")
        assert isinstance(result, ShortVersionResult)
        assert result.summary == "Test short version"
        assert result.usages.total_tokens == 150
