"""
Tests for the medical_records_summary module.

This module contains tests for the medical_records_summary functionality, which generates
summaries of medical records for legal cases.
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
from pydantic_ai.messages import (
    ModelResponse,
)
from pydantic_ai.agent import AgentRunResult
from pydantic_ai import capture_run_messages

from src.modules.medical_records_summary import (
    process_chunk,
    consolidate_summaries,
    generate_title_description,
    process_medical_records,
    run_medical_records_summary,
    medical_records_summary_agent,
    consolidation_agent,
    reasoning_model,
    finalization_agent,
    title_description_agent,
    MEDICAL_RECORDS_SUMMARY_PROMPT,
)
from src.models import (
    ConversionResult,
    MedicalRecordsSummaryResult,
    TitleAndDescriptionResult,
    Page,
    TextChunk,
    ProcessedDocument,
)


# Disable real model requests during tests
pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


class TestMedicalRecordsSummary:
    """Test suite for the medical_records_summary module."""

    @pytest.fixture
    def mock_medical_records(self) -> List[ConversionResult]:
        """
        Create mock medical records for testing.

        Returns:
            A list of ConversionResult objects containing sample medical records
        """
        # Add skipped attribute to Page model for compatibility with utils.create_dynamic_text_chunks
        Page.model_rebuild(force=False)
        Page.model_fields["skipped"] = (bool, False)

        record1 = ConversionResult(
            name="Northside Hospital Records.pdf",
            text="Northside Hospital Gwinnett\nPatient: John Doe\nDOB: 10/15/1980\nDate of Service: 10/28/2022\n\nPatient presented to the E.R. via EMS reporting assault with gunshot wound to left thigh. Admitted for surgical repair of femur fracture.",
            text_trimmed="Northside Hospital Gwinnett\nPatient: John Doe\nDOB: 10/15/1980\nDate of Service: 10/28/2022\n\nPatient presented to the E.R. via EMS reporting assault with gunshot wound to left thigh. Admitted for surgical repair of femur fracture.",
            page_text="Page 1:\nNorthside Hospital Gwinnett\nPatient: John Doe\nDOB: 10/15/1980\nDate of Service: 10/28/2022\n\nPatient presented to the E.R. via EMS reporting assault with gunshot wound to left thigh. Admitted for surgical repair of femur fracture.",
            pages=[
                Page(
                    page_number=1,
                    text="Northside Hospital Gwinnett\nPatient: John Doe\nDOB: 10/15/1980\nDate of Service: 10/28/2022\n\nPatient presented to the E.R. via EMS reporting assault with gunshot wound to left thigh. Admitted for surgical repair of femur fracture.",
                    table_headers=[],
                    skipped=False,
                )
            ],
        )

        record2 = ConversionResult(
            name="Resurgens Orthopaedics.pdf",
            text="Resurgens Orthopaedics\nPatient: John Doe\nDOB: 10/15/1980\nDate of Service: 11/08/2022\n\nFollow-up visit for surgical wound check. Patient had ORIF surgery on 10/30/2022 for gunshot wound to left thigh with femur fracture.",
            text_trimmed="Resurgens Orthopaedics\nPatient: John Doe\nDOB: 10/15/1980\nDate of Service: 11/08/2022\n\nFollow-up visit for surgical wound check. Patient had ORIF surgery on 10/30/2022 for gunshot wound to left thigh with femur fracture.",
            page_text="Page 1:\nResurgens Orthopaedics\nPatient: John Doe\nDOB: 10/15/1980\nDate of Service: 11/08/2022\n\nFollow-up visit for surgical wound check. Patient had ORIF surgery on 10/30/2022 for gunshot wound to left thigh with femur fracture.",
            pages=[
                Page(
                    page_number=1,
                    text="Resurgens Orthopaedics\nPatient: John Doe\nDOB: 10/15/1980\nDate of Service: 11/08/2022\n\nFollow-up visit for surgical wound check. Patient had ORIF surgery on 10/30/2022 for gunshot wound to left thigh with femur fracture.",
                    table_headers=[],
                    skipped=False,
                )
            ],
        )

        return [record1, record2]

    @pytest.fixture
    def mock_processed_documents(self) -> List[ProcessedDocument]:
        """
        Create mock processed documents for testing.

        Returns:
            A list of ProcessedDocument objects
        """
        # Create a text chunk
        chunk1 = TextChunk(
            text="Northside Hospital Gwinnett\nPatient: John Doe\nDOB: 10/15/1980\nDate of Service: 10/28/2022\n\nPatient presented to the E.R. via EMS reporting assault with gunshot wound to left thigh.",
            start_page=1,
            end_page=1,
            token_count=100,
        )

        chunk2 = TextChunk(
            text="Resurgens Orthopaedics\nPatient: John Doe\nDOB: 10/15/1980\nDate of Service: 11/08/2022\n\nFollow-up visit for surgical wound check.",
            start_page=1,
            end_page=1,
            token_count=80,
        )

        # Create processed documents
        doc1 = ProcessedDocument(
            name="Northside Hospital Records.pdf",
            text="Full text 1",
            text_trimmed="Trimmed text 1",
            page_text="Page text 1",
            pages=[],
            processed_text="Processed text 1",
            token_count=100,
            text_chunks=[chunk1],
        )

        doc2 = ProcessedDocument(
            name="Resurgens Orthopaedics.pdf",
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
        # Override all the agents with TestModel
        with (
            medical_records_summary_agent.override(model=TestModel()),
            consolidation_agent.override(model=TestModel()),
            finalization_agent.override(model=TestModel()),
            title_description_agent.override(model=TestModel()),
        ):
            yield

    @pytest.mark.asyncio
    async def test_process_chunk(self, setup_test_models):
        """
        Test the process_chunk function using TestModel.

        This test verifies that the function correctly processes a chunk of text
        and returns the expected summary and usage.

        Args:
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        chunk = "Northside Hospital Gwinnett\nPatient: John Doe\nDOB: 10/15/1980\nDate of Service: 10/28/2022\n\nPatient presented to the E.R. via EMS reporting assault with gunshot wound to left thigh."

        # Act
        summary, usage = await process_chunk(chunk)

        # Assert
        assert isinstance(summary, str)
        assert isinstance(usage, Usage)
        assert usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_consolidate_summaries(self, setup_test_models):
        """
        Test the consolidate_summaries function using TestModel.

        This test verifies that the function correctly consolidates multiple summaries
        and returns a unified summary with usage information.

        Args:
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        summaries = [
            "Northside Hospital Gwinnett (10/28/22 – 11/9/22)\nPatient presented to the E.R. via EMS reporting assault with gunshot wound to left thigh.",
            "Resurgens Orthopaedics (10/29/22 – 2/21/23)\nFollow-up visit for surgical wound check. Patient had ORIF surgery for gunshot wound.",
        ]
        chunk_size = 20000

        # Act
        consolidated_summary, usages = await consolidate_summaries(
            summaries, chunk_size
        )

        # Assert
        assert isinstance(consolidated_summary, str)
        assert isinstance(usages, list)
        assert all(isinstance(usage, Usage) for usage in usages)
        assert len(usages) > 0

    @pytest.mark.asyncio
    async def test_generate_title_description(self, setup_test_models):
        """
        Test the generate_title_description function using TestModel.

        This test verifies that the function correctly generates a title and description
        for a medical records summary.

        Args:
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        summary = "SUMMARY OF PLAINTIFF'S MEDICAL RECORDS:\n\nNorthside Hospital Gwinnett (10/28/22 – 11/9/22)\nPatient presented to the E.R. via EMS reporting assault with gunshot wound to left thigh."

        # Act
        result, usage = await generate_title_description(summary)

        # Assert
        assert isinstance(result, TitleAndDescriptionResult)
        assert isinstance(usage, Usage)
        assert usage.total_tokens > 0
        assert isinstance(result.title, str)
        assert isinstance(result.description, str)

    @pytest.mark.asyncio
    async def test_process_medical_records_standard(
        self, mock_medical_records, mock_processed_documents, setup_test_models
    ):
        """
        Test the process_medical_records function in standard mode using TestModel.

        This test verifies that the function correctly processes medical records
        and returns a comprehensive summary without labels.

        Args:
            mock_medical_records: Mock medical records fixture
            mock_processed_documents: Mock processed documents fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange - Mock the utility functions
        with (
            patch(
                "src.modules.medical_records_summary.prepare_processed_document_chunks",
                return_value=mock_processed_documents,
            ),
            patch("src.modules.medical_records_summary.count_tokens", return_value=5000),
        ):
            # Create test usage
            test_usage = Usage(request_tokens=100, response_tokens=50, total_tokens=150)

            # Create mock process_chunk and consolidate_summaries results
            mock_process_chunk_tuple = ("Test chunk summary", test_usage)
            mock_consolidate_summaries_tuple = (
                "Consolidated summary",
                [test_usage, test_usage],
            )

            # Create mock finalization_agent.run result
            mock_finalization_result = MagicMock(spec=AgentRunResult)
            mock_finalization_result.data = "Finalized summary"
            mock_finalization_result.usage.return_value = test_usage

            # Patch the functions
            with (
                patch(
                    "src.modules.medical_records_summary.process_chunk",
                    return_value=mock_process_chunk_tuple,
                ),
                patch(
                    "src.modules.medical_records_summary.consolidate_summaries",
                    return_value=mock_consolidate_summaries_tuple,
                ),
                patch(
                    "src.modules.medical_records_summary.finalization_agent.run",
                    return_value=mock_finalization_result,
                ),
            ):
                # Act
                result = await process_medical_records(mock_medical_records)
                print(result)

        # Assert
        assert isinstance(result, MedicalRecordsSummaryResult)
        assert isinstance(result.summary, str)
        assert result.summary == "Finalized summary"
        assert len(result.usages) > 0
        assert result.document_title == ""  # No labels in standard mode
        assert result.document_description == ""  # No labels in standard mode

    @pytest.mark.asyncio
    async def test_process_medical_records_with_labels(
        self, mock_medical_records, mock_processed_documents, setup_test_models
    ):
        """
        Test the process_medical_records function with labels using TestModel.

        This test verifies that the function correctly processes medical records
        and returns a comprehensive summary with title and description labels.

        Args:
            mock_medical_records: Mock medical records fixture
            mock_processed_documents: Mock processed documents fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange - Mock the utility functions
        with (
            patch(
                "src.modules.medical_records_summary.prepare_processed_document_chunks",
                return_value=mock_processed_documents,
            ),
            patch("src.modules.medical_records_summary.count_tokens", return_value=5000),
        ):
            # Create test usage
            test_usage = Usage(request_tokens=100, response_tokens=50, total_tokens=150)

            # Create mock process_chunk and consolidate_summaries results
            mock_process_chunk_tuple = ("Test chunk summary", test_usage)
            mock_consolidate_summaries_tuple = (
                "Consolidated summary",
                [test_usage, test_usage],
            )

            # Create mock finalization_agent.run result
            mock_finalization_result = MagicMock(spec=AgentRunResult)
            mock_finalization_result.data = "Finalized summary"
            mock_finalization_result.usage.return_value = test_usage

            # Create mock title_description result
            mock_title_desc_tuple = (
                TitleAndDescriptionResult(
                    title="Test Title", description="Test Description"
                ),
                test_usage,
            )

            # Patch the functions
            with (
                patch(
                    "src.modules.medical_records_summary.process_chunk",
                    return_value=mock_process_chunk_tuple,
                ),
                patch(
                    "src.modules.medical_records_summary.consolidate_summaries",
                    return_value=mock_consolidate_summaries_tuple,
                ),
                patch(
                    "src.modules.medical_records_summary.finalization_agent.run",
                    return_value=mock_finalization_result,
                ),
                patch(
                    "src.modules.medical_records_summary.generate_title_description",
                    return_value=mock_title_desc_tuple,
                ),
            ):
                # Act
                result = await process_medical_records(
                    mock_medical_records, add_labels=True
                )
                print(result)

        # Assert
        assert isinstance(result, MedicalRecordsSummaryResult)
        assert isinstance(result.summary, str)
        assert result.summary == "Finalized summary"
        assert len(result.usages) > 0
        assert result.document_title == "Test Title"
        assert result.document_description == "Test Description"

    @pytest.mark.asyncio
    async def test_process_medical_records_custom_chunk_size(
        self, mock_medical_records, mock_processed_documents, setup_test_models
    ):
        """
        Test the process_medical_records function with custom chunk size using TestModel.

        This test verifies that the function correctly processes medical records
        with a custom chunk size.

        Args:
            mock_medical_records: Mock medical records fixture
            mock_processed_documents: Mock processed documents fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange - Mock the utility functions
        with (
            patch(
                "src.modules.medical_records_summary.prepare_processed_document_chunks",
                return_value=mock_processed_documents,
            ),
            patch("src.modules.medical_records_summary.count_tokens", return_value=5000),
        ):
            # Create test usage
            test_usage = Usage(request_tokens=100, response_tokens=50, total_tokens=150)

            # Create mock process_chunk and consolidate_summaries results
            mock_process_chunk_tuple = ("Test chunk summary", test_usage)
            mock_consolidate_summaries_tuple = (
                "Consolidated summary",
                [test_usage, test_usage],
            )

            # Create mock finalization_agent.run result
            mock_finalization_result = MagicMock(spec=AgentRunResult)
            mock_finalization_result.data = "Finalized summary"
            mock_finalization_result.usage.return_value = test_usage

            # Patch the functions
            with (
                patch(
                    "src.modules.medical_records_summary.process_chunk",
                    return_value=mock_process_chunk_tuple,
                ),
                patch(
                    "src.modules.medical_records_summary.consolidate_summaries",
                    return_value=mock_consolidate_summaries_tuple,
                ),
                patch(
                    "src.modules.medical_records_summary.finalization_agent.run",
                    return_value=mock_finalization_result,
                ),
            ):
                # Act
                result = await process_medical_records(
                    mock_medical_records, chunk_size=10000
                )
                print(result)

        # Assert
        assert isinstance(result, MedicalRecordsSummaryResult)
        assert isinstance(result.summary, str)
        assert result.summary == "Finalized summary"
        assert len(result.usages) > 0

    def test_run_medical_records_summary(self, mock_medical_records, setup_test_models):
        """
        Test the run_medical_records_summary function using TestModel.

        This test verifies that the synchronous wrapper function correctly
        processes medical records.

        Args:
            mock_medical_records: Mock medical records fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange - Create a mock MedicalRecordsSummaryResult
        expected_result = MedicalRecordsSummaryResult(
            document_name="Summary of plaintiff's medical records",
            document_title="",
            document_description="",
            summary="Test summary",
            usages=[Usage(request_tokens=100, response_tokens=50, total_tokens=150)],
        )

        # Act - Mock the asynchronous function and call the synchronous wrapper
        with patch(
            "src.modules.medical_records_summary.process_medical_records",
            return_value=expected_result,
        ):
            result = run_medical_records_summary(mock_medical_records)

        # Assert
        assert isinstance(result, MedicalRecordsSummaryResult)
        assert result.summary == "Test summary"
        assert len(result.usages) > 0

    def test_run_medical_records_summary_with_options(
        self, mock_medical_records, setup_test_models
    ):
        """
        Test the run_medical_records_summary function with custom options using TestModel.

        This test verifies that the synchronous wrapper function correctly
        processes medical records with custom options.

        Args:
            mock_medical_records: Mock medical records fixture
            setup_test_models: Fixture to set up the test models
        """
        # Arrange - Create a mock MedicalRecordsSummaryResult
        expected_result = MedicalRecordsSummaryResult(
            document_name="Summary of plaintiff's medical records",
            document_title="Test Title",
            document_description="Test Description",
            summary="Test summary with options",
            usages=[Usage(request_tokens=100, response_tokens=50, total_tokens=150)],
        )

        # Act - Mock the asynchronous function and call the synchronous wrapper
        with patch(
            "src.modules.medical_records_summary.process_medical_records",
            return_value=expected_result,
        ):
            result = run_medical_records_summary(
                mock_medical_records,
                chunk_size=15000,
                add_labels=True,
                cap_multiplier=1.1,
            )

        # Assert
        assert isinstance(result, MedicalRecordsSummaryResult)
        assert result.summary == "Test summary with options"
        assert result.document_title == "Test Title"
        assert result.document_description == "Test Description"
        assert len(result.usages) > 0

    def test_medical_records_summary_prompt(self):
        """
        Test the medical records summary system prompt.

        This test verifies that the system prompt for the medical records summary
        contains the expected key instructions and information.
        """
        # Assert
        assert (
            "You are a world class legal assistant AI" in MEDICAL_RECORDS_SUMMARY_PROMPT
        )
        assert (
            "generate a **narrative summary report**" in MEDICAL_RECORDS_SUMMARY_PROMPT
        )
        assert (
            "grouped by **provider and full date range** of service"
            in MEDICAL_RECORDS_SUMMARY_PROMPT
        )
        assert "EXAMPLE OUTPUT" in MEDICAL_RECORDS_SUMMARY_PROMPT
        assert (
            "Do not use bullet points or numbered lists"
            in MEDICAL_RECORDS_SUMMARY_PROMPT
        )
        assert "PII Rules" in MEDICAL_RECORDS_SUMMARY_PROMPT

    def test_agent_initialization(self):
        """
        Test that the agents are initialized correctly.

        This test verifies that the agents are configured with the correct models,
        result types, and system prompts.
        """
        # Assert for medical_records_summary_agent
        assert medical_records_summary_agent.result_type == str  # noqa: E721
        assert medical_records_summary_agent.model.client.max_retries == 2

        # Assert for consolidation_agent
        assert consolidation_agent.result_type == str  # noqa: E721
        assert consolidation_agent.model.client.max_retries == 2

        # Assert for finalization_agent
        assert finalization_agent.result_type == str  # noqa: E721
        assert finalization_agent.model.client.max_retries == 2

        # Assert for title_description_agent
        assert title_description_agent.result_type == TitleAndDescriptionResult
        assert title_description_agent.model.client.max_retries == 2

        # Assert for reasoning_model
        assert reasoning_model.system_prompt_role == "user"

    @pytest.mark.asyncio
    async def test_message_flow(self, setup_test_models):
        """
        Test the message flow when using the medical_records_summary_agent.

        This test verifies that the correct sequence of messages is sent to and from
        the agent using TestModel.

        Args:
            setup_test_models: Fixture to set up the test models
        """
        # Arrange
        chunk = "Northside Hospital Gwinnett\nPatient: John Doe\nDate of Service: 10/28/2022\n\nPatient presented with gunshot wound."

        # Act
        # Use the context manager to capture messages
        with medical_records_summary_agent.override(model=TestModel()):
            with capture_run_messages() as messages:
                await medical_records_summary_agent.run(chunk)

        # Assert
        # Verify the message flow between the agent and model using the captured messages
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
        assert "Northside Hospital Gwinnett" in user_part.content

        # Check for response
        response_msg = next(
            (msg for msg in messages if isinstance(msg, ModelResponse)), None
        )
        assert response_msg is not None
