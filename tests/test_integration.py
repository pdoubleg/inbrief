"""Integration tests for the document processing workflow.

This module tests the end-to-end workflow of document processing,
focusing on the automatic strategy selection and integration between components.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.models import (
    ConversionResult,
    SummaryResult,
    DiscoverySummaryResult,
    MedicalRecordsSummaryResult,
    ShortVersionResult,
    ProviderListingResult,
    ProcessingType
)
from src.summary_engine_v2.context import ProcessingInput, DocumentProcessor


def test_primary_only_workflow_simple():
    """Simple test for the primary-only workflow to check test discovery."""
    # Create a fake document
    primary_doc = ConversionResult(
        name="test.pdf",
        text="test primary content",
        text_trimmed="test primary content",
        page_text="test primary content",
        pages=[]
    )
    
    # Create input data with only primary docs
    input_data = ProcessingInput(
        primary_docs=[primary_doc]
    )
    
    # Verify the processing type is determined correctly
    assert input_data.processing_type == ProcessingType.PRIMARY_ONLY


class TestEndToEndWorkflow(unittest.TestCase):
    """Tests for the end-to-end document processing workflow."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.primary_doc = ConversionResult(
            name="test.pdf",
            text="test primary content",
            text_trimmed="test primary content",
            page_text="test primary content",
            pages=[]
        )
        self.supporting_doc = ConversionResult(
            name="support.pdf",
            text="test supporting content",
            text_trimmed="test supporting content",
            page_text="test supporting content",
            pages=[]
        )
    
    @patch("src.summary_engine_v2.context.StrategyFactory.create_strategy")
    @patch("src.summary_engine_v2.strategies.primary_only.run_discovery_summary")
    @patch("src.summary_engine_v2.strategies.primary_only.run_short_version")
    @patch("src.summary_engine_v2.strategies.primary_only.run_provider_listings")
    def test_primary_only_automatic_workflow(
        self, mock_provider_listings, mock_short_version, mock_discovery_summary, mock_create_strategy
    ):
        """Test the primary-only workflow with automatic strategy selection."""
        # Setup mock strategy
        primary_strategy = MagicMock()
        mock_create_strategy.return_value = primary_strategy
        
        # Setup mock returns
        mock_discovery_summary.return_value = DiscoverySummaryResult(summary="Long summary")
        mock_short_version.return_value = ShortVersionResult(summary="Short summary")
        mock_provider_listings.return_value = ProviderListingResult(resolved_listing="Provider list")
        
        # Setup result
        result = SummaryResult(
            long_version="Long summary",
            short_version="Short summary",
            processing_time=1.0
        )
        primary_strategy.process.return_value = result
        
        # Create processor without a strategy (will be auto-selected)
        processor = DocumentProcessor(
            primary_docs=[self.primary_doc],
            job_id="test_job"
        )
        
        # Create input with primary docs only
        input_data = ProcessingInput(
            primary_docs=[self.primary_doc],
            job_id="test_job"
        )
        
        # Verify correct strategy type was computed
        self.assertEqual(input_data.processing_type, ProcessingType.PRIMARY_ONLY)
        
        # Process documents
        process_result = processor.process(input_data)
        
        # Verify result
        self.assertIsInstance(process_result, SummaryResult)
        self.assertEqual(process_result.long_version, "Long summary")
        self.assertEqual(process_result.short_version, "Short summary")
        
        # Verify the correct mocks were called
        mock_create_strategy.assert_called_once()
    
    @patch("src.summary_engine_v2.context.StrategyFactory.create_strategy")
    @patch("src.summary_engine_v2.strategies.medical_only.run_medical_records_summary")
    @patch("src.summary_engine_v2.strategies.medical_only.run_provider_listings")
    def test_medical_only_automatic_workflow(
        self, mock_provider_listings, mock_medical_records_summary, mock_create_strategy
    ):
        """Test the medical-only workflow with automatic strategy selection."""
        # Setup mock strategy
        medical_strategy = MagicMock()
        mock_create_strategy.return_value = medical_strategy
        
        # Setup mock returns
        mock_medical_records_summary.return_value = MedicalRecordsSummaryResult(summary="Medical summary")
        mock_provider_listings.return_value = ProviderListingResult(resolved_listing="Medical providers")
        
        # Setup result
        result = SummaryResult(
            long_version="Medical summary",
            short_version="",
            processing_time=1.0
        )
        medical_strategy.process.return_value = result
        
        # Create processor without a strategy (will be auto-selected)
        processor = DocumentProcessor(
            supporting_docs=[self.supporting_doc],
            has_medical_records=True,
            job_id="test_job"
        )
        
        # Create input with medical docs
        input_data = ProcessingInput(
            supporting_docs=[self.supporting_doc],
            has_medical_records=True,
            job_id="test_job"
        )
        
        # Verify correct strategy type was computed
        self.assertEqual(input_data.processing_type, ProcessingType.MEDICAL_ONLY)
        
        # Process documents
        process_result = processor.process(input_data)
        
        # Verify result
        self.assertIsInstance(process_result, SummaryResult)
        self.assertEqual(process_result.long_version, "Medical summary")
        
        # Verify the correct mocks were called
        mock_create_strategy.assert_called_once()
    
    @patch("src.summary_engine_v2.context.StrategyFactory.create_strategy")
    def test_integration_of_validation_and_factory(self, mock_create_strategy):
        """Test integration between ProcessingInput validation and StrategyFactory."""
        # Create mock strategies
        mock_primary_strategy = MagicMock()
        mock_medical_strategy = MagicMock()
        
        # Setup return values for factory
        mock_create_strategy.side_effect = [
            mock_primary_strategy, 
            mock_medical_strategy
        ]
        
        # Create mock results
        primary_result = SummaryResult(
            long_version="Primary summary",
            short_version="Short primary",
            processing_time=1.0
        )
        medical_result = SummaryResult(
            long_version="Medical summary",
            short_version="",
            processing_time=1.0
        )
        
        mock_primary_strategy.process.return_value = primary_result
        mock_medical_strategy.process.return_value = medical_result
        
        # Process primary docs with first processor
        primary_processor = DocumentProcessor()
        primary_input = ProcessingInput(
            primary_docs=[self.primary_doc],
            job_id="test_job"
        )
        primary_output = primary_processor.process(primary_input)
        
        # Process medical docs with second processor
        medical_processor = DocumentProcessor()
        medical_input = ProcessingInput(
            supporting_docs=[self.supporting_doc],
            has_medical_records=True,
            job_id="test_job"
        )
        medical_output = medical_processor.process(medical_input)
        
        # Verify factory was called with both inputs in correct order
        calls = mock_create_strategy.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0][0].processing_type, ProcessingType.PRIMARY_ONLY)
        self.assertEqual(calls[1][0][0].processing_type, ProcessingType.MEDICAL_ONLY)
        
        # Verify outputs match expected results
        self.assertEqual(primary_output, primary_result)
        self.assertEqual(medical_output, medical_result)


if __name__ == "__main__":
    unittest.main() 