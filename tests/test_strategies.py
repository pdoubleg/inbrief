"""Tests for document processing strategies.

This module contains tests for the various document processing strategies
in the summary engine.
"""

import pytest
import unittest
from unittest.mock import patch, MagicMock

from pydantic_ai import models
from pydantic import ValidationError

from src.models import (
    ConversionResult,
    SummaryResult,
    DiscoverySummaryResult,
    ShortVersionResult,
    ProcessingType
)
from src.strategies.primary_only import PrimaryOnlyStrategy
from src.strategies.supporting_only import SupportingOnlyStrategy
from src.strategies.medical_only import MedicalOnlyStrategy
from src.strategies.primary_and_supporting import PrimaryAndSupportingStrategy
from src.strategies.factory import StrategyFactory
from src.context.process import DocumentProcessor
from src.context.input import ProcessingInput


# Disable real model requests during tests
pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


class TestProcessingInputValidation(unittest.TestCase):
    """Tests for the ProcessingInput validation logic."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.primary_doc = ConversionResult(
            filename="test.pdf",
            content="test primary content",
            metadata={}
        )
        self.supporting_doc = ConversionResult(
            filename="support.pdf",
            content="test supporting content",
            metadata={}
        )
    
    def test_validates_primary_only(self):
        """Test that a ProcessingInput with only primary docs validates."""
        input_data = ProcessingInput(
            primary_docs=[self.primary_doc]
        )
        
        self.assertEqual(input_data.processing_type, ProcessingType.PRIMARY_ONLY)
    
    def test_validates_supporting_only(self):
        """Test that a ProcessingInput with only supporting docs validates."""
        input_data = ProcessingInput(
            supporting_docs=[self.supporting_doc]
        )
        
        self.assertEqual(input_data.processing_type, ProcessingType.SUPPORTING_ONLY)
    
    def test_validates_primary_and_supporting(self):
        """Test that a ProcessingInput with both document types validates."""
        input_data = ProcessingInput(
            primary_docs=[self.primary_doc],
            supporting_docs=[self.supporting_doc]
        )
        
        self.assertEqual(input_data.processing_type, ProcessingType.PRIMARY_AND_SUPPORTING)
    
    def test_validates_medical_records(self):
        """Test that a ProcessingInput with medical records validates."""
        input_data = ProcessingInput(
            supporting_docs=[self.supporting_doc],
            has_medical_records=True
        )
        
        self.assertEqual(input_data.processing_type, ProcessingType.MEDICAL_ONLY)
    
    def test_rejects_empty_input(self):
        """Test that empty ProcessingInput fails validation."""
        with self.assertRaises(ValueError):
            ProcessingInput()
    
    def test_requires_supporting_docs_for_medical(self):
        """Test that medical records require supporting docs."""
        with self.assertRaises(ValueError):
            ProcessingInput(has_medical_records=True)
    
    def test_processing_type_computation(self):
        """Test that processing_type is computed correctly based on input data."""
        # Test all combinations
        combinations = [
            (
                {"primary_docs": [self.primary_doc], "has_medical_records": True},
                ProcessingType.MEDICAL_ONLY
            ),
            (
                {"primary_docs": [self.primary_doc], "supporting_docs": [self.supporting_doc]},
                ProcessingType.PRIMARY_AND_SUPPORTING
            ),
            (
                {"primary_docs": [self.primary_doc]},
                ProcessingType.PRIMARY_ONLY
            ),
            (
                {"supporting_docs": [self.supporting_doc]},
                ProcessingType.SUPPORTING_ONLY
            ),
            (
                {"supporting_docs": [self.supporting_doc], "has_medical_records": True},
                ProcessingType.MEDICAL_ONLY
            ),
        ]
        
        for params, expected_type in combinations:
            input_data = ProcessingInput(**params)
            self.assertEqual(input_data.processing_type, expected_type)


class TestStrategyFactory(unittest.TestCase):
    """Tests for the strategy factory."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.primary_doc = ConversionResult(
            filename="test.pdf",
            content="test primary content",
            metadata={}
        )
        self.supporting_doc = ConversionResult(
            filename="support.pdf",
            content="test supporting content",
            metadata={}
        )
    
    def test_create_primary_only_strategy(self):
        """Test creating a primary only strategy."""
        input_data = ProcessingInput(
            primary_docs=[self.primary_doc]
        )
        strategy = StrategyFactory.create_strategy(input_data)
        self.assertIsInstance(strategy, PrimaryOnlyStrategy)
    
    def test_create_supporting_only_strategy(self):
        """Test creating a supporting only strategy."""
        input_data = ProcessingInput(
            supporting_docs=[self.supporting_doc]
        )
        strategy = StrategyFactory.create_strategy(input_data)
        self.assertIsInstance(strategy, SupportingOnlyStrategy)
    
    def test_create_primary_and_supporting_strategy(self):
        """Test creating a primary and supporting strategy."""
        input_data = ProcessingInput(
            primary_docs=[self.primary_doc],
            supporting_docs=[self.supporting_doc]
        )
        strategy = StrategyFactory.create_strategy(input_data)
        self.assertIsInstance(strategy, PrimaryAndSupportingStrategy)
    
    def test_create_medical_only_strategy(self):
        """Test creating a medical only strategy."""
        input_data = ProcessingInput(
            supporting_docs=[self.supporting_doc],
            has_medical_records=True
        )
        strategy = StrategyFactory.create_strategy(input_data)
        self.assertIsInstance(strategy, MedicalOnlyStrategy)
    
    def test_factory_with_input_validator(self):
        """Test that factory works with the input validation."""
        # This test verifies the integration between ProcessingInput and StrategyFactory
        try:
            input_data = ProcessingInput(primary_docs=[self.primary_doc])
            strategy = StrategyFactory.create_strategy(input_data)
            self.assertIsInstance(strategy, PrimaryOnlyStrategy)
        except ValidationError:
            self.fail("ProcessingInput validation failed unexpectedly")
    
    def test_invalid_processing_type(self):
        """Test handling of invalid processing types."""
        # Create a mock with an invalid processing_type
        input_data = MagicMock()
        input_data.processing_type = "INVALID_TYPE"
        
        with self.assertRaises(ValueError):
            StrategyFactory.create_strategy(input_data)


class TestDocumentProcessor(unittest.TestCase):
    """Tests for the document processor context class."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.primary_doc = ConversionResult(
            filename="test.pdf",
            content="test primary content",
            metadata={}
        )
        self.supporting_doc = ConversionResult(
            filename="support.pdf",
            content="test supporting content",
            metadata={}
        )
        self.mock_strategy = MagicMock()
    
    def test_auto_strategy_selection(self):
        """Test that the processor automatically selects the right strategy."""
        # Initialize without explicit strategy
        processor = DocumentProcessor(
            primary_docs=[self.primary_doc],
            job_id="test_job"
        )
        
        # Mock StrategyFactory to verify it's called
        original_create_strategy = StrategyFactory.create_strategy
        try:
            # Create a mock strategy
            mock_strategy = MagicMock()
            mock_result = SummaryResult(
                doc_id="test_job",
                long_version="Test result",
                short_version="",
                provider_listing="",
                documents_produced="",
                exhibits_research="",
                processing_time=0.0
            )
            mock_strategy.process.return_value = mock_result
            
            # Mock the factory
            StrategyFactory.create_strategy = MagicMock(return_value=mock_strategy)
            
            # Create input data
            input_data = ProcessingInput(
                primary_docs=[self.primary_doc],
                job_id="test_job"
            )
            
            # Process
            result = processor.process(input_data)
            
            # Verify the factory was called with the input data
            StrategyFactory.create_strategy.assert_called_once_with(input_data)
            
            # Verify strategy was called with input data
            mock_strategy.process.assert_called_once_with(input_data)
            
            # Verify we got the expected result
            self.assertEqual(result, mock_result)
        finally:
            # Restore original factory
            StrategyFactory.create_strategy = original_create_strategy
    
    def test_process_delegates_to_strategy(self):
        """Test that process delegates to the current strategy."""
        # Setup mock return
        mock_result = SummaryResult(
            doc_id="test_job",
            long_version="Test summary",
            short_version="Short summary",
            provider_listing="Provider list",
            documents_produced="",
            exhibits_research="",
            processing_time=1.0
        )
        self.mock_strategy.process.return_value = mock_result
        
        # Create processor with explicit strategy
        processor = DocumentProcessor(
            strategy=self.mock_strategy,
            primary_docs=[self.primary_doc],
            job_id="test_job"
        )
        
        # Create input data
        input_data = ProcessingInput(
            primary_docs=[self.primary_doc],
            job_id="test_job"
        )
        
        # Call the method under test
        result = processor.process(input_data)
        
        # Verify the result
        self.assertEqual(result, mock_result)
        
        # Verify the strategy was called with correct input
        self.mock_strategy.process.assert_called_once()
        actual_input = self.mock_strategy.process.call_args[0][0]
        self.assertEqual(actual_input.primary_docs, input_data.primary_docs)
        self.assertEqual(actual_input.job_id, input_data.job_id)
    
    def test_set_strategy(self):
        """Test setting a new strategy."""
        new_strategy = MagicMock()
        processor = DocumentProcessor(
            strategy=self.mock_strategy
        )
        processor.set_strategy(new_strategy)
        self.assertEqual(processor.strategy, new_strategy)
    


class TestPrimaryOnlyStrategy(unittest.TestCase):
    """Tests for the primary only strategy."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.strategy = PrimaryOnlyStrategy()
        self.primary_doc = ConversionResult(
            filename="test.pdf",
            content="test primary content",
            metadata={}
        )
        self.input_data = ProcessingInput(
            primary_docs=[self.primary_doc],
            job_id="test_job"
        )
    
    @patch("src.strategies.primary_only.run_discovery_summary")
    @patch("src.strategies.primary_only.run_short_version")
    def test_process(self, mock_short_version, mock_discovery_summary):
        """Test the process method of PrimaryOnlyStrategy."""
        # Setup mock returns
        mock_discovery_summary.return_value = DiscoverySummaryResult(summary="Long summary")
        mock_short_version.return_value = ShortVersionResult(summary="Short summary")
        
        # Call the method under test
        result = self.strategy.process(self.input_data)
        
        # Verify the result
        self.assertIsInstance(result, SummaryResult)
        self.assertEqual(result.long_version, "Long summary")
        self.assertEqual(result.short_version, "Short summary")
        
        # Verify the mocks were called with expected arguments
        mock_discovery_summary.assert_called_once()
        mock_short_version.assert_called_once_with("Long summary")


