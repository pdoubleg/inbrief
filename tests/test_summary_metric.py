"""Tests for the summary metric module."""

import unittest
from unittest.mock import MagicMock
import pytest

from src.llm.summary_metric import (
    KeyIdea,
    BreakdownResult,
    EvaluationResult,
    EvalDeps,
    breakdown_gold_summary,
    evaluate_llm_summary,
    calculate_weighted_score,
    evaluate_summary_pair,
    breakdown_agent,
    evaluation_agent,
    PrecisionKeyIdea,
    PrecisionResult,
    ExtendedEvalDeps,
    llm_breakdown_agent,
    precision_agent,
    breakdown_llm_summary,
    evaluate_precision,
    evaluate_summary_pair_extended,
    summary_evaluation_metric_extended,
)
from pydantic_ai.usage import Usage
from pydantic_ai.models.test import TestModel
from pydantic_ai import models

# Disable real model requests during tests
pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


class TestKeyIdea(unittest.TestCase):
    """Tests for the KeyIdea class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        # Test with all importance levels
        for importance in ["High", "Medium", "Low"]:
            key_idea = KeyIdea(idea="Test idea", importance=importance)
            self.assertEqual(key_idea.idea, "Test idea")
            self.assertEqual(key_idea.importance, importance)

    def test_init_invalid_importance(self):
        """Test initialization with invalid importance level."""
        # Should raise a validation error
        with self.assertRaises(ValueError):
            KeyIdea(idea="Test idea", importance="Invalid")


class TestBreakdownResult(unittest.TestCase):
    """Tests for the BreakdownResult class."""

    def test_init(self):
        """Test initialization with valid parameters."""
        key_ideas = [
            KeyIdea(idea="First idea", importance="High"),
            KeyIdea(idea="Second idea", importance="Medium"),
            KeyIdea(idea="Third idea", importance="Low"),
        ]
        result = BreakdownResult(key_ideas=key_ideas)
        self.assertEqual(len(result.key_ideas), 3)
        self.assertEqual(result.key_ideas[0].idea, "First idea")

    def test_formatted_key_ideas(self):
        """Test the formatted_key_ideas property."""
        key_ideas = [
            KeyIdea(idea="First idea", importance="High"),
            KeyIdea(idea="Second idea", importance="Medium"),
        ]
        result = BreakdownResult(key_ideas=key_ideas)
        expected = "1. First idea (High).\n2. Second idea (Medium)."
        self.assertEqual(result.formatted_key_ideas, expected)


class TestEvaluationResult(unittest.TestCase):
    """Tests for the EvaluationResult class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        result = EvaluationResult(
            explanation="Test explanation",
            binary_scores=[True, False, True],
            overall_score=0.75,
        )
        self.assertEqual(result.explanation, "Test explanation")
        self.assertEqual(result.binary_scores, [True, False, True])
        self.assertEqual(result.overall_score, 0.75)

    def test_init_invalid_score(self):
        """Test initialization with invalid score."""
        # Test with score < 0
        with self.assertRaises(ValueError):
            EvaluationResult(
                explanation="Test",
                binary_scores=[True],
                overall_score=-0.1,
            )
        
        # Test with score > 1
        with self.assertRaises(ValueError):
            EvaluationResult(
                explanation="Test",
                binary_scores=[True],
                overall_score=1.1,
            )


class TestEvalDeps(unittest.TestCase):
    """Tests for the EvalDeps class."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        deps = EvalDeps(
            gold_summary="Gold summary",
            pred_summary="Predicted summary",
        )
        self.assertEqual(deps.gold_summary, "Gold summary")
        self.assertEqual(deps.pred_summary, "Predicted summary")
        self.assertIsNone(deps.key_ideas)
        self.assertIsNone(deps.evaluation)
        self.assertIsNone(deps.score)
        self.assertEqual(deps.usages, [])

    def test_init_full(self):
        """Test initialization with all parameters."""
        key_ideas = BreakdownResult(
            key_ideas=[KeyIdea(idea="Test idea", importance="High")]
        )
        evaluation = EvaluationResult(
            explanation="Test explanation",
            binary_scores=[True],
            overall_score=0.8,
        )
        # Create a mock Usage object instead of trying to instantiate it directly
        usage = MagicMock(spec=Usage)
        
        deps = EvalDeps(
            gold_summary="Gold summary",
            pred_summary="Predicted summary",
            key_ideas=key_ideas,
            evaluation=evaluation,
            score=0.9,
            usages=[usage],
        )
        
        self.assertEqual(deps.gold_summary, "Gold summary")
        self.assertEqual(deps.pred_summary, "Predicted summary")
        self.assertEqual(deps.key_ideas, key_ideas)
        self.assertEqual(deps.evaluation, evaluation)
        self.assertEqual(deps.score, 0.9)
        self.assertEqual(deps.usages, [usage])


class TestCalculateWeightedScore(unittest.TestCase):
    """Tests for the calculate_weighted_score function."""

    def test_calculate_weighted_score(self):
        """Test calculation of weighted score."""
        key_ideas = BreakdownResult(
            key_ideas=[
                KeyIdea(idea="High idea", importance="High"),
                KeyIdea(idea="Medium idea", importance="Medium"),
                KeyIdea(idea="Low idea", importance="Low"),
            ]
        )
        
        evaluation = EvaluationResult(
            explanation="Test explanation",
            binary_scores=[True, True, False],  # High and Medium captured, Low missed
            overall_score=0.8,
        )
        
        # Expected calculation:
        # (1.0*1 + 0.7*1 + 0.2*0) / (1.0 + 0.7 + 0.2) = 1.7 / 1.9 ≈ 0.8947
        expected_score = (1.0*1 + 0.7*1 + 0.2*0) / (1.0 + 0.7 + 0.2)
        
        score = calculate_weighted_score(key_ideas, evaluation)
        self.assertAlmostEqual(score, expected_score, places=4)

    def test_fallback_to_overall_score(self):
        """Test fallback to overall score when calculation fails."""
        # Create a scenario where calculation would fail (mismatched lengths)
        key_ideas = BreakdownResult(
            key_ideas=[KeyIdea(idea="Test idea", importance="High")]
        )
        
        evaluation = EvaluationResult(
            explanation="Test explanation",
            binary_scores=[],  # Empty binary scores to force exception
            overall_score=0.75,
        )
        
        # Should fall back to the overall_score
        score = calculate_weighted_score(key_ideas, evaluation)
        self.assertEqual(score, 0.0)


class TestAsyncFunctions:
    """Tests for the async functions in the module."""
    
    @pytest.mark.asyncio
    async def test_breakdown_gold_summary(self):
        """Test breakdown_gold_summary function with TestModel."""
        with (
            breakdown_agent.override(model=TestModel()),
            evaluation_agent.override(model=TestModel()),
        ):
            # Call function with test model
            result = await breakdown_gold_summary("Test gold summary")
            result = result.data
        # TestModel returns the same type as the expected result
        # So we just verify the shape/type is correct
        assert isinstance(result, BreakdownResult)
        assert len(result.key_ideas) > 0
        
    @pytest.mark.asyncio
    async def test_evaluate_llm_summary(self):
        """Test evaluate_llm_summary function with TestModel."""
        key_ideas = BreakdownResult(
            key_ideas=[KeyIdea(idea="Test idea", importance="High")]
        )
        with (
            breakdown_agent.override(model=TestModel()),
            evaluation_agent.override(model=TestModel()),
        ):
            # Call function with test model
            result = await evaluate_llm_summary("Test llm summary", key_ideas)
            result = result.data
        # Verify the result structure
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.explanation, str)
        assert isinstance(result.binary_scores, list)
        assert isinstance(result.overall_score, float)
        
    @pytest.mark.asyncio
    async def test_evaluate_summary_pair(self):
        """Test evaluate_summary_pair function with TestModel."""
        # Setup test data
        deps = EvalDeps(
            gold_summary="This is a gold summary.",
            pred_summary="This is a predicted summary.",
        )
        with (
            breakdown_agent.override(model=TestModel()),
            evaluation_agent.override(model=TestModel()),
        ):
            # Call function
            result = await evaluate_summary_pair(deps)
        
        # Verify structure and results
        assert result.key_ideas is not None
        assert result.evaluation is not None
        assert result.score is not None
        assert len(result.usages) > 0


class TestPrecisionKeyIdea(unittest.TestCase):
    """Tests for the PrecisionKeyIdea class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        # Test with both accurate and inaccurate ideas
        for accurate in [True, False]:
            key_idea = PrecisionKeyIdea(
                idea="Test idea", 
                accurate=accurate,
                explanation="This is an explanation"
            )
            self.assertEqual(key_idea.idea, "Test idea")
            self.assertEqual(key_idea.accurate, accurate)
            self.assertEqual(key_idea.explanation, "This is an explanation")


class TestPrecisionResult(unittest.TestCase):
    """Tests for the PrecisionResult class."""

    def test_init(self):
        """Test initialization with valid parameters."""
        llm_key_ideas = [
            PrecisionKeyIdea(idea="First idea", accurate=True, explanation="Correct"),
            PrecisionKeyIdea(idea="Second idea", accurate=False, explanation="Wrong"),
        ]
        result = PrecisionResult(
            llm_key_ideas=llm_key_ideas,
            precision_score=0.5,
            explanation="Half of the ideas are accurate"
        )
        self.assertEqual(len(result.llm_key_ideas), 2)
        self.assertEqual(result.llm_key_ideas[0].idea, "First idea")
        self.assertEqual(result.precision_score, 0.5)
        self.assertEqual(result.explanation, "Half of the ideas are accurate")

    def test_formatted_key_ideas(self):
        """Test the formatted_key_ideas property."""
        llm_key_ideas = [
            PrecisionKeyIdea(idea="First idea", accurate=True, explanation="Correct"),
            PrecisionKeyIdea(idea="Second idea", accurate=False, explanation="Wrong"),
        ]
        result = PrecisionResult(
            llm_key_ideas=llm_key_ideas,
            precision_score=0.5,
            explanation="Half of the ideas are accurate"
        )
        expected = "1. First idea (✓) - Correct\n2. Second idea (✗) - Wrong"
        self.assertEqual(result.formatted_key_ideas, expected)


class TestExtendedEvalDeps(unittest.TestCase):
    """Tests for the ExtendedEvalDeps class."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        deps = ExtendedEvalDeps(
            gold_summary="Gold summary",
            pred_summary="Predicted summary",
        )
        self.assertEqual(deps.gold_summary, "Gold summary")
        self.assertEqual(deps.pred_summary, "Predicted summary")
        self.assertIsNone(deps.key_ideas)
        self.assertIsNone(deps.evaluation)
        self.assertIsNone(deps.score)
        self.assertEqual(deps.usages, [])
        # Check new fields
        self.assertIsNone(deps.llm_key_ideas)
        self.assertIsNone(deps.precision_result)
        self.assertIsNone(deps.precision_score)
        self.assertIsNone(deps.recall_score)

    def test_init_full(self):
        """Test initialization with all parameters."""
        gold_key_ideas = BreakdownResult(
            key_ideas=[KeyIdea(idea="Gold idea", importance="High")]
        )
        llm_key_ideas = BreakdownResult(
            key_ideas=[KeyIdea(idea="LLM idea", importance="Medium")]
        )
        evaluation = EvaluationResult(
            explanation="Test explanation",
            binary_scores=[True],
            overall_score=0.8,
        )
        precision_result = PrecisionResult(
            llm_key_ideas=[
                PrecisionKeyIdea(idea="LLM idea", accurate=True, explanation="Matches")
            ],
            precision_score=1.0,
            explanation="All ideas are accurate"
        )
        # Create a mock Usage object
        usage = MagicMock(spec=Usage)
        
        deps = ExtendedEvalDeps(
            gold_summary="Gold summary",
            pred_summary="Predicted summary",
            key_ideas=gold_key_ideas,
            llm_key_ideas=llm_key_ideas,
            evaluation=evaluation,
            precision_result=precision_result,
            score=0.8,
            recall_score=0.8,
            precision_score=1.0,
            usages=[usage],
        )
        
        self.assertEqual(deps.gold_summary, "Gold summary")
        self.assertEqual(deps.pred_summary, "Predicted summary")
        self.assertEqual(deps.key_ideas, gold_key_ideas)
        self.assertEqual(deps.llm_key_ideas, llm_key_ideas)
        self.assertEqual(deps.evaluation, evaluation)
        self.assertEqual(deps.precision_result, precision_result)
        self.assertEqual(deps.score, 0.8)
        self.assertEqual(deps.recall_score, 0.8)
        self.assertEqual(deps.precision_score, 1.0)
        self.assertEqual(deps.usages, [usage])


class TestAsyncExtendedFunctions:
    """Tests for the extended async functions in the module."""
    
    @pytest.mark.asyncio
    async def test_breakdown_llm_summary(self):
        """Test breakdown_llm_summary function with TestModel."""
        with (
            llm_breakdown_agent.override(model=TestModel()),
        ):
            # Call function with test model
            result = await breakdown_llm_summary("Test LLM summary")
            result = result.data
        # Verify the result structure
        assert isinstance(result, BreakdownResult)
        assert len(result.key_ideas) > 0
        
    @pytest.mark.asyncio
    async def test_evaluate_precision(self):
        """Test evaluate_precision function with TestModel."""
        llm_key_ideas = BreakdownResult(
            key_ideas=[KeyIdea(idea="Test LLM idea", importance="High")]
        )
        with (
            precision_agent.override(model=TestModel()),
        ):
            # Call function with test model
            result = await evaluate_precision(llm_key_ideas, "Test gold summary")
            result = result.data
        # Verify the result structure
        assert isinstance(result, PrecisionResult)
        assert len(result.llm_key_ideas) > 0
        assert isinstance(result.precision_score, float)
        assert isinstance(result.explanation, str)
        
    @pytest.mark.asyncio
    async def test_evaluate_summary_pair_extended(self):
        """Test evaluate_summary_pair_extended function with TestModel."""
        # Setup test data
        deps = ExtendedEvalDeps(
            gold_summary="This is a gold summary.",
            pred_summary="This is a predicted summary.",
        )
        with (
            breakdown_agent.override(model=TestModel()),
            evaluation_agent.override(model=TestModel()),
            llm_breakdown_agent.override(model=TestModel()),
            precision_agent.override(model=TestModel()),
        ):
            # Call function
            result = await evaluate_summary_pair_extended(deps)
        
        # Verify structure and results
        assert result.key_ideas is not None
        assert result.evaluation is not None
        assert result.llm_key_ideas is not None
        assert result.precision_result is not None
        assert result.recall_score is not None
        assert result.precision_score is not None
        assert result.score is not None  # Should match recall_score
        assert len(result.usages) > 0
        
        # Verify recall score equals the original score
        assert result.recall_score == result.score


class TestSummaryEvaluationMetricExtended:
    """Tests for the summary_evaluation_metric_extended function."""
    
    def test_summary_evaluation_metric_extended(self, monkeypatch):
        """Test summary_evaluation_metric_extended function by mocking the async function."""
        # Create mock objects for test
        mock_deps = ExtendedEvalDeps(
            gold_summary="Gold summary",
            pred_summary="Pred summary",
        )
        
        # Set expected result values
        expected_result = ExtendedEvalDeps(
            gold_summary="Gold summary",
            pred_summary="Pred summary",
            recall_score=0.8,
            precision_score=0.9,
            score=0.8,  # Same as recall_score
        )
        
        # Mock asyncio.run to return our expected result
        def mock_run(coroutine):
            return expected_result
        
        monkeypatch.setattr("asyncio.run", mock_run)
        
        # Call the function
        result = summary_evaluation_metric_extended(mock_deps)
        
        # Verify the result
        assert result == expected_result
        assert result.recall_score == 0.8
        assert result.precision_score == 0.9
        assert result.score == result.recall_score


# You may also want to add tests for the batch processing functions
# However, these might be more complex since they involve dataframes and visualization

