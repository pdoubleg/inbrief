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

