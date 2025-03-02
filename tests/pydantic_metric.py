from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.usage import Usage


class KeyIdea(BaseModel):
    """Represents a key idea extracted from a passage with its importance grade."""

    idea: str
    importance: Literal["High", "Medium", "Low"] = Field(
        ..., description="Importance grade: 'High', 'Medium', or 'Low'"
    )


class BreakdownResult(BaseModel):
    """Result of breaking down a passage into key ideas."""

    key_ideas: List[KeyIdea]

    @property
    def formatted_key_ideas(self) -> str:
        return "\n".join(
            [
                f"{i + 1}. {idea.idea} ({idea.importance})."
                for i, idea in enumerate(self.key_ideas)
            ]
        )


class EvaluationResult(BaseModel):
    """Result of evaluating a summary against key ideas."""

    explanation: str = Field(description="Explanation of the evaluation result")
    binary_scores: List[bool]
    overall_score: float = Field(..., ge=0.0, le=1.0)


class EvalDeps(BaseModel):
    gold_summary: str
    pred_summary: str
    key_ideas: Optional[BreakdownResult] = None
    evaluation: Optional[EvaluationResult] = None
    score: Optional[float] = None
    usages: List[Usage] = Field(default_factory=list)


# Agent for breaking down passages into key ideas
breakdown_agent = Agent[None, BreakdownResult](
    "openai:gpt-4o",
    result_type=BreakdownResult,
    system_prompt=(
        "You are an expert at analyzing passages and extracting key ideas.\n"
        "Given a passage, break it down into key ideas and assign each an importance grade.\n"
        "Importance grades should be 'High', 'Medium', or 'Low' based on how central the idea is to the passage.\n"
        "Return both a structured list of ideas with grades and a formatted numbered list."
    ),
)


# Agent for evaluating summaries against key ideas
evaluation_agent = Agent[None, EvaluationResult](
    "openai:gpt-4o",
    result_type=EvaluationResult,
    system_prompt=(
        "You are an expert at evaluating the quality of summaries.\n"
        "Compare a summary to a list of key ideas from the original passage.\n"
        "For each key idea, determine if the summary contains it (True) or not (False).\n"
        "Then compute an overall score from 0.0 to 1.0 based on how well the summary captures the key ideas,\n"
        "weighted by their importance."
    ),
)


async def breakdown_gold_summary(gold_summary: str) -> BreakdownResult:
    """
    Break down a gold label user summary into key ideas with importance grades.

    Args:
        gold_summary: The gold label user summary to analyze

    Returns:
        BreakdownResult containing key ideas and their importance grades
    """
    result = await breakdown_agent.run(
        f"Analyze the following gold label user summary and extract its key ideas with importance grades:\n\n{gold_summary}",
    )
    return result


async def evaluate_llm_summary(
    llm_summary: str, key_ideas: BreakdownResult
) -> EvaluationResult:
    """
    Evaluate a llm summary against key ideas from a gold label user summary.

    Args:
        llm_summary: The llm summary to evaluate
        key_ideas: The breakdown of key ideas from the gold summary

    Returns:
        EvaluationResult containing binary scores and overall score
    """
    result = await evaluation_agent.run(
        f"Evaluate the following llm summary against these key ideas:\n\n"
        f"KEY IDEAS:\n{key_ideas.formatted_key_ideas}\n\n"
        f"LLM SUMMARY TO EVALUATE:\n{llm_summary}",
    )
    return result


def calculate_weighted_score(
    key_ideas: BreakdownResult, evaluation: EvaluationResult
) -> float:
    """
    Calculate a weighted score based on the key idea importance and binary scores.

    Args:
        key_ideas: The breakdown of key ideas from the gold summary
        evaluation: The evaluation result containing binary scores

    Returns:
        Weighted score between 0.0 and 1.0
    """
    try:
        # Define weight for each importance grade
        weight_map = {"High": 1.0, "Medium": 0.7, "Low": 0.2}

        # Calculate weighted sum of binary scores
        weighted_sum = sum(
            weight_map.get(idea.importance, 0.2) * int(score)
            for idea, score in zip(key_ideas.key_ideas, evaluation.binary_scores)
        )

        # Calculate total possible weight
        total_weight = sum(
            weight_map.get(idea.importance, 0.2) for idea in key_ideas.key_ideas
        )
        return weighted_sum / total_weight
    except Exception as _:
        # Fallback to the overall score from the evaluation
        return evaluation.overall_score or 0.0


async def evaluate_summary_pair(
    deps: EvalDeps,
) -> EvalDeps:
    """
    Evaluate how well a summary captures the key ideas in a gold summary.

    This function uses a sequential approach to:
    1. Break down the gold summary into key ideas with importance grades
    2. Evaluate the summary against those key ideas
    3. Calculate a weighted score

    Args:
        deps: Object containing the gold label user summary and llm summary
        usage_limits: Optional limits on API usage

    Returns:
        The populated EvalDeps object with evaluation results

    Example:
        >>> deps = EvalDeps(
        ...     gold_summary="Climate change is causing rising sea levels and extreme weather...",
        ...     pred_summary="Climate change leads to sea level rise and weather extremes."
        ... )
        >>> result = await evaluate_summary_pair(deps)
        >>> print(result.score)
        0.85
    """
    usages: List[Usage] = []

    # Step 1: Break down gold summary into key ideas
    key_ideas = await breakdown_gold_summary(deps.gold_summary)
    deps.key_ideas = key_ideas.data
    usages.append(key_ideas.usage())

    # Step 2: Evaluate llm summary against key ideas
    evaluation = await evaluate_llm_summary(deps.pred_summary, deps.key_ideas)
    deps.evaluation = evaluation.data
    usages.append(evaluation.usage())

    # Step 3: Calculate weighted score
    deps.score = calculate_weighted_score(deps.key_ideas, deps.evaluation)
    deps.usages = usages
    return deps


def summary_evaluation_metric(deps: EvalDeps) -> EvalDeps:
    """
    Compare the llm summary to the gold label user summary.

    Args:
        deps: Object containing the gold label user summary and llm summary

    Returns:
        EvalDeps object with the results of the evaluation
    """
    import asyncio

    # Run the evaluation asynchronously and populate the deps object
    populated_deps = asyncio.run(evaluate_summary_pair(deps))

    return populated_deps
