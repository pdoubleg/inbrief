from pathlib import Path
from typing import List, Literal, Optional, Dict, Any, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.usage import Usage
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
    # Run the evaluation asynchronously and populate the deps object
    populated_deps = asyncio.run(evaluate_summary_pair(deps))

    return populated_deps


async def batch_evaluate_summaries(
    summary_pairs: Sequence[Dict[str, str]],
    gold_key: str = "gold_summary",
    pred_key: str = "pred_summary",
    id_key: Optional[str] = None,
    metadata_keys: Optional[List[str]] = None,
    max_concurrency: int = 5,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Evaluate multiple summary pairs in batch and return results as a pandas DataFrame.
    
    This function processes multiple summary pairs concurrently (up to max_concurrency)
    and collects detailed diagnostic information for each evaluation.
    
    Args:
        summary_pairs: A sequence of dictionaries, each containing at least gold and predicted summaries
        gold_key: The key in each dict that contains the gold summary (default: "gold_summary")
        pred_key: The key in each dict that contains the predicted summary (default: "pred_summary")
        id_key: Optional key in each dict to use as an identifier in the results
        metadata_keys: Optional list of additional keys from input dicts to include in results
        max_concurrency: Maximum number of concurrent evaluations (default: 5)
        show_progress: Whether to show a progress bar (default: True)
    
    Returns:
        A pandas DataFrame containing evaluation results and diagnostic information
        
    Example:
        >>> summary_pairs = [
        ...     {"id": "doc1", "gold_summary": "Climate change is causing...", "pred_summary": "Climate change leads to..."},
        ...     {"id": "doc2", "gold_summary": "The new healthcare policy...", "pred_summary": "Healthcare reforms..."}
        ... ]
        >>> results_df = await batch_evaluate_summaries(summary_pairs, id_key="id")
        >>> print(results_df.columns)
        Index(['id', 'gold_summary', 'pred_summary', 'score', 'key_ideas_count',
               'high_importance_count', 'medium_importance_count', 'low_importance_count',
               'binary_scores', 'explanation', 'token_usage'], dtype='object')
    """
    # Prepare tasks for concurrent execution
    async def process_pair(pair: Dict[str, str], pair_idx: int) -> Dict[str, Any]:
        # Create EvalDeps object
        deps = EvalDeps(
            gold_summary=pair[gold_key],
            pred_summary=pair[pred_key],
        )
        
        # Run evaluation
        result = await evaluate_summary_pair(deps)
        
        # Prepare result dictionary with all relevant data
        result_dict = {
            "gold_summary": result.gold_summary,
            "pred_summary": result.pred_summary,
            "score": result.score,
        }
        
        # Add identifier if provided
        if id_key and id_key in pair:
            result_dict["id"] = pair[id_key]
        else:
            result_dict["id"] = f"pair_{pair_idx}"
            
        # Add any requested metadata
        if metadata_keys:
            for key in metadata_keys:
                if key in pair:
                    result_dict[key] = pair[key]
        
        # Add diagnostic data from key ideas
        if result.key_ideas:
            result_dict["key_ideas_count"] = len(result.key_ideas.key_ideas)
            
            # Count by importance
            importance_counts = {"High": 0, "Medium": 0, "Low": 0}
            for idea in result.key_ideas.key_ideas:
                importance_counts[idea.importance] += 1
                
            result_dict["high_importance_count"] = importance_counts["High"]
            result_dict["medium_importance_count"] = importance_counts["Medium"]
            result_dict["low_importance_count"] = importance_counts["Low"]
            
            # Store formatted key ideas for reference
            result_dict["formatted_key_ideas"] = result.key_ideas.formatted_key_ideas
        
        # Add evaluation details
        if result.evaluation:
            result_dict["binary_scores"] = result.evaluation.binary_scores
            result_dict["explanation"] = result.evaluation.explanation
            result_dict["overall_model_score"] = result.evaluation.overall_score
            
        # Add token usage information
        if result.usages:
            total_prompt_tokens = sum(usage.request_tokens for usage in result.usages)
            total_completion_tokens = sum(usage.response_tokens for usage in result.usages)
            total_cost = 1
            
            result_dict["token_usage"] = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "cost": total_cost
            }
            
        return result_dict
    
    # Create tasks for all pairs
    tasks = [
        process_pair(pair, idx) 
        for idx, pair in enumerate(summary_pairs)
    ]
    
    # Process tasks with concurrency limit
    results = []
    
    # Use tqdm if progress bar is requested
    if show_progress:
        for task_batch in [tasks[i:i+max_concurrency] for i in range(0, len(tasks), max_concurrency)]:
            batch_results = await tqdm.gather(*task_batch)
            results.extend(batch_results)
    else:
        # Process in batches to respect concurrency limit
        for i in range(0, len(tasks), max_concurrency):
            batch = tasks[i:i+max_concurrency]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df


def batch_summary_evaluation(
    summary_pairs: Sequence[Dict[str, str]],
    gold_key: str = "gold_summary",
    pred_key: str = "pred_summary",
    id_key: Optional[str] = None,
    metadata_keys: Optional[List[str]] = None,
    max_concurrency: int = 5,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Synchronous wrapper for batch_evaluate_summaries.
    
    Evaluates multiple summary pairs and returns results as a pandas DataFrame.
    
    Args:
        summary_pairs: A sequence of dictionaries, each containing at least gold and predicted summaries
        gold_key: The key in each dict that contains the gold summary (default: "gold_summary")
        pred_key: The key in each dict that contains the predicted summary (default: "pred_summary")
        id_key: Optional key in each dict to use as an identifier in the results
        metadata_keys: Optional list of additional keys from input dicts to include in results
        max_concurrency: Maximum number of concurrent evaluations (default: 5)
        show_progress: Whether to show a progress bar (default: True)
    
    Returns:
        A pandas DataFrame containing evaluation results and diagnostic information
    """
    return asyncio.run(batch_evaluate_summaries(
        summary_pairs=summary_pairs,
        gold_key=gold_key,
        pred_key=pred_key,
        id_key=id_key,
        metadata_keys=metadata_keys,
        max_concurrency=max_concurrency,
        show_progress=show_progress,
    ))


def save_evaluation_results(
    df: pd.DataFrame,
    output_path: str,
    format: str = "csv",
    include_token_usage: bool = True,
) -> str:
    """
    Save batch evaluation results to a file.
    
    Args:
        df: DataFrame containing evaluation results
        output_path: Path where to save the results
        format: Output format, one of 'csv', 'json', 'excel', or 'parquet' (default: 'csv')
        include_token_usage: Whether to include token usage details in the output (default: True)
        
    Returns:
        Path to the saved file
        
    Example:
        >>> results_df = batch_summary_evaluation(summary_pairs)
        >>> save_path = save_evaluation_results(results_df, "evaluation_results.csv")
        >>> print(f"Results saved to {save_path}")
        Results saved to evaluation_results.csv
    """
    # Create a copy of the DataFrame to avoid modifying the original
    output_df = df.copy()
    
    # Handle token usage column which is a dictionary
    if "token_usage" in output_df.columns:
        if include_token_usage:
            # Expand token usage dictionary into separate columns
            for row_idx, row in output_df.iterrows():
                if isinstance(row["token_usage"], dict):
                    for key, value in row["token_usage"].items():
                        output_df.at[row_idx, f"token_usage_{key}"] = value
        
        # Remove the original token_usage column
        output_df = output_df.drop(columns=["token_usage"])
    
    # Save in the requested format
    format = format.lower()
    if format == "csv":
        output_df.to_csv(output_path, index=False)
    elif format == "json":
        output_df.to_json(output_path, orient="records", indent=2)
    elif format == "excel":
        output_df.to_excel(output_path, index=False)
    elif format == "parquet":
        output_df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'json', 'excel', or 'parquet'.")
    
    return output_path


def analyze_evaluation_results(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze batch evaluation results and generate summary statistics.
    
    This function computes various statistics and metrics from the evaluation results,
    which can be useful for understanding model performance and tuning.
    
    Args:
        df: DataFrame containing evaluation results from batch_summary_evaluation
        
    Returns:
        Dictionary containing summary statistics and analysis
        
    Example:
        >>> results_df = batch_summary_evaluation(summary_pairs)
        >>> analysis = analyze_evaluation_results(results_df)
        >>> print(f"Average score: {analysis['average_score']:.2f}")
        Average score: 0.78
        >>> print(f"High importance coverage: {analysis['importance_coverage']['High']:.2%}")
        High importance coverage: 85.50%
    """
    analysis = {}
    
    # Basic statistics on scores
    if "score" in df.columns and not df["score"].empty:
        # Handle cases where there are scores to analyze
        if len(df["score"].dropna()) > 0:
            analysis["average_score"] = df["score"].mean()
            analysis["median_score"] = df["score"].median() 
            analysis["min_score"] = df["score"].min()
            analysis["max_score"] = df["score"].max()
            analysis["std_score"] = df["score"].std()
        else:
            # Set default values if no valid scores
            analysis["average_score"] = 0.0
            analysis["median_score"] = 0.0
            analysis["min_score"] = 0.0
            analysis["max_score"] = 0.0
            analysis["std_score"] = 0.0
    else:
        # Set default values if score column doesn't exist
        analysis["average_score"] = 0.0
        analysis["median_score"] = 0.0
        analysis["min_score"] = 0.0
        analysis["max_score"] = 0.0
        analysis["std_score"] = 0.0
        
        # Score distribution
        analysis["score_distribution"] = {
            "excellent (0.9-1.0)": ((df["score"] >= 0.9) & (df["score"] <= 1.0)).sum() / len(df),
            "good (0.7-0.9)": ((df["score"] >= 0.7) & (df["score"] < 0.9)).sum() / len(df),
            "fair (0.5-0.7)": ((df["score"] >= 0.5) & (df["score"] < 0.7)).sum() / len(df),
            "poor (0.0-0.5)": ((df["score"] >= 0.0) & (df["score"] < 0.5)).sum() / len(df),
        }
    
    # Analyze key ideas coverage
    if all(col in df.columns for col in ["high_importance_count", "medium_importance_count", "low_importance_count"]):
        # Calculate total key ideas by importance
        total_high = df["high_importance_count"].sum()
        total_medium = df["medium_importance_count"].sum()
        total_low = df["low_importance_count"].sum()
        
        # Calculate coverage by importance level
        if "binary_scores" in df.columns:
            # This is more complex as we need to extract from the binary_scores lists
            high_covered = 0
            medium_covered = 0
            low_covered = 0
            total_high_ideas = 0
            total_medium_ideas = 0
            total_low_ideas = 0
            
            for idx, row in df.iterrows():
                if not isinstance(row.get("binary_scores"), list):
                    continue
                    
                binary_scores = row["binary_scores"]
                
                # Count covered ideas by importance
                high_count = row["high_importance_count"]
                medium_count = row["medium_importance_count"]
                low_count = row["low_importance_count"]
                
                # Track position in binary_scores list
                pos = 0
                
                # Count high importance coverage
                for i in range(high_count):
                    if pos < len(binary_scores) and binary_scores[pos]:
                        high_covered += 1
                    total_high_ideas += 1
                    pos += 1
                
                # Count medium importance coverage
                for i in range(medium_count):
                    if pos < len(binary_scores) and binary_scores[pos]:
                        medium_covered += 1
                    total_medium_ideas += 1
                    pos += 1
                
                # Count low importance coverage
                for i in range(low_count):
                    if pos < len(binary_scores) and binary_scores[pos]:
                        low_covered += 1
                    total_low_ideas += 1
                    pos += 1
            
            # Calculate coverage percentages
            analysis["importance_coverage"] = {
                "High": high_covered / total_high_ideas if total_high_ideas > 0 else 0,
                "Medium": medium_covered / total_medium_ideas if total_medium_ideas > 0 else 0,
                "Low": low_covered / total_low_ideas if total_low_ideas > 0 else 0,
                "Overall": (high_covered + medium_covered + low_covered) / 
                           (total_high_ideas + total_medium_ideas + total_low_ideas) 
                           if (total_high_ideas + total_medium_ideas + total_low_ideas) > 0 else 0
            }
        
        # Key idea distribution
        total_ideas = total_high + total_medium + total_low
        if total_ideas > 0:
            analysis["key_idea_distribution"] = {
                "High": total_high / total_ideas,
                "Medium": total_medium / total_ideas,
                "Low": total_low / total_ideas
            }
            
        # Average number of key ideas per summary
        analysis["avg_key_ideas_per_summary"] = df["key_ideas_count"].mean() if "key_ideas_count" in df.columns else None
    
    # Token usage statistics if available
    token_usage_cols = [col for col in df.columns if col.startswith("token_usage_")]
    if token_usage_cols:
        analysis["token_usage"] = {}
        for col in token_usage_cols:
            metric = col.replace("token_usage_", "")
            analysis["token_usage"][f"avg_{metric}"] = df[col].mean()
            analysis["token_usage"][f"total_{metric}"] = df[col].sum()
    
    # Correlation between metrics
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) > 1:
        analysis["correlations"] = {}
        if "score" in numeric_cols:
            # Correlations with the score
            for col in numeric_cols:
                if col != "score":
                    analysis["correlations"][f"score_vs_{col}"] = df["score"].corr(df[col])
    
    return analysis


def visualize_evaluation_results(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
    show_plots: bool = True,
    plot_types: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Generate visualizations from batch evaluation results.
    
    This function creates various plots to help analyze and understand the evaluation results.
    
    Args:
        df: DataFrame containing evaluation results from batch_summary_evaluation
        output_dir: Directory to save plots (if None, plots are not saved)
        show_plots: Whether to display the plots (default: True)
        plot_types: List of plot types to generate, options include:
                   ['score_distribution', 'importance_coverage', 'key_ideas_count',
                    'token_usage', 'correlations', 'all']
                   If None, generates all plots
        
    Returns:
        Dictionary mapping plot names to their file paths (if saved)
        
    Example:
        >>> results_df = batch_summary_evaluation(summary_pairs)
        >>> plot_paths = visualize_evaluation_results(results_df, output_dir="./plots")
        >>> print(f"Score distribution plot saved to: {plot_paths['score_distribution']}")
        Score distribution plot saved to: ./plots/score_distribution.png
    """
    # Set up plot style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Determine which plots to generate
    all_plot_types = [
        'score_distribution', 'importance_coverage', 'key_ideas_count',
        'token_usage', 'correlations'
    ]
    
    if plot_types is None or 'all' in plot_types:
        plot_types = all_plot_types
    else:
        plot_types = [pt for pt in plot_types if pt in all_plot_types]
    
    # Create output directory if needed
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store plot file paths
    plot_paths = {}
    
    # 1. Score Distribution
    if 'score_distribution' in plot_types and 'score' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Histogram of scores
        ax = sns.histplot(df['score'], bins=20, kde=True)
        ax.set_title('Distribution of Summary Evaluation Scores')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        
        # Add vertical lines for quartiles
        quartiles = df['score'].quantile([0.25, 0.5, 0.75]).values
        colors = ['r', 'g', 'b']
        labels = ['25th Percentile', 'Median', '75th Percentile']
        
        for q, c, l in zip(quartiles, colors, labels):
            plt.axvline(q, color=c, linestyle='--', label=l)
        
        plt.legend()
        
        if output_dir:
            path = f"{output_dir}/score_distribution.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plot_paths['score_distribution'] = path
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 2. Importance Coverage
    if 'importance_coverage' in plot_types and all(col in df.columns for col in 
                                                 ["high_importance_count", "medium_importance_count", "low_importance_count"]):
        # Calculate coverage data
        analysis = analyze_evaluation_results(df)
        if 'importance_coverage' in analysis:
            plt.figure(figsize=(10, 6))
            
            # Create bar chart of coverage by importance
            importance_levels = ['High', 'Medium', 'Low', 'Overall']
            coverage_values = [analysis['importance_coverage'].get(level, 0) for level in importance_levels]
            
            ax = sns.barplot(x=importance_levels, y=coverage_values)
            ax.set_title('Coverage of Key Ideas by Importance Level')
            ax.set_xlabel('Importance Level')
            ax.set_ylabel('Coverage Rate')
            ax.set_ylim(0, 1)
            
            # Add percentage labels on bars
            for i, v in enumerate(coverage_values):
                ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
            
            if output_dir:
                path = f"{output_dir}/importance_coverage.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plot_paths['importance_coverage'] = path
            
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    # 3. Key Ideas Count
    if 'key_ideas_count' in plot_types and 'key_ideas_count' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Distribution of key ideas count
        sns.histplot(df['key_ideas_count'], bins=10, kde=True, ax=ax1)
        ax1.set_title('Distribution of Key Ideas Count')
        ax1.set_xlabel('Number of Key Ideas')
        ax1.set_ylabel('Count')
        
        # Stacked bar chart of importance levels
        if all(col in df.columns for col in ["high_importance_count", "medium_importance_count", "low_importance_count"]):
            # Prepare data for stacked bar chart
            importance_data = df[['high_importance_count', 'medium_importance_count', 'low_importance_count']].mean()
            importance_data.index = ['High', 'Medium', 'Low']
            
            importance_data.plot.bar(stacked=False, ax=ax2, color=['red', 'blue', 'green'])
            ax2.set_title('Average Number of Key Ideas by Importance')
            ax2.set_xlabel('Importance Level')
            ax2.set_ylabel('Average Count')
            
            # Add value labels
            for i, v in enumerate(importance_data):
                ax2.text(i, v + 0.1, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        
        if output_dir:
            path = f"{output_dir}/key_ideas_count.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plot_paths['key_ideas_count'] = path
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 4. Token Usage
    token_usage_cols = [col for col in df.columns if col.startswith("token_usage_")]
    if 'token_usage' in plot_types and token_usage_cols:
        plt.figure(figsize=(12, 6))
        
        # Select relevant columns
        usage_df = df[token_usage_cols].copy()
        
        # Rename columns for better display
        usage_df.columns = [col.replace("token_usage_", "") for col in usage_df.columns]
        
        # Calculate average usage
        avg_usage = usage_df.mean()
        
        # Create bar chart
        ax = avg_usage.plot.bar(color='skyblue')
        ax.set_title('Average Token Usage per Evaluation')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Average Count')
        
        # Add value labels
        for i, v in enumerate(avg_usage):
            ax.text(i, v + 5, f"{v:.1f}", ha='center')
        
        if output_dir:
            path = f"{output_dir}/token_usage.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plot_paths['token_usage'] = path
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 5. Correlations
    if 'correlations' in plot_types:
        # Select numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Filter out token usage columns to avoid too many variables
        numeric_cols = [col for col in numeric_cols if not col.startswith("token_usage_")]
        
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            
            # Create correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            ax = sns.heatmap(
                corr_matrix, 
                mask=mask,
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm",
                vmin=-1, 
                vmax=1, 
                center=0,
                square=True, 
                linewidths=.5
            )
            ax.set_title('Correlation Matrix of Evaluation Metrics')
            
            if output_dir:
                path = f"{output_dir}/correlations.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plot_paths['correlations'] = path
            
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    return plot_paths
