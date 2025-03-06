#!/usr/bin/env python
"""
Custom Dataset Summary Evaluation Example

This example demonstrates how to:
1. Load a custom dataset from a file (CSV, JSON, etc.)
2. Run batch evaluation on the dataset
3. Generate detailed analysis and visualizations
4. Compare different summary models or approaches

Usage:
    python custom_dataset_evaluation.py --input-file INPUT_FILE --output-dir OUTPUT_DIR
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm.summary_metric import (
    batch_summary_evaluation,
    analyze_evaluation_results,
    visualize_evaluation_results,
    save_evaluation_results,
)


def load_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    Load a dataset from a file.
    
    Supports CSV, JSON, Excel, and JSONL formats based on file extension.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        List of dictionaries containing summary pairs
        
    Example format for CSV:
        id,document_id,gold_summary,model_a_summary,model_b_summary
        1,doc1,"Climate change is...", "Climate change leads...", "Global warming..."
    
    Example format for JSON:
        [
            {
                "id": "1",
                "document_id": "doc1",
                "gold_summary": "Climate change is...",
                "model_a_summary": "Climate change leads...",
                "model_b_summary": "Global warming..."
            },
            ...
        ]
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
        return df.to_dict(orient='records')
    
    elif file_ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif file_ext == '.jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
        return df.to_dict(orient='records')
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .csv, .json, .jsonl, .xlsx, or .xls")


def compare_models(
    dataset: List[Dict[str, str]],
    gold_key: str,
    model_keys: List[str],
    output_dir: Optional[str] = None,
    id_key: Optional[str] = None,
    metadata_keys: Optional[List[str]] = None,
    max_concurrency: int = 5,
) -> Dict[str, Any]:
    """
    Compare multiple summary models against the same gold summaries.
    
    Args:
        dataset: List of dictionaries containing summaries
        gold_key: Key for gold summaries in the dataset
        model_keys: List of keys for model summaries to compare
        output_dir: Directory to save results and visualizations
        id_key: Key for unique identifiers in the dataset
        metadata_keys: Additional metadata keys to include
        max_concurrency: Maximum number of concurrent evaluations
        
    Returns:
        Dictionary with evaluation results for each model
    """
    results = {}
    
    # Create output directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    # Evaluate each model
    for model_key in model_keys:
        print(f"\nEvaluating model: {model_key}")
        
        # Create model-specific output directory
        model_output_dir = None
        if output_dir:
            model_output_dir = os.path.join(output_dir, model_key)
            Path(model_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run batch evaluation
        results_df = batch_summary_evaluation(
            summary_pairs=dataset,
            gold_key=gold_key,
            pred_key=model_key,
            id_key=id_key,
            metadata_keys=metadata_keys,
            max_concurrency=max_concurrency,
            show_progress=True
        )
        
        # Analyze results
        analysis = analyze_evaluation_results(results_df)
        
        # Save results
        if model_output_dir:
            # Save DataFrame
            save_evaluation_results(
                results_df,
                os.path.join(model_output_dir, f"{model_key}_results.csv")
            )
            
            # Save analysis as JSON
            with open(os.path.join(model_output_dir, f"{model_key}_analysis.json"), 'w') as f:
                # Convert numpy values to Python native types for JSON serialization
                clean_analysis = {}
                for k, v in analysis.items():
                    if isinstance(v, dict):
                        clean_analysis[k] = {k2: float(v2) if hasattr(v2, 'item') else v2 
                                            for k2, v2 in v.items()}
                    else:
                        clean_analysis[k] = float(v) if hasattr(v, 'item') else v
                
                json.dump(clean_analysis, f, indent=2)
            
            # Generate visualizations
            visualize_evaluation_results(
                results_df,
                output_dir=model_output_dir,
                show_plots=False  # Don't show plots during batch processing
            )
        
        # Store results
        results[model_key] = {
            'df': results_df,
            'analysis': analysis
        }
    
    # Generate comparison visualizations
    if output_dir and len(model_keys) > 1:
        generate_model_comparison_plots(results, output_dir)
    
    return results


def generate_model_comparison_plots(results: Dict[str, Dict], output_dir: str) -> None:
    """
    Generate plots comparing different models.
    
    Args:
        results: Dictionary with evaluation results for each model
        output_dir: Directory to save comparison plots
    """
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    Path(comparison_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Score comparison
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    model_names = list(results.keys())
    avg_scores = [results[model]['analysis']['average_score'] for model in model_names]
    
    # Create bar chart
    plt.bar(model_names, avg_scores, color='skyblue')
    plt.title('Average Score Comparison Across Models')
    plt.xlabel('Model')
    plt.ylabel('Average Score')
    plt.ylim(0, 1)
    
    # Add value labels
    for i, score in enumerate(avg_scores):
        plt.text(i, score + 0.02, f"{score:.2f}", ha='center')
    
    # Save plot
    plt.savefig(os.path.join(comparison_dir, "score_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Importance coverage comparison
    if all('analysis' in results[model] and 'importance_coverage' in results[model]['analysis'] 
           for model in model_names):
        plt.figure(figsize=(15, 8))
        
        # Prepare data
        importance_levels = ['High', 'Medium', 'Low']
        model_coverage = {
            model: [results[model]['analysis']['importance_coverage'].get(level, 0) 
                   for level in importance_levels]
            for model in model_names
        }
        
        # Set width of bars
        bar_width = 0.2
        positions = list(range(len(importance_levels)))
        
        # Create grouped bar chart
        for i, (model, coverage) in enumerate(model_coverage.items()):
            plt.bar(
                [p + i * bar_width for p in positions],
                coverage,
                width=bar_width,
                label=model
            )
        
        # Add labels and legend
        plt.title('Key Idea Coverage by Importance Level')
        plt.xlabel('Importance Level')
        plt.ylabel('Coverage Rate')
        plt.xticks([p + bar_width * (len(model_names) - 1) / 2 for p in positions], importance_levels)
        plt.ylim(0, 1)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(comparison_dir, "coverage_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Score distribution comparison
    plt.figure(figsize=(15, 8))
    
    # Create subplots for each model
    fig, axes = plt.subplots(1, len(model_names), figsize=(15, 6), sharey=True)
    
    for i, model in enumerate(model_names):
        ax = axes[i] if len(model_names) > 1 else axes
        
        # Get scores
        scores = results[model]['df']['score']
        
        # Create histogram
        ax.hist(scores, bins=10, alpha=0.7, color=f'C{i}')
        ax.set_title(f'{model}')
        ax.set_xlabel('Score')
        
        if i == 0:
            ax.set_ylabel('Count')
    
    plt.suptitle('Score Distribution Comparison')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(comparison_dir, "score_distribution_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def run_example_with_file(
    input_file: str,
    output_dir: str = None,
    gold_key: str = "gold_summary",
    model_keys: List[str] = None,
    id_key: str = None,
) -> Dict[str, Any]:
    """
    Run the custom dataset evaluation example.
    
    Args:
        input_file: Path to the dataset file
        output_dir: Directory to save output files and visualizations
        gold_key: Key for gold summaries in the dataset
        model_keys: List of keys for model summaries to compare
        id_key: Key for unique identifiers in the dataset
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Loading dataset from: {input_file}")
    dataset = load_dataset(input_file)
    print(f"Loaded {len(dataset)} examples")
    
    # If model_keys not provided, try to infer from the first item
    if not model_keys and dataset:
        first_item = dataset[0]
        # Find keys that might contain summaries (excluding gold_key and id_key)
        potential_model_keys = [
            k for k in first_item.keys() 
            if k != gold_key and k != id_key and "summary" in k.lower()
        ]
        
        if potential_model_keys:
            model_keys = potential_model_keys
            print(f"Automatically detected model keys: {model_keys}")
        else:
            raise ValueError(
                "Could not automatically detect model keys. Please specify them using --model-keys."
            )
    
    # Run comparison
    results = compare_models(
        dataset=dataset,
        gold_key=gold_key,
        model_keys=model_keys,
        output_dir=output_dir,
        id_key=id_key,
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    print("-" * 80)
    for model, model_results in results.items():
        analysis = model_results['analysis']
        print(f"\nModel: {model}")
        print(f"- Average score: {analysis['average_score']:.2f}")
        print(f"- Score range: {analysis['min_score']:.2f} - {analysis['max_score']:.2f}")
        
        if "importance_coverage" in analysis:
            print("- Key idea coverage:")
            for level, coverage in analysis["importance_coverage"].items():
                print(f"  - {level}: {coverage:.1%}")
    
    return results


def main():
    """Parse command line arguments and run the example."""
    parser = argparse.ArgumentParser(description="Custom Dataset Summary Evaluation")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the dataset file (CSV, JSON, JSONL, or Excel)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./custom_eval_output",
        help="Directory to save output files and visualizations"
    )
    parser.add_argument(
        "--gold-key",
        type=str,
        default="gold_summary",
        help="Key for gold summaries in the dataset"
    )
    parser.add_argument(
        "--model-keys",
        type=str,
        nargs="+",
        help="Keys for model summaries to compare"
    )
    parser.add_argument(
        "--id-key",
        type=str,
        default="id",
        help="Key for unique identifiers in the dataset"
    )
    
    args = parser.parse_args()
    
    run_example_with_file(
        input_file=args.input_file,
        output_dir=args.output_dir,
        gold_key=args.gold_key,
        model_keys=args.model_keys,
        id_key=args.id_key,
    )


if __name__ == "__main__":
    main() 