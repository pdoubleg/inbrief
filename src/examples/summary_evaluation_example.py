#!/usr/bin/env python
"""
Summary Evaluation Batch Processing Example

This example demonstrates how to use the batch evaluation functionality to:
1. Process multiple summary pairs
2. Analyze the evaluation results
3. Visualize the results
4. Save the results to files

Usage:
    python summary_evaluation_example.py [--output-dir OUTPUT_DIR]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any


# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm.summary_metric import (
    batch_summary_evaluation,
    analyze_evaluation_results,
    visualize_evaluation_results,
    save_evaluation_results,
)


# Sample data for demonstration
SAMPLE_SUMMARIES = [
    {
        "id": "climate_change",
        "gold_summary": """
        Climate change is causing rising sea levels and extreme weather events worldwide. 
        The primary driver is human activity, especially the burning of fossil fuels which releases greenhouse gases.
        These gases trap heat in the atmosphere, leading to global warming.
        The Paris Agreement aims to limit global warming to well below 2°C compared to pre-industrial levels.
        Renewable energy sources like solar and wind power are crucial for reducing carbon emissions.
        Individual actions such as reducing meat consumption and using public transportation can help mitigate climate change.
        """,
        "pred_summary": """
        Climate change is leading to rising sea levels and more frequent extreme weather. 
        It's primarily caused by human activities that release greenhouse gases, especially burning fossil fuels.
        The Paris Agreement is an international effort to limit warming to below 2°C.
        Renewable energy is important for reducing emissions, and individuals can help by changing their habits.
        """,
        "category": "science",
    },
    {
        "id": "healthcare_policy",
        "gold_summary": """
        The new healthcare policy expands coverage to previously uninsured populations.
        It includes subsidies for low-income individuals to purchase insurance.
        The policy implements price controls on prescription medications.
        It establishes preventive care services with no co-payments.
        The policy creates a public option to compete with private insurance plans.
        It increases funding for rural healthcare facilities to address access disparities.
        """,
        "pred_summary": """
        The healthcare policy provides coverage to more people who were previously uninsured.
        It offers financial assistance to help low-income people buy insurance.
        The policy controls prescription drug prices and makes preventive care free.
        It also increases funding for healthcare in rural areas.
        """,
        "category": "policy",
    },
    {
        "id": "quantum_computing",
        "gold_summary": """
        Quantum computing uses quantum bits or qubits instead of classical bits.
        Qubits can exist in multiple states simultaneously due to superposition.
        Quantum entanglement allows qubits to be correlated in ways impossible with classical bits.
        Quantum computers can potentially solve certain problems exponentially faster than classical computers.
        Current quantum computers are noisy and prone to errors, requiring error correction.
        Potential applications include cryptography, drug discovery, and optimization problems.
        """,
        "pred_summary": """
        Quantum computing is based on qubits rather than classical bits, allowing for superposition.
        These computers can potentially solve certain problems much faster than traditional computers.
        Current quantum systems have high error rates and require error correction.
        They could revolutionize fields like cryptography and drug discovery.
        """,
        "category": "science",
    },
]


def run_example(output_dir: str = None) -> Dict[str, Any]:
    """
    Run the summary evaluation example.
    
    Args:
        output_dir: Directory to save output files and visualizations
        
    Returns:
        Dictionary containing results and analysis
    """
    print("Running Summary Evaluation Batch Processing Example")
    print("-" * 80)
    
    # Create output directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {output_dir}")
    
    # Step 1: Run batch evaluation
    print("\n1. Running batch evaluation on sample summaries...")
    results_df = batch_summary_evaluation(
        summary_pairs=SAMPLE_SUMMARIES,
        id_key="id",
        metadata_keys=["category"],
        show_progress=True
    )
    
    # Print summary of results
    print(f"\nProcessed {len(results_df)} summary pairs")
    print(f"Average score: {results_df['score'].mean():.2f}")
    
    # Step 2: Analyze results
    print("\n2. Analyzing evaluation results...")
    analysis = analyze_evaluation_results(results_df)
    
    # Print key analysis findings
    print("\nKey Analysis Findings:")
    print(f"- Average score: {analysis['average_score']:.2f}")
    print(f"- Score range: {analysis['min_score']:.2f} - {analysis['max_score']:.2f}")
    
    if "importance_coverage" in analysis:
        print("\nKey Idea Coverage by Importance:")
        for level, coverage in analysis["importance_coverage"].items():
            print(f"- {level}: {coverage:.1%}")
    
    # Step 3: Visualize results
    print("\n3. Generating visualizations...")
    if output_dir:
        plot_paths = visualize_evaluation_results(
            results_df,
            output_dir=output_dir,
            show_plots=True
        )
        print(f"\nGenerated {len(plot_paths)} visualization plots in {output_dir}")
    else:
        visualize_evaluation_results(results_df, show_plots=True)
    
    # Step 4: Save results
    if output_dir:
        print("\n4. Saving evaluation results...")
        
        # Save as CSV
        csv_path = save_evaluation_results(
            results_df,
            os.path.join(output_dir, "evaluation_results.csv")
        )
        print(f"Results saved as CSV: {csv_path}")
        
        # Save as JSON
        json_path = save_evaluation_results(
            results_df,
            os.path.join(output_dir, "evaluation_results.json"),
            format="json"
        )
        print(f"Results saved as JSON: {json_path}")
    
    # Return results for potential further use
    return {
        "results_df": results_df,
        "analysis": analysis
    }


def main():
    """Parse command line arguments and run the example."""
    parser = argparse.ArgumentParser(description="Summary Evaluation Example")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./summary_eval_output",
        help="Directory to save output files and visualizations"
    )
    args = parser.parse_args()
    
    run_example(args.output_dir)


if __name__ == "__main__":
    main() 