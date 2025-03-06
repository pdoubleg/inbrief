# Summary Evaluation Examples

This directory contains example scripts that demonstrate how to use the summary evaluation functionality in the `src/llm/summary_metric.py` module.

## Overview

The summary evaluation system allows you to:

1. Evaluate how well a generated summary captures the key ideas in a gold standard summary
2. Process multiple summary pairs in batch
3. Analyze the evaluation results with detailed metrics
4. Visualize the results with various plots
5. Compare different summary models or approaches

## Available Examples

### 1. Basic Example

`summary_evaluation_example.py` demonstrates the basic usage of the summary evaluation functionality with sample data.

```bash
python summary_evaluation_example.py --output-dir ./summary_eval_output
```

This example:
- Processes a few sample summary pairs
- Analyzes the evaluation results
- Generates visualizations
- Saves the results to CSV and JSON files

### 2. Custom Dataset Example

`custom_dataset_evaluation.py` shows how to evaluate summaries from a custom dataset loaded from a file.

```bash
python custom_dataset_evaluation.py --input-file your_dataset.csv --output-dir ./custom_eval_output
```

This example:
- Loads a dataset from a file (CSV, JSON, JSONL, or Excel)
- Evaluates multiple summary models against the same gold summaries
- Generates detailed analysis and visualizations for each model
- Creates comparison plots to compare different models

#### Supported File Formats

- **CSV**: Comma-separated values file with headers
- **JSON**: JSON array of objects
- **JSONL**: JSON Lines format (one JSON object per line)
- **Excel**: Excel spreadsheet (.xlsx or .xls)

#### Example Dataset Format

For CSV:
```
id,document_id,gold_summary,model_a_summary,model_b_summary
1,doc1,"Climate change is...", "Climate change leads...", "Global warming..."
2,doc2,"The healthcare policy...", "Healthcare reforms...", "The new policy..."
```

For JSON:
```json
[
    {
        "id": "1",
        "document_id": "doc1",
        "gold_summary": "Climate change is...",
        "model_a_summary": "Climate change leads...",
        "model_b_summary": "Global warming..."
    },
    {
        "id": "2",
        "document_id": "doc2",
        "gold_summary": "The healthcare policy...",
        "model_a_summary": "Healthcare reforms...",
        "model_b_summary": "The new policy..."
    }
]
```

## Command Line Arguments

### Summary Evaluation Example

```
usage: summary_evaluation_example.py [-h] [--output-dir OUTPUT_DIR]

Summary Evaluation Example

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Directory to save output files and visualizations
```

### Custom Dataset Evaluation

```
usage: custom_dataset_evaluation.py [-h] --input-file INPUT_FILE [--output-dir OUTPUT_DIR] [--gold-key GOLD_KEY] [--model-keys MODEL_KEYS [MODEL_KEYS ...]] [--id-key ID_KEY]

Custom Dataset Summary Evaluation

options:
  -h, --help            show this help message and exit
  --input-file INPUT_FILE
                        Path to the dataset file (CSV, JSON, JSONL, or Excel)
  --output-dir OUTPUT_DIR
                        Directory to save output files and visualizations
  --gold-key GOLD_KEY   Key for gold summaries in the dataset
  --model-keys MODEL_KEYS [MODEL_KEYS ...]
                        Keys for model summaries to compare
  --id-key ID_KEY       Key for unique identifiers in the dataset
```

## Output

The examples generate the following outputs:

1. **CSV/JSON files** with detailed evaluation results
2. **Analysis JSON files** with summary statistics
3. **Visualization plots**:
   - Score distribution
   - Importance coverage
   - Key ideas count
   - Token usage
   - Correlations
   - Model comparisons (when evaluating multiple models)

## Using in Your Own Code

You can also use the summary evaluation functionality in your own code:

```python
from src.llm.summary_metric import (
    batch_summary_evaluation,
    analyze_evaluation_results,
    visualize_evaluation_results,
    save_evaluation_results,
)

# Define your summary pairs
summary_pairs = [
    {
        "id": "doc1",
        "gold_summary": "Climate change is causing...",
        "pred_summary": "Climate change leads to..."
    },
    # More pairs...
]

# Run batch evaluation
results_df = batch_summary_evaluation(
    summary_pairs=summary_pairs,
    id_key="id",
    show_progress=True
)

# Analyze results
analysis = analyze_evaluation_results(results_df)
print(f"Average score: {analysis['average_score']:.2f}")

# Visualize results
visualize_evaluation_results(
    results_df,
    output_dir="./plots",
    show_plots=True
)

# Save results
save_evaluation_results(
    results_df,
    "evaluation_results.csv"
)
``` 