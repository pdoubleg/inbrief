# Summary Engine V2

A redesigned implementation of the document processing engine, following the Strategy Pattern for better separation of concerns and maintainability.

## Motivation

The original `summary_engine` had several issues:
1. Large, monolithic files with multiple responsibilities
2. Tight coupling between strategies and the engine
3. Limited extensibility for new processing strategies
4. Unclear separation of concerns

This implementation addresses these issues by properly applying the Strategy Pattern and ensuring clean separation of responsibilities.

## Architecture

```
src/summary_engine_v2/
├── __init__.py           # Package exports
├── base.py               # ProcessingStrategy abstract base class
├── context.py            # DocumentProcessor (context class)
├── examples.py           # Usage examples
├── strategies/           # Concrete strategy implementations
│   ├── __init__.py
│   ├── primary_and_supporting.py
│   ├── primary_only.py
│   ├── medical_only.py
│   └── supporting_only.py
└── README.md             # This file
```

## Key Components

### ProcessingStrategy (Abstract Base Class)

Defines the interface that all document processing strategies must implement:
- `process()` - Main entry point for document processing
- Various helper methods for specific processing tasks

### DocumentProcessor (Context)

Acts as the context in the Strategy Pattern:
- Maintains a reference to the current strategy
- Allows strategies to be swapped at runtime
- Provides data and services to strategies
- Delegates the actual processing work to the strategy

### Concrete Strategies

Four strategies are implemented:
1. **PrimaryAndSupportingStrategy** - Processes both primary and supporting documents
2. **PrimaryOnlyStrategy** - Processes only primary documents
3. **MedicalOnlyStrategy** - Processes only medical records
4. **SupportingOnlyStrategy** - Processes only supporting documents

## Usage Examples

### Basic Usage

```python
from src.models import ConversionResult
from src.summary_engine_v2 import (
    DocumentProcessor,
    PrimaryAndSupportingStrategy,
)

# Create some documents
primary_docs = [ConversionResult(content="Primary content", filename="primary.txt")]
supporting_docs = [ConversionResult(content="Support content", filename="support.txt")]

# Create a strategy
strategy = PrimaryAndSupportingStrategy()

# Create a processor with the strategy
processor = DocumentProcessor(
    strategy=strategy,
    primary_docs=primary_docs,
    supporting_docs=supporting_docs,
)

# Process the documents
result = processor.process()
```

### Using Helper Functions

For convenience, helper functions are provided in the `examples.py` module:

```python
from src.models import ConversionResult
from src.summary_engine_v2.examples import (
    process_primary_and_supporting,
    process_primary_only,
    process_medical_only,
    process_supporting_only,
)

# Process primary and supporting documents
result = process_primary_and_supporting(
    primary_docs=[ConversionResult(...)],
    supporting_docs=[ConversionResult(...), ConversionResult(...)],
    include_exhibits_research=True,
)

# Process only primary documents
result = process_primary_only(
    primary_docs=[ConversionResult(...)],
)
```

### Switching Strategies at Runtime

One of the key benefits of the Strategy Pattern is the ability to switch strategies at runtime:

```python
# Create a processor
processor = DocumentProcessor(
    primary_docs=primary_docs,
    supporting_docs=supporting_docs,
)

# Start with a primary-only strategy
processor.set_strategy(PrimaryOnlyStrategy())
result1 = processor.process()

# Switch to a different strategy
processor.set_strategy(PrimaryAndSupportingStrategy())
result2 = processor.process()
```

## Implementation Details

### Strategy Independence

Unlike the original implementation, strategies don't take the engine as a parameter. Instead:
- Strategies receive a dictionary of data they need for processing
- They return structured results that the processor can use
- They don't maintain references to the processor

### Error Handling

Error handling is consistently applied through decorators:
- The `@handle_llm_errors` decorator is used for LLM-related operations
- Clear error messages and proper exception handling

### Documentation

All classes and methods are documented with Google-style docstrings, including:
- Descriptions of functionality
- Parameter and return type documentation
- Usage examples
- Type hints throughout the code 