# Pydantic-AI Graph Application Guide

## Introduction

[Pydantic-AI](https://ai.pydantic.dev/) is a Python framework designed to build structured AI applications using **graph-based orchestration**. This guide explains how to design, implement, and execute **Pydantic-AI Graphs** step by step, using real-world examples.

## What is a Pydantic-AI Graph?

A **Pydantic-AI Graph** is a directed workflow where execution flows from **nodes** (units of execution) based on logical transitions. Each node represents an action, often an **LLM call** or a **tool execution**, and can lead to one or more next steps. 

- Nodes are subclasses of `BaseNode` and define a `run(ctx)` function.
- Edges are **implicit**: the next step is determined by the return type of `run()`.
- The graph is dynamically constructed based on these transitions.

## Components of a Graph

1. **State (`StateT`)**: The shared object passed between nodes to maintain context.
2. **Dependencies (`DepsT`)**: External resources (databases, APIs) injected into nodes.
3. **Nodes (`BaseNode`)**: Execution units with logic for each step.
4. **Edges (Transitions)**: Defined by the return type of `run()`.
5. **Graph (`Graph`)**: Assembles nodes and manages execution.

---

## Example: A Simple Question-Answering Graph

This example builds an **interactive agent** that asks a question, evaluates the user's response, and provides feedback.

### Step 1: Define the State Object
```python
from dataclasses import dataclass

@dataclass
class QuestionState:
    question: str | None = None
    ask_history: list = None
    eval_history: list = None
```

- `question`: The question asked by the AI.
- `ask_history`: Keeps track of previous interactions.
- `eval_history`: Stores past evaluations.

### Step 2: Define LLM Agents
```python
from pydantic import BaseModel
from pydantic_ai import Agent, format_as_xml

ask_agent = Agent('openai:gpt-4', result_type=str)
class EvaluationResult(BaseModel):
    correct: bool
    comment: str

evaluate_agent = Agent('openai:gpt-4', result_type=EvaluationResult,
                       system_prompt="Evaluate the correctness of an answer.")
```

- `ask_agent`: Generates a question.
- `evaluate_agent`: Assesses whether an answer is correct.

### Step 3: Define Graph Nodes

#### Node 1: Ask a Question
```python
from pydantic_graph import BaseNode, GraphRunContext

class Ask(BaseNode[QuestionState]):
    async def run(self, ctx: GraphRunContext[QuestionState]) -> "Answer":
        result = await ask_agent.run("Ask a trivia question.")
        ctx.state.question = result.data
        return Answer()
```

#### Node 2: Capture User Answer
```python
class Answer(BaseNode[QuestionState]):
    answer: str | None = None
    async def run(self, ctx: GraphRunContext[QuestionState]) -> "Evaluate":
        return Evaluate(self.answer)
```

#### Node 3: Evaluate the Answer
```python
class Evaluate(BaseNode[QuestionState]):
    answer: str
    async def run(self, ctx: GraphRunContext[QuestionState]) -> "Congratulate | Reprimand":
        eval_input = format_as_xml({"question": ctx.state.question, "answer": self.answer})
        result = await evaluate_agent.run(eval_input)
        return Congratulate() if result.data.correct else Reprimand()
```

#### Node 4: Provide Feedback
```python
class Congratulate(BaseNode[QuestionState]):
    async def run(self, ctx: GraphRunContext[QuestionState]) -> None:
        print("✅ Correct! Great job!")
        return None  # End the flow

class Reprimand(BaseNode[QuestionState]):
    async def run(self, ctx: GraphRunContext[QuestionState]) -> "Ask":
        print("❌ Incorrect. Try again!")
        return Ask()
```

### Step 4: Define the Graph
```python
from pydantic_graph import Graph

question_graph = Graph(
    nodes=(Ask, Answer, Evaluate, Congratulate, Reprimand),
    state_type=QuestionState
)
```

### Step 5: Run the Graph
```python
state = QuestionState()
await question_graph.run(Ask(), state=state)
```

---

## Best Practices for Writing Graphs

- **Keep Nodes Modular**: Each node should perform one action.
- **Use Pydantic Models**: Define structured outputs for LLMs.
- **Leverage State**: Store necessary information for multi-step interactions.
- **Type Annotations Matter**: Use explicit return types to define flow.

## Advanced Use Cases

### 1. Multi-Agent Collaboration
- A node can delegate tasks to different agents (e.g., ResearchAgent → SummarizeAgent).

### 2. External API Calls in Nodes
```python
@support_agent.tool
async def fetch_account_balance(ctx, customer_id: int) -> str:
    return f"Balance: ${123.45}"
```

### 3. Conditional Paths
```python
if ctx.state.some_flag:
    return NodeA()
else:
    return NodeB()
```

---

## Summary

This guide covers:
✅ How Pydantic-AI graphs work  
✅ How to structure nodes and transitions  
✅ A real-world example of a Q&A loop  
✅ Best practices for building scalable workflows  

By following these steps, you can **build robust AI-driven applications** using Pydantic-AI graphs efficiently!
