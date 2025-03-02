from dataclasses import dataclass, field
from typing import Dict, List, Union

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.usage import Usage
from pydantic_graph import BaseNode, Graph, GraphRunContext, End

from ..models import (
    ContextSummaries,
    ExhibitsResearchItem,
    ExhibitsResearchNotNeeded,
    ExhibitsSelection,
    ExhibitsResearchResult,
)


ISSUE_FINDER_SYSTEM_PROMPT = """\
You are a world class legal AI assistant. Users will provide a legal discovery document. \
Your task is to determine if knowledge gaps/issues exist and if so generate research items that can be addressed using the case exhibits.

# Please follow these instructions:
1. **Review the document carefully** to identify and understand each issue.
2. **Identify key issues** that depend on the exhibits.
3. **Ignore issues related to objections or refusals** and focus solely on what can be addressed using the exhibits that were submitted.
4. Do not generate issues related to documents we are **not in possession of**.
5. Consider references to 'See Attached' or 'See Exhibits' as research items that need to be addressed.
6. Note that research items may not be needed if the document stands on its own.

# IMPORTANT NOTE: Excerpts should include the item, its number, and the response whenever possible.
"""

EXHIBITS_SELECTION_PROMPT = """\
You are a world class legal AI assistant. Users will provide a question that needs to be addressed using discovery exhibits. \
Your task is to select the **Exhibit Number(s)** to pull from case records so that we can address the question. \
If none of the exhibits are relevant to the question simply choose the closest match.
"""

QUESTION_ANSWER_PROMPT = """\
You are a world class legal AI assistant. Users will provide an excerpt from a discovery document \
and related exhibit(s). \
Using only the exhibits and without prior knowledge, address the item or items contained in the excerpt. \
If the item is not found in the exhibits, briefly explain what is found by summarizing the exhibit(s). 

# Please follow these instructions:
1. Focus on details and accuracy.
2. Your answer should be fact focused and information dense.
3. Do not include any opinions or general statements. Facts only.
4. When multiple exhibits are provided maintain clear separation in your response by referencing the file name.
"""


# Define the state to be passed between nodes
@dataclass
class ExhibitsResearchState:
    """
    State object for the Exhibits Research Graph.

    This state is passed between nodes to maintain context throughout the graph execution.

    Attributes:
        research_items: Items identified for research.
        current_item_index: Index of the current item being processed.
        results: Dictionary of results for each research item.
        usages: List of usage statistics from all agent calls.
        final_result: Final result of the research.
    """

    research_items: List[ExhibitsResearchItem] = field(default_factory=list)
    current_item_index: int = 0
    results: Dict[int, str] = field(default_factory=dict)
    usages: List[Usage] = field(default_factory=list)
    final_result: str = ""


# Define the dependencies
@dataclass
class ExhibitsDeps:
    """
    Dependencies for the Exhibits Research Graph.

    This class holds agents and other resources that will be injected into the graph.

    Attributes:
        primary_document: The primary document to analyze.
        context_summaries: Collection of exhibit summaries to search through.
        issue_finder_agent: Agent to find research items.
        exhibit_selector_agent: Agent to select relevant exhibits.
        question_answer_agent: Agent to answer questions based on exhibits.
        max_research_tasks: Maximum number of research tasks to perform.
    """

    primary_document: str
    context_summaries: ContextSummaries
    issue_finder_agent: Agent
    exhibit_selector_agent: Agent
    question_answer_agent: Agent
    max_research_tasks: int = 10


# Create agents with proper types
issue_finder_agent = Agent[
    ExhibitsDeps, Union[List[ExhibitsResearchItem], ExhibitsResearchNotNeeded]
](
    model="openai:gpt-4o",
    result_type=Union[List[ExhibitsResearchItem], ExhibitsResearchNotNeeded],
    deps_type=ExhibitsDeps,
    retries=5,
    system_prompt=ISSUE_FINDER_SYSTEM_PROMPT,
)


@issue_finder_agent.system_prompt
def add_issue_limit(ctx: RunContext[ExhibitsDeps]) -> str:
    """Add issue limit to system prompt based on context."""
    return f"Please limit your response to no more than **{str(ctx.deps.max_research_tasks)} items**."


# Exhibit Selector Agent
exhibit_selector_agent = Agent[ExhibitsDeps, ExhibitsSelection](
    model="openai:gpt-4o",
    result_type=ExhibitsSelection,
    deps_type=ExhibitsDeps,
    retries=5,
    system_prompt=EXHIBITS_SELECTION_PROMPT,
)


@exhibit_selector_agent.system_prompt
def add_available_exhibits(ctx: RunContext[ExhibitsDeps]) -> str:
    """Add available exhibits to system prompt based on context."""
    return f"Here are the available exhibits to choose from:\n\n{ctx.deps.context_summaries.docs_info_string(number=True)}"


@exhibit_selector_agent.result_validator
async def validate_result(
    ctx: RunContext[ExhibitsDeps],
    result: ExhibitsSelection,
) -> ExhibitsSelection:
    """
    Validate the result to ensure the selected exhibit number(s) exist.

    Args:
        ctx: Run context.
        result: The exhibits selection result.

    Returns:
        The validated exhibits selection result.

    Raises:
        ModelRetry: If an invalid exhibit number is selected.
    """
    valid_file_numbers = [
        c.exhibit_number for c in ctx.deps.context_summaries.summaries
    ]

    for idx in result.exhibit_numbers:
        if idx not in valid_file_numbers:
            raise ModelRetry(
                f"Invalid exhibit number: {idx}. Please select from the following: {valid_file_numbers}"
            )
    return result


# Question Answer Agent
question_answer_agent = Agent[None, str](
    model="openai:gpt-4o",
    result_type=str,
    system_prompt=QUESTION_ANSWER_PROMPT,
    retries=2,
)


# Define nodes for the graph
class Initialize(BaseNode[ExhibitsResearchState, ExhibitsDeps]):
    """
    Initial node in the Exhibits Research Graph.

    This node sets up the state and transitions to the FindIssues node.
    """

    async def run(
        self, ctx: GraphRunContext[ExhibitsResearchState, ExhibitsDeps]
    ) -> "FindIssues":
        """
        Initialize the research process.

        Args:
            ctx: Graph run context with state and dependencies.

        Returns:
            FindIssues node to find research items.
        """
        return FindIssues()


class FindIssues(BaseNode[ExhibitsResearchState, ExhibitsDeps]):
    """
    Node to find research items in the primary document.

    This node uses the issue_finder_agent to identify research items.
    """

    async def run(
        self, ctx: GraphRunContext[ExhibitsResearchState, ExhibitsDeps]
    ) -> Union["ProcessItems", "NoResearchNeeded"]:
        """
        Find research items in the primary document.

        Args:
            ctx: Graph run context with state and dependencies.

        Returns:
            ProcessItems node if research items are found, otherwise NoResearchNeeded node.
        """

        # Run the agent
        result = await ctx.deps.issue_finder_agent.run(
            ctx.deps.primary_document, deps=ctx.deps
        )
        ctx.state.usages.append(result.usage())

        # Handle the result
        if isinstance(result.data, ExhibitsResearchNotNeeded):
            return NoResearchNeeded()

        # Store research items and move to processing
        ctx.state.research_items = result.data
        return ProcessItems()


class NoResearchNeeded(BaseNode[ExhibitsResearchState, ExhibitsDeps]):
    """
    Node executed when no research is needed.

    This node is reached when the issue_finder_agent determines that
    no research items are present in the document.
    """

    async def run(
        self, ctx: GraphRunContext[ExhibitsResearchState, ExhibitsDeps]
    ) -> "Finalize":
        """
        Handle the case where no research is needed.

        Args:
            ctx: Graph run context with state and dependencies.

        Returns:
            Finalize node to generate the final result.
        """
        return Finalize()


class ProcessItems(BaseNode[ExhibitsResearchState, ExhibitsDeps]):
    """
    Node to process research items.

    This node manages the iteration through research items and
    transitions to ProcessCurrentItem to process each item.
    """

    async def run(
        self, ctx: GraphRunContext[ExhibitsResearchState, ExhibitsDeps]
    ) -> Union["ProcessCurrentItem", "Finalize"]:
        """
        Process research items by dispatching to the appropriate next node.

        Args:
            ctx: Graph run context with state and dependencies.

        Returns:
            ProcessCurrentItem node if there are more items to process,
            otherwise Finalize node.
        """
        # Check if we have more items to process
        if ctx.state.current_item_index < len(ctx.state.research_items):
            return ProcessCurrentItem()

        # If all items have been processed, move to finalization
        return Finalize()


class ProcessCurrentItem(BaseNode[ExhibitsResearchState, ExhibitsDeps]):
    """
    Node to process the current research item.

    This node uses the exhibit_selector_agent to select relevant exhibits
    for the current research item.
    """

    async def run(
        self, ctx: GraphRunContext[ExhibitsResearchState, ExhibitsDeps]
    ) -> "AnswerQuestion":
        """
        Process the current research item by selecting relevant exhibits.

        Args:
            ctx: Graph run context with state and dependencies.

        Returns:
            AnswerQuestion node to answer the question.
        """
        # Get the current item
        current_item = ctx.state.research_items[ctx.state.current_item_index]

        # Run the agent
        selection_result = await ctx.deps.exhibit_selector_agent.run(
            current_item.question, deps=ctx.deps
        )
        ctx.state.usages.append(selection_result.usage())

        # Store the exhibit numbers in state for the next node
        ctx.state.selected_exhibits = selection_result.data.exhibit_numbers

        return AnswerQuestion()


class AnswerQuestion(BaseNode[ExhibitsResearchState, ExhibitsDeps]):
    """
    Node to answer the research question.

    This node uses the question_answer_agent to answer the question
    based on the selected exhibits.
    """

    async def run(
        self, ctx: GraphRunContext[ExhibitsResearchState, ExhibitsDeps]
    ) -> "RecordResult":
        """
        Answer the research question using the selected exhibits.

        Args:
            ctx: Graph run context with state and dependencies.

        Returns:
            RecordResult node to record the result.
        """
        # Get the current item
        current_item = ctx.state.research_items[ctx.state.current_item_index]

        # Retrieve exhibit texts
        exhibits_text = "\n\n".join(
            ctx.deps.context_summaries.get_exhibit_string(num)
            for num in ctx.state.selected_exhibits
        )

        # Prepare input for the agent
        qa_input = (
            f"Discovery excerpt: {current_item.excerpt}\n\nExhibits:\n\n{exhibits_text}"
        )

        # Run the agent
        answer_result = await ctx.deps.question_answer_agent.run(qa_input)
        ctx.state.usages.append(answer_result.usage())

        # Store the answer in state for the next node
        ctx.state.current_answer = answer_result.data

        return RecordResult()


class RecordResult(BaseNode[ExhibitsResearchState, ExhibitsDeps]):
    """
    Node to record the result of the current research item.

    This node stores the result and moves to the next item.
    """

    async def run(
        self, ctx: GraphRunContext[ExhibitsResearchState, ExhibitsDeps]
    ) -> "ProcessItems":
        """
        Record the result and move to the next item.

        Args:
            ctx: Graph run context with state and dependencies.

        Returns:
            ProcessItems node to process the next item.
        """
        # Get the current item
        current_item = ctx.state.research_items[ctx.state.current_item_index]

        # Format the result
        result = f"**Discovery Item:**\n{current_item.excerpt}\n\n**Answer:**\n{ctx.state.current_answer}\n---\n"

        # Store the result
        ctx.state.results[ctx.state.current_item_index] = result

        # Move to the next item
        ctx.state.current_item_index += 1

        # Delete temporary state variables
        if hasattr(ctx.state, "selected_exhibits"):
            delattr(ctx.state, "selected_exhibits")
        if hasattr(ctx.state, "current_answer"):
            delattr(ctx.state, "current_answer")

        # Return to ProcessItems to check for more items
        return ProcessItems()


class Finalize(BaseNode[ExhibitsResearchState, ExhibitsDeps]):
    """
    Node to finalize the research process.

    This node combines all results and completes the graph execution.
    """

    async def run(
        self, ctx: GraphRunContext[ExhibitsResearchState, ExhibitsDeps]
    ) -> End:
        """
        Finalize the research process by combining all results.

        Args:
            ctx: Graph run context with state and dependencies.

        Returns:
            None to end the graph execution.
        """
        # If there are no research items, return a special message
        if not ctx.state.research_items:
            ctx.state.final_result = "No research needed."
            return None

        # Combine all results
        result_strings = [
            ctx.state.results[i] for i in range(len(ctx.state.research_items))
        ]
        ctx.state.final_result = "\n".join(result_strings)

        return End()


# Create the graph
exhibits_research_graph = Graph(
    nodes=[
        Initialize,
        FindIssues,
        NoResearchNeeded,
        ProcessItems,
        ProcessCurrentItem,
        AnswerQuestion,
        RecordResult,
        Finalize,
    ],
    state_type=ExhibitsResearchState,
)
exhibits_research_graph.mermaid_save(
    "exhibits_research_graph.png",
    start_node=Initialize,
    direction="TB",
    highlighted_nodes=[RecordResult],
)


# Function to run the graph
async def perform_exhibits_research(
    primary_doc: str, context_summaries: ContextSummaries, max_research_tasks: int = 10
) -> ExhibitsResearchResult:
    """
    Performs research on exhibits using a graph-based approach.

    This function executes the exhibits research graph to process a primary document
    and find relevant information in the provided exhibits.

    Args:
        primary_doc: The primary document to analyze.
        context_summaries: Collection of exhibit summaries to search through.
        max_research_tasks: Maximum number of research tasks to perform.

    Returns:
        ExhibitsResearchResult containing the research findings and usage data.

    Example:
        ```python
        result = await perform_exhibits_research(
            primary_doc="Plaintiff's Response to Interrogatories...",
            context_summaries=context_summaries
        )
        print(result.result_string)
        ```
    """
    # Initialize state
    state = ExhibitsResearchState()

    # Initialize dependencies
    deps = ExhibitsDeps(
        primary_document=primary_doc,
        context_summaries=context_summaries,
        issue_finder_agent=issue_finder_agent,
        exhibit_selector_agent=exhibit_selector_agent,
        question_answer_agent=question_answer_agent,
        max_research_tasks=max_research_tasks,
    )

    # Run the graph
    await exhibits_research_graph.run(Initialize(), state=state, deps=deps)

    # Return the result
    return ExhibitsResearchResult(
        result_string=state.final_result,
        usages=state.usages,
    )


# Synchronous wrapper
def run_exhibits_research(
    primary_doc: str, context_summaries: ContextSummaries, max_research_tasks: int = 10
) -> ExhibitsResearchResult:
    """
    Synchronous wrapper for perform_exhibits_research.

    Args:
        primary_doc: The primary document to analyze.
        context_summaries: Collection of exhibit summaries to search through.
        max_research_tasks: Maximum number of research tasks to perform.

    Returns:
        ExhibitsResearchResult containing the research findings and usage data.
    """
    import asyncio

    return asyncio.run(
        perform_exhibits_research(primary_doc, context_summaries, max_research_tasks)
    )
