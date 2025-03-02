import asyncio
from typing import List, Tuple, Union
from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.usage import Usage


from models import (
    ContextSummaries,
    ExhibitsResearchItem,
    ExhibitsResearchNotNeeded,
    ExhibitsSelection,
    ExhibitsResearchResult,
)


# System prompts for agents
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


@dataclass
class ResearchDeps:
    """Dependencies for research agents."""

    primary_document: str
    context_summaries: ContextSummaries
    max_research_tasks: int


# Issue Finder Agent
issue_finder_agent = Agent[
    ResearchDeps, Union[List[ExhibitsResearchItem], ExhibitsResearchNotNeeded]
](
    model="openai:gpt-4o",
    result_type=Union[List[ExhibitsResearchItem], ExhibitsResearchNotNeeded],
    deps_type=ResearchDeps,
    retries=5,
    system_prompt=ISSUE_FINDER_SYSTEM_PROMPT,
)


@issue_finder_agent.system_prompt
def add_issue_limit(ctx: RunContext[ResearchDeps]) -> str:
    """Add issue limit to system prompt based on context."""
    return f"Please limit your response to no more than **{str(ctx.deps.max_research_tasks)} items**."


# Exhibit Selector Agent
exhibit_selector_agent = Agent[ResearchDeps, ExhibitsSelection](
    model="openai:gpt-4o",
    result_type=ExhibitsSelection,
    deps_type=ResearchDeps,
    retries=5,
    system_prompt=EXHIBITS_SELECTION_PROMPT,
)


@exhibit_selector_agent.system_prompt
def add_available_exhibits(ctx: RunContext[ResearchDeps]) -> str:
    """Add available exhibits to system prompt based on context."""
    return f"Here are the available exhibits to choose from:\n\n{ctx.deps.context_summaries.docs_info_string(number=True)}"


@exhibit_selector_agent.result_validator
async def validate_result(
    ctx: RunContext[ResearchDeps], result: ExhibitsSelection
) -> ExhibitsSelection:
    """
    Validate the result to ensure the selected exhibit number(s) exist.

    Args:
        ctx: Run context with dependencies.
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


async def perform_exhibits_research(
    primary_doc: str, context_summaries: ContextSummaries
) -> ExhibitsResearchResult:
    """
    Performs research on exhibits asynchronously by processing multiple research items in parallel.

    Args:
        primary_doc: The primary document to analyze
        context_summaries: Collection of exhibit summaries to search through

    Returns:
        ExhibitsResearchResult containing the research findings and usage data
    """
    usages = []

    deps = ResearchDeps(
        primary_document=primary_doc,
        context_summaries=context_summaries,
        max_research_tasks=10,
    )

    # Step 1: Issue Finder
    issue_result = await issue_finder_agent.run(primary_doc, deps=deps)
    usages.append(issue_result.usage())
    if isinstance(issue_result.data, ExhibitsResearchNotNeeded):
        return "No research needed."

    research_items = issue_result.data

    # Step 2: Process all research items concurrently
    async def process_research_item(
        item: ExhibitsResearchItem,
    ) -> Tuple[str, List[Usage]]:
        item_usages = []

        # Exhibit Selector
        selection_result = await exhibit_selector_agent.run(item.question, deps=deps)
        item_usages.append(selection_result.usage())

        # Retrieve exhibit texts
        exhibits_text = "\n\n".join(
            context_summaries.get_exhibit_string(num)
            for num in selection_result.data.exhibit_numbers
        )

        # Question Answer
        qa_input = f"Discovery excerpt: {item.excerpt}\n\nExhibits:\n\n{exhibits_text}"
        answer_result = await question_answer_agent.run(qa_input)
        item_usages.append(answer_result.usage())

        result = f"**Discovery Item:**\n{item.excerpt}\n\n**Answer:**\n{answer_result.data}\n---\n"
        return result, item_usages

    # Create tasks for all research items
    tasks = [process_research_item(item) for item in research_items]

    # Process all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine results and usages
    output_strings = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Error processing research item: {result}")
            continue
        result_string, item_usages = result
        output_strings.append(result_string)
        usages.extend(item_usages)

    result_string = "\n".join(output_strings)
    return ExhibitsResearchResult(
        result_string=result_string,
        usages=usages,
    )


# Synchronous wrapper
def run_exhibits_research(
    primary_doc: str, context_summaries: ContextSummaries
) -> ExhibitsResearchResult:
    return asyncio.run(perform_exhibits_research(primary_doc, context_summaries))
