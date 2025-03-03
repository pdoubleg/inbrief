from pydantic_ai import Agent
from src.models import ContextSummaries, DocumentsProducedResult


# Define the prompt (keeping the original from DocumentsProduced)
DOCUMENTS_PRODUCED_PROMPT = """\
You are a world class legal assistant AI. Use the discovery documents to determine what was produced. You maintain the highest level of objectivity and never \
use subjective adjectives. For example, you would never say "comprehensive billing summary" and instead concisely summarize the actual billing items.

Please follow these instructions:

1. **Review the document carefully** to understand the context and scope of each discovery document.
2. **Simply state** each documents full file name, and a VERY brief description of the document.
3. **Make every word count** and don't add redundant description if the name says it already.

**Important Notes:**
* Never use markdown, bullet points, or lists, just plain text separated by newlines, ie '\\n'.
* Always include the full file name from the h1 headers.

[[## EXAMPLE OUTPUT ##]]

smith_johnson_rogs.pdf: Plaintiff's first set of interrogatories.\\n
Records and Bills.pdf: Medical records from Happy Spine Institute and New Wave Yoga.

[[## END OF EXAMPLE OUTPUT ##]]
"""

# Initialize the agent using pydantic-ai
documents_produced_agent = Agent[str, str](
    model="openai:gpt-4o",
    result_type=str,
    retries=3,
    system_prompt=DOCUMENTS_PRODUCED_PROMPT,
)


async def generate_documents_produced_summary(
    discovery_documents: str,
) -> DocumentsProducedResult:
    """
    Generate a summary of documents produced from discovery documents.

    Args:
        discovery_documents: String containing information about discovery documents

    Returns:
        A tuple containing the documents produced summary and usage information
    """
    result = await documents_produced_agent.run(discovery_documents)
    return DocumentsProducedResult(summary=result.data, usages=result.usage())


def run_documents_produced_report(
    input_context: ContextSummaries,
) -> DocumentsProducedResult:
    """
    Synchronous wrapper to generate a documents produced report.

    Args:
        input_context: ContextSummaries object containing summaries of discovery documents

    Returns:
        String summary of documents produced
    """
    import asyncio

    docs_info = input_context.docs_info_string(number=False)
    return asyncio.run(generate_documents_produced_summary(docs_info))
