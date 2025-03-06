import time
import asyncio
from typing import List, Tuple
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.usage import Usage
from src.models import (
    ConversionResult,
    ContextSummary,
    ContextSummaries,
    TitleAndDescriptionResult,
)
from src.utils import prepare_processed_document_chunks, count_tokens


DOCUMENT_SUMMARY_PROMPT = f"""\
You are a world class legal assistant AI. Summarize the context, which is a packet of discovery documents for a legal matter.

# Please follow these instructions:

1. **Review the document carefully** to identify and understand the information relevant to litigation.
2. **Summarize the document contents** in a clear, detailed, and fact-focused manner.
3. **Retain key details** regarding the timeline of events, parties and/or service providers involved.
4. **Focus on** information that will help the attorney understand the case.
5. **On medical records** include the **diagnosis** the doctor listed from the visit.
6. **Do NOT include** any general statements or overall thoughts, or otherwise unnecessary text, simply consolidate the facts.
7. Whenever possible, favor **paragraph format** with over lists to ensure readability.

# PII Rules:

* NEVER include a person's DOB, physical address, or phone number.
* If possible, include the plaintiff's age as of {time.strftime("%B %d, %Y", time.localtime())}.
* Include city and state of residence when possible. NEVER include the full address for people.

# For medical related documents, include the following information:

* The Summary of medical record/bills
* Provider and date range
* Complaints made (by date)
* Any Objective testing (X-rays, CT Scans, MRIs, EMG studies)
* Diagnoses/Findings (by date)
* Treatment Plan (by date)
* Orders made (medication, testing, referral to other providers) (by date)
* Overall – note any gaps in time in treatment
* If bills submitted, $ amounts:  billed, adjustments, paid by insurance, write-offs, outstanding amount
* Any prior/subsequent accidents/injuries noted

# **Important Notes:**

* Include contact information for any parties and/or service providers when available.
* Consider all parties related to the Plaintiff even if not directly involved in the legal matter. Examples can include, but are not limited to employer and primary care physician.
* Always highlight the following items:
    * Prior and/or subsequent accidents
    * Prior and/or subsequent injuries
* **Medical Records** should always be grouped by provider.
* **Multiple documents** may be included. If so, maintain clear separation in your summary.
* Remember to include all information on **monetary amounts**, **diagnoses**, and **treatments**.

"""


INTERMEDIATE_SUMMARY_PROMPT = """\
You are a world class legal assistant AI. Generate a detailed summary of the excerpt from a larger packet of discovery documents.

Please follow these instructions:

1. **Review the document carefully** to identify and understand the information relevant to litigation.
2. **Summarize the document** in a clear, detailed, and fact-focused manner.
3. **Retain details** regarding the timeline of events, parties involved, and specific **service providers**.
4. **Focus on** information that will help the attorney understand the case.
5. **On medical records** include the **diagnosis** the doctor listed from the visit.
6. **Do NOT include** any general statements or overall thoughts, or otherwise unnecessary text, simply consolidate the facts.

# For medical related documents, include the following information:

* The Summary of medical record/bills
* Provider and date range
* Complaints made (by date)
* Any Objective testing (X-rays, CT Scans, MRIs, EMG studies)
* Diagnoses/Findings (by date)
* Treatment Plan (by date)
* Orders made (medication, testing, referral to other providers) (by date)
* Overall – note any gaps in time in treatment
* If bills submitted, $ amounts:  billed, adjustments, paid by insurance, write-offs, outstanding amount
* Any prior/subsequent accidents/injuries noted

**Important Notes:**

* Include contact information for any parties and/or service providers when available.
* Consider all parties related to the Plaintiff even if not directly involved in the legal matter. Examples can include, but are not limited to employer and primary care physician.
* Always highlight the following items:
    * Prior and/or subsequent accidents
    * Prior and/or subsequent injuries
* **Medical Records** should always be grouped by provider.
* **Multiple documents** may be included. If so, maintain clear separation in your summary.
* Remember to include all information on **monetary amounts**, **diagnoses**, and **treatments**.

"""


CONSOLIDATED_SUMMARY_PROMPT = """\
You are a world class legal assistant AI. Consolidate the context, which is a set of intermediate summaries for a large packet of discovery documents.

# Please follow these instructions:

1. **Review the intermediate summaries carefully** to identify and understand the information relevant to litigation.
2. **Consolidated and the document** in a clear, detailed, and fact-focused manner.
3. **Retain details** regarding the timeline of events, parties involved, and specific **service providers**.
4. **Focus on** information that will help the attorney understand the case.
5. **On medical records** include the **diagnosis** the doctor listed from the visit.
6. **Do NOT include** any general statements or overall thoughts, or otherwise unnecessary text, simply consolidate the facts.
7. Favor **paragraph format** to ensure readability.

# PII Rules:

* NEVER include a person's DOB, physical address, or phone number.
* If possible, include the plaintiff's age as of {time.strftime('%B %d, %Y', time.localtime())}.
* Include city and state of residence when possible. NEVER include the full address for people.

# For medical related documents, include the following information:

* The Summary of medical record/bills
* Provider and date range
* Complaints made (by date)
* Any Objective testing (X-rays, CT Scans, MRIs, EMG studies)
* Diagnoses/Findings (by date)
* Treatment Plan (by date)
* Orders made (medication, testing, referral to other providers) (by date)
* Overall – note any gaps in time in treatment
* If bills submitted, $ amounts:  billed, adjustments, paid by insurance, write-offs, outstanding amount
* Any prior/subsequent accidents/injuries noted

# **Important Notes:**

* Include contact information for any parties and/or service providers when available.
* Consider all parties related to the Plaintiff even if not directly involved in the legal matter. Examples can include, but are not limited to employer and primary care physician.
* Always highlight the following items:
    * Prior and/or subsequent accidents
    * Prior and/or subsequent injuries
* **Medical Records** should always be grouped by provider.
* **Multiple documents** may be included. If so, maintain clear separation in your summary.
* Remember to include all information on **monetary amounts**, **diagnoses**, and **treatments**.

"""


TITLE_DESCRIPTION_PROMPT = """\
    # Please follow these instructions:
    1. Titles should be concise, specific, and avoid filler words or general statements.
    2. Descriptions should be concise, fact-focused, and information dense.
    3. Make every word count!
    
"""


document_summary_agent = Agent[str, str](
    model="openai:gpt-4o",
    result_type=str,
    retries=3,
    system_prompt=DOCUMENT_SUMMARY_PROMPT,
)

intermediate_summary_agent = Agent[str, str](
    model="openai:gpt-4o",
    result_type=str,
    retries=3,
    system_prompt=INTERMEDIATE_SUMMARY_PROMPT,
)

consolidated_summary_agent = Agent[str, str](
    model="openai:gpt-4o",
    result_type=str,
    retries=3,
    system_prompt=CONSOLIDATED_SUMMARY_PROMPT,
)

title_description_agent = Agent[str, TitleAndDescriptionResult](
    model="openai:gpt-4o",
    result_type=TitleAndDescriptionResult,
    retries=3,
    system_prompt=TITLE_DESCRIPTION_PROMPT,
)

# Asynchronous processing functions


async def summarize_document(document: str) -> Tuple[str, Usage]:
    """Summarize a single discovery document."""
    result = await document_summary_agent.run(document)
    return result.data, result.usage()


async def summarize_intermediate_excerpt(excerpt: str) -> AgentRunResult:
    """Summarize an intermediate document excerpt."""
    result = await intermediate_summary_agent.run(excerpt)
    return result

async def generate_title_and_description(summary: str) -> Tuple[str, Usage]:
    """Get a title and description for a summary."""
    result = await title_description_agent.run(summary)
    return result.data, result.usage()


async def consolidate_summaries(
    intermediate_summaries: List[str],
    chunk_size: int = 20000,
) -> Tuple[str, List[Usage]]:
    """
    Consolidate multiple intermediate summaries into a single summary.

    Args:
        intermediate_summaries: List of intermediate summaries to consolidate.
        chunk_size: Maximum token size allowed per consolidation step.

    Returns:
        A tuple containing the consolidated summary and usage information.
    """
    usages = []

    # Join summaries with separators
    combined_summaries = "\n\n---\n\n".join(intermediate_summaries)

    # Check if the combined summaries are too large
    tokens = count_tokens(combined_summaries)

    if tokens > chunk_size:
        # If too large, do multi-step consolidation
        # Split summaries into smaller groups
        batch_size = max(
            2, len(intermediate_summaries) // 2
        )  # At least 2 summaries per batch
        batches = [
            intermediate_summaries[i : i + batch_size]
            for i in range(0, len(intermediate_summaries), batch_size)
        ]

        # First level consolidation
        first_level_results = []
        for batch in batches:
            batch_text = "\n\n---\n\n".join(batch)
            result = await consolidated_summary_agent.run(batch_text)
            first_level_results.append(result.data)
            usages.append(result.usage())

        # Second level consolidation
        combined_text = "\n\n---\n\n".join(first_level_results)
        result = await consolidated_summary_agent.run(combined_text)
        usages.append(result.usage())
        return result.data, usages
    else:
        # Single consolidation step
        result = await consolidated_summary_agent.run(combined_summaries)
        usages.append(result.usage())
        return result.data, usages


# Main summarization workflow


async def process_discovery_document(
    input_document: ConversionResult,
    chunk_size: int = 20000,
    add_labels: bool = False,
    cap_multiplier: float = 1.0,
) -> dict:
    """Process a discovery document asynchronously."""
    document_chunks = prepare_processed_document_chunks(
        [input_document], chunk_size, cap_multiplier
    )[0]

    usages_list: List[Usage] = []

    if document_chunks.token_count <= chunk_size:
        summary, usage = await summarize_document(document_chunks.processed_text)
        usages_list.append(usage)
    else:
        # Process chunks asynchronously
        tasks = [
            summarize_intermediate_excerpt(chunk.text)
            for chunk in document_chunks.text_chunks
        ]
        intermediate_summaries_responses = await asyncio.gather(*tasks)

        intermediate_summaries = [
            response.data for response in intermediate_summaries_responses
        ]
        intermediate_usages = [
            response.usage() for response in intermediate_summaries_responses
        ]
        usages_list.extend(intermediate_usages)

        summary, usages = await consolidate_summaries(
            intermediate_summaries, chunk_size
        )
        usages_list.extend(usages)

    # Optionally generate title and description
    title, description = "", ""
    if add_labels:
        result = await title_description_agent.run(summary)
        title, description = result.data.title, result.data.description
        usages_list.append(result.usage())

    return {
        "document_name": input_document.name,
        "document_title": title,
        "document_description": description,
        "summary": summary,
        "usages": usages_list,
    }


async def process_single_document(
    doc: ConversionResult,
    exhibit_number: int = 0,
    chunk_size: int = 20000,
    add_labels: bool = True,
    cap_multiplier: float = 1.0,
) -> ContextSummary:
    """
    Asynchronously process a single ConversionResult document into a ContextSummary.

    Args:
        doc: The ConversionResult document to process.
        exhibit_number: The exhibit number for this document.
        chunk_size: Maximum chunk size for processing.
        add_labels: Whether to generate title and description.
        cap_multiplier: Multiplier for chunk size cap.

    Returns:
        ContextSummary object containing the processed document summary.
    """
    try:
        result = await process_discovery_document(
            input_document=doc,
            chunk_size=chunk_size,
            add_labels=add_labels,
            cap_multiplier=cap_multiplier,
        )

        return ContextSummary(
            exhibit_number=exhibit_number,
            file_name=doc.name,
            summary=result["summary"],
            document_title=result["document_title"],
            document_description=result["document_description"],
            usages=result["usages"],
        )

    except Exception as e:
        # Log the exception and return a placeholder summary
        error_message = f"Error processing document: {doc.name}"
        print(f"{error_message}\nException: {e}")

        return ContextSummary(
            exhibit_number=exhibit_number,
            file_name=doc.name,
            summary=error_message,
            document_title="Unknown due to processing error",
            document_description="Unknown due to processing error",
            usages=[],
        )


async def process_documents_async(
    documents: List[ConversionResult],
    chunk_size: int = 20000,
    add_labels: bool = True,
    cap_multiplier: float = 1.0,
) -> ContextSummaries:
    """
    Asynchronously process multiple ConversionResult documents concurrently.

    Args:
        documents: List of ConversionResult documents to process.
        chunk_size: Maximum chunk size for processing.
        add_labels: Whether to generate title and description.
        cap_multiplier: Multiplier for chunk size cap.

    Returns:
        ContextSummaries object containing all processed document summaries.
    """
    tasks = []
    seen = set()

    for idx, doc in enumerate(documents, start=1):
        if doc.name not in seen:
            seen.add(doc.name)
            task = asyncio.create_task(
                process_single_document(
                    doc=doc,
                    exhibit_number=idx,
                    chunk_size=chunk_size,
                    add_labels=add_labels,
                    cap_multiplier=cap_multiplier,
                )
            )
            tasks.append(task)

    context_summaries = []
    for task in asyncio.as_completed(tasks):
        summary = await task
        context_summaries.append(summary)

    return ContextSummaries(summaries=context_summaries)


def run_documents_summary(
    documents: List[ConversionResult],
    chunk_size: int = 20000,
    add_labels: bool = True,
    cap_multiplier: float = 1.0,
) -> ContextSummaries:
    """
    Synchronous wrapper to process multiple ConversionResult documents.

    Args:
        documents: List of ConversionResult documents to process.
        chunk_size: Maximum chunk size for processing.
        add_labels: Whether to generate title and description.
        cap_multiplier: Multiplier for chunk size cap.

    Returns:
        ContextSummaries object containing all processed document summaries.
    """
    return asyncio.run(
        process_documents_async(
            documents=documents,
            chunk_size=chunk_size,
            add_labels=add_labels,
            cap_multiplier=cap_multiplier,
        )
    )
