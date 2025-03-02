import asyncio
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.usage import Usage

from models import (
    ConversionResult,
    EntityListing,
    FinalizedEntityListing,
    ProviderListingResult,
    QCResult,
    ResolvedEntityListing,
    TextChunk,
)
from utils import prepare_processed_document_chunks

ENTITY_EXTRACTOR_PROMPT = """\
You are a world class legal assistant AI. Use the context to generate a list-like summary of ALL **medical providers** and \
**parties involved** in the case. Include detailed notes on services, amounts, and dates to support downstream consolidation.

# MEDICAL PROVIDERS INSTRUCTIONS:

1. **Extract the following information:**
    - **Provider Name**
    - **Dates**
    - **Services and/or Fees**
    - **Provider Address**
    - **Phone Number**
    - **Fax Number**
    - **Notes** (if applicable)

# PARTIES INVOLVED INSTRUCTIONS:

1. Include eyewitnesses, expert witnesses, and other parties with direct knowledge of the case, or directly associated to the plaintiff.
2. Include parties or entities tangentially related to the case, such as employers or primary care providers.
3. Ignore lawyers, law firms, judges, or other legal professionals.
4. **Extract the following information:**
    - **Party Name**
    - **Party Type**
    - **Contact Info**
    - **Notes** (if applicable)

# **IMPORTANT NOTE:** If any information is missing in the context **simply omit it** from your output. DO NOT include 'unknown', 'not provided', or similar language.

# FORMATTING INSTRUCTIONS:
* Do NOT use bullet points or numbered lists when generating your output.
* Follow closely the format provided in the example below.

[[## EXAMPLE OUTPUT ##]]

Medical Providers:

European Health Center P.S. Corp. (08/04/2021)
32812 Pacific Hwy South, Federal Way, WA 98003
Phone: (253) 874-5028
Fax: (253) 835-7224

Marc-Anthony Chiropractic Clinic (08/04/2021 to 09/29/2021)
32818 Pacific Hwy South, Federal Way, WA 98003
Phone: (253) 874-5028

Other Parties:

Pamela Johnson
Independent Witness to the accident
1234 Elm Street, Springfield, Ohio 45005
email: pj123@gmail.com
Notes: Witnessed the accident on 08/04/2021 while walking her dog.

Pizza Hut Restaurant
Plaintiff's place of employment
100 Main Street, Springfield, Ohio 45005
Notes: Plaintiff was employed here and out on delivery at the time of the accident.

[[## END OF EXAMPLE OUTPUT ##]]
"""

ENTITY_RESOLVER_PROMPT = """\
You are a world class legal assistant AI. Users will provide an initial listing of providers and parties/entities found in discovery documents. \
Your task is to generate a final comprehensive listing with all deduplicates resolved, and their values aggregated. For example, if a provider is listed multiple times, \
you will need to combine the monetary exposure and use the min and max dates for the range.

# INSTRUCTIONS:
1. **Review the listings carefully** to identify and understand each party's role.
2. Always **favor provider name** over individual physician name.

# PII Rules:
* NEVER include a person's DOB, full address, or phone number.
* Include city and state of residence when possible. NEVER include the full address for people.

# FORMATTING INSTRUCTIONS:
* Do NOT use bullet points or numbered lists when generating your output.
* Follow closely the format provided in the example below.

# **IMPORTANT NOTE:** If any information is missing **simply omit it** from your output. DO NOT include 'unknown', 'not provided', or similar language.

[[## EXAMPLE OUTPUT ##]]

Medical Providers:

European Health Center P.S. Corp.
32812 Pacific Hwy South, Federal Way, WA 98003
Phone: (253) 874-5028
Fax: (253) 835-7224

Marc-Anthony Chiropractic Clinic (08/04/2021 to 09/29/2021)
32818 Pacific Hwy South, Federal Way, WA 98003
Phone: (253) 874-5028

Other Parties:

Pamela Johnson
Independent Witness to the accident
Springfield, Ohio
email: pj123@gmail.com
Notes: Witnessed the accident on 08/04/2021 while walking her dog.

Pizza Hut Restaurant
Plaintiff's place of employment
100 Main Street, Springfield, Ohio 45005
Notes: Plaintiff was employed here and out on delivery at the time of the accident.

[[## END OF EXAMPLE OUTPUT ##]]
"""

ENTITY_FINALIZER_PROMPT = """\
You are a world class legal assistant AI. Users will provide listings of entities, one from a 'primary' document and optionally another from a 'supporting' document. \
Your task is to follow the **PII Rules** while **prioritizing entities** from the primary document over those in the supporting document in order to \
generate a final consolidated listing. Note that **prioritization** means you should always include providers and entities from the primary document, even \
if they are not in the supporting document. You should also **prioritize** the primary document's details over the supporting document. \
If you only receive one type of listing simply skip prioritization and focus on the following rules.

# INSTRUCTIONS:
1. **Always prioritize** providers or entities from the **primary document** over those in the supporting document.
2. If information conflicts, favor the primary document.
3. **Exclude all** attorneys, law firms, judges, or other legal professionals from your output.

# PII RULES:
* NEVER include a person's date of birth, full address, or phone number.
* Include city and state of residence when possible. NEVER include the full address for people.

## MEDICAL PROVIDERS INSTRUCTIONS:
1. On **Date Range**, please provide the **earliest and latest** dates of service if possible.
2. Always favor **provider name** over individual physician name.
3. **Include the following information:**
    - **Provider Name**
    - **Date Range**
    - **Provider Address**
    - **Phone Number**
    - **Fax Number**

## PARTIES INVOLVED INSTRUCTIONS:
1. Include eyewitnesses, expert witnesses, and other parties with direct knowledge of the case.
2. Include parties or entities tangentially related to the case, such as employers or primary care providers.
3. Ignore lawyers, law firms, judges, or other legal professionals.
4. **Include the following information:**
    - **Party Name**
    - **Party Type**
    - **Contact Info**
    - **Notes**

# FORMATTING INSTRUCTIONS:
* Do NOT use bullet points or numbered lists when generating your output.
* Follow closely the format provided in the example below.
* If information is missing in the listings simply omit it from your output without mention.

[[## EXAMPLE OUTPUT ##]]

Medical Providers:

European Health Center P.S. Corp.
32812 Pacific Hwy South, Federal Way, WA 98003
Phone: (253) 874-5028
Fax: (253) 835-7224
Notes: The address on the primary document differs from the supporting document.

Marc-Anthony Chiropractic Clinic (08/04/2021 to 09/29/2021)
32818 Pacific Hwy South, Federal Way, WA 98003
Phone: (253) 874-5028

Other Parties:

Pamela Johnson
Independent Witness to the accident
Springfield, Ohio
email: pj123@gmail.com
Notes: Multiple addresses were listed for this individual.

Pizza Hut Restaurant
Plaintiff's place of employment
100 Main Street, Springfield, Ohio 45005
Notes: Plaintiff was employed here and out on delivery at the time of the accident.

[[## END OF EXAMPLE OUTPUT ##]]

# **IMPORTANT NOTE:** If any information is missing **simply omit it** from your output. DO NOT include 'unknown', 'not provided', or similar language.
"""


QC_PROMPT = """\
You are a world class legal assistant AI performing a quality control review. Users will provide a finalized listing of medical providers and other parties/entities. \
Your task is to audit the entity listing and check adherence to the following quality control rules:

# PRIMARY RULES:
* Group by provider and the date range, or ranges of service.
* Only include multiple date ranges when a meaningful gap exists, otherwise use a single date range with min and max dates.
* If any information is missing **simply omit it** from your output. DO NOT include 'unknown', 'not provided', or similar language.

# PII RULES:
* NEVER include a person's date of birth, full address, or phone number.
* Include city and state of residence when possible.
* NEVER include attorneys, law firms, judges, or other legal professionals from your output.

## MEDICAL PROVIDERS RULES:
1. Always favor **provider name** over individual physician name.
2. Include the following information:
    - Provider Name
    - Date Range
    - Provider Address
    - Phone Number
    - Fax Number

## PARTIES INVOLVED RULES:
1. Include eyewitnesses, expert witnesses, and other parties with direct knowledge of the case.
2. Include parties or entities tangentially related to the case, such as employers or primary care providers.
3. NEVER include lawyers, law firms, judges, or other legal professionals.
4. Include the following information:
    - Party Name
    - Party Type
    - Contact Info
    - Notes (if needed)

# FORMATTING RULES:
* Do NOT use bullet points or numbered lists when generating your output.

"""


@dataclass
class ProviderDeps:
    """Dependencies for provider listing agents."""

    chunk_size: int
    primary_documents: Optional[List[ConversionResult]] = None
    supporting_documents: Optional[List[ConversionResult]] = None


# QC Agent
qc_agent = Agent[None, QCResult](
    model="openai:gpt-4o",
    result_type=QCResult,
    retries=5,
    system_prompt=QC_PROMPT,
)

# Entity Extractor Agent
entity_extractor_agent = Agent[None, EntityListing](
    model="openai:gpt-4o",
    result_type=EntityListing,
    retries=5,
    system_prompt=ENTITY_EXTRACTOR_PROMPT,
)


# Entity Resolver Agent
entity_resolver_agent = Agent[None, ResolvedEntityListing](
    model="openai:gpt-4o",
    result_type=ResolvedEntityListing,
    retries=5,
    system_prompt=ENTITY_RESOLVER_PROMPT,
)


# Entity Finalizer Agent
entity_finalizer_agent = Agent[ProviderDeps, FinalizedEntityListing](
    model="openai:gpt-4o",
    result_type=FinalizedEntityListing,
    deps_type=ProviderDeps,
    retries=5,
    system_prompt=ENTITY_FINALIZER_PROMPT,
)


@entity_finalizer_agent.result_validator
async def validate_result(
    ctx: RunContext[ProviderDeps], result: FinalizedEntityListing
) -> FinalizedEntityListing:
    qc_response = qc_agent.run(result.finalized_listing, usage=ctx.usage)
    if qc_response.data.score >= 8:
        return result
    else:
        raise ModelRetry(
            f"Please address the following QC findings and try again: {qc_response.data.feedback}"
        )


async def generate_provider_listings(
    primary_documents: Optional[List[ConversionResult]] = None,
    supporting_documents: Optional[List[ConversionResult]] = None,
    chunk_size: int = 20000,
) -> ProviderListingResult:
    """
    Generates a comprehensive listing of medical providers and other entities
    found in discovery documents asynchronously.

    Args:
        primary_documents: List of primary documents to analyze
        supporting_documents: List of supporting documents to analyze
        chunk_size: Size of each chunk in tokens

    Returns:
        ProviderListingResult containing the resolved listing and usage data

    Raises:
        ValueError: If neither primary_documents nor supporting_documents is provided
    """
    if primary_documents is None and supporting_documents is None:
        raise ValueError(
            "Either primary_documents or supporting_documents must be provided."
        )

    usages = []

    deps = ProviderDeps(
        chunk_size=chunk_size,
        primary_documents=primary_documents,
        supporting_documents=supporting_documents,
    )

    resolved_primary_entities = None
    resolved_supporting_entities = None

    # Process primary documents if provided
    if primary_documents is not None:
        primary_chunks = prepare_processed_document_chunks(
            primary_documents, chunk_size=chunk_size, cap_multiplier=1
        )

        # Process chunks in parallel
        async def process_chunk(chunk: TextChunk) -> Tuple[str, Usage]:
            entity_result = await entity_extractor_agent.run(chunk.text)
            return entity_result.data.entity_listing, entity_result.usage()

        # Create tasks for all chunks across all documents
        chunk_tasks = []
        for doc in primary_chunks:
            for chunk in doc.text_chunks:
                chunk_tasks.append(process_chunk(chunk))

        # Process all chunks concurrently
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        # Collect results and handle any exceptions
        entity_listings = []
        for result in chunk_results:
            if isinstance(result, Exception):
                print(f"Error processing chunk: {result}")
                continue
            entity_listing, usage = result
            entity_listings.append(entity_listing)
            usages.append(usage)

        # Resolve entities from primary documents
        if entity_listings:
            resolver_result = await entity_resolver_agent.run(
                "\n\n".join(entity_listings),
            )
            usages.append(resolver_result.usage())
            resolved_primary_entities = resolver_result.data.resolved_listings

    # Process supporting documents if provided
    if supporting_documents is not None:
        supporting_chunks = prepare_processed_document_chunks(
            supporting_documents, chunk_size=chunk_size, cap_multiplier=1
        )

        # Process chunks in parallel
        async def process_chunk(chunk: TextChunk) -> Tuple[str, Usage]:
            entity_result = await entity_extractor_agent.run(chunk.text)
            return entity_result.data.entity_listing, entity_result.usage()

        # Create tasks for all chunks across all documents
        chunk_tasks = []
        for doc in supporting_chunks:
            for chunk in doc.text_chunks:
                chunk_tasks.append(process_chunk(chunk))

        # Process all chunks concurrently
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        # Collect results and handle any exceptions
        entity_listings = []
        for result in chunk_results:
            if isinstance(result, Exception):
                print(f"Error processing chunk: {result}")
                continue
            entity_listing, usage = result
            entity_listings.append(entity_listing)
            usages.append(usage)

        # Resolve entities from supporting documents
        if entity_listings:
            resolver_result = await entity_resolver_agent.run(
                "\n\n".join(entity_listings),
            )
            usages.append(resolver_result.usage())
            resolved_supporting_entities = resolver_result.data.resolved_listings

    # Always run finalizer with available entities
    input_text = ""
    if resolved_primary_entities is not None:
        input_text += f"Primary Document Entities:\n{resolved_primary_entities}\n\n"
    else:
        input_text += "Primary Document Entities: None provided\n\n"

    if resolved_supporting_entities is not None:
        input_text += f"Supporting Document Entities:\n{resolved_supporting_entities}"
    else:
        input_text += "Supporting Document Entities: None provided"

    # Run finalizer even if only one type of document is present
    if (
        resolved_primary_entities is not None
        or resolved_supporting_entities is not None
    ):
        finalizer_result = await entity_finalizer_agent.run(
            input_text,
            deps=deps,
        )
        usages.append(finalizer_result.usage())
        final_listing = finalizer_result.data.finalized_listing
    else:
        final_listing = "No entities found in the provided documents."

    return ProviderListingResult(
        resolved_listing=final_listing,
        usages=usages,
    )


# Synchronous wrapper
def run_provider_listings(
    primary_documents: Optional[List[ConversionResult]] = None,
    supporting_documents: Optional[List[ConversionResult]] = None,
    chunk_size: int = 20000,
) -> ProviderListingResult:
    """
    Synchronous wrapper for generate_provider_listings.

    Args:
        primary_documents: List of primary documents to analyze
        supporting_documents: List of supporting documents to analyze
        chunk_size: Size of each chunk in tokens

    Returns:
        ProviderListingResult containing the resolved listing and usage data
    """
    return asyncio.run(
        generate_provider_listings(
            primary_documents=primary_documents,
            supporting_documents=supporting_documents,
            chunk_size=chunk_size,
        )
    )
