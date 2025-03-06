import asyncio
import time

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from src.models import ShortVersionResult


def get_short_version_prompt(long_version: str) -> str:
    return f"""\
Your goal is to produce a summary report suitable for a Senior Attorney that is more concise than the draft while retaining all relevant information. \
Aim to weave together insights from the draft sections to form a cohesive narrative. You maintain the highest level of objectivity and never \
use subjective adjectives. For example, you would never say "comprehensive billing summary" and instead should concisely summarize the actual billing items.

# Instructions:
- You will retain all details from the draft while being more concise.
- Information on damages, monetary or otherwise should always be included.
- When the draft contains multiple DISCOVERY SUMMARY consolidate them into a single cohesive narrative summary.
- **Never** include generalities, opinions or overall thoughts.
- Ensure the summary is clear, concise and free of errors.
- **Never use markdown** formatting. Only plain text using newlines to denote breaks and sections.
- Always favor sentence-like structure over bullet points or lists to mimic a professional legal report.
- Focus on **aggregating and consolidating details** from the draft into a polished final version.
- Include counts, sums, and date ranges to express details. For example, when summarizing medical records, use total number of \
treatments for the full date range of treatment.

# PII Rules:
* NEVER include a person's DOB, full address, or phone number.
* If possible, include the plaintiff's age as of {time.strftime("%B %d, %Y", time.localtime())}.
* Include city and state of residence when possible. NEVER include the full address for people.

# FORMAT STRUCTURE GUIDELINES:
* Brief summary of information from draft HIGH LEVEL SUMMARY(s).
* Separate paragraph for brief summary of Request for Admissions draft (only if included in the draft).
* Separate paragraph for brief summary of the draft SUMMARY OF PLAINTIFF'S MEDICAL RECORDS (only if included in the draft).
* List-like summary of the submitted documents, one or two sentences only (only if included in the draft).
* List of providers with addresses, dates and phone numbers, one provider per line with comma separated values.
* If something is not included in the draft simply omit it from your summary report without mention.

## EXAMPLE STRUCTURE TO FOLLOW:

CASE CAPTION: DAVID BILYY V. CHRISTOPHER KENNEDY, ET AL

CLAIM NUMBER: 24-2-16107-4 KNT

---

FIELD LEGAL - DISCOVERY SUMMARY

PARTY RESPONDING / TITLE OF RESPONSE: your response here

HIGH LEVEL SUMMARY: A brief, 1 to 3 paragraph well-written narrative summary that consolidates and distills the draft HIGH LEVEL SUMMARY(s).

Separate paragraph for a brief summary of Requests for Admissions if applicable.

Separate paragraph for a brief SUMMARY OF PLAINTIFF MEDICAL RECORDS if applicable.

A separate brief narrative paragraph summary of the submitted documents if applicable, one or two sentences only.

A separate listing of providers with addresses, dates and phone numbers using one line per provider for example: SeaMar Clinic Mental Health Provider 10217 125th St Ct E, 2nd Floor, Puyallup, WA 98374, Phone: 253-848-5951.

Here is the draft report:\n\n{long_version}
"""


def get_short_version_exhibits_prompt(draft_report: str) -> str:
    return f"""\
Your goal is to produce a summary report suitable for a Senior Attorney that is more concise than the draft while retaining all relevant information. Aim to weave together insights from the draft sections to form a cohesive narrative. You maintain the highest level of objectivity and never use subjective adjectives. For example, you would never say "comprehensive billing summary" and instead concisely summarize the actual billing items.

# Instructions:
* You will retain all details from the draft while being more concise.
* **Never** include generalities, opinions or overall thoughts.
* Ensure the summary is clear, concise and free of errors.
* **Never use markdown** formatting. Only plain text using newlines to denote breaks and sections.
* Always favor sentence-like structure over bullet points or lists to mimic a professional legal report.
* Focus on **aggregating and consolidating details** from the draft into a polished final version.
* Include counts, sums, and date ranges to express details. For example, when summarizing medical records, use total number of treatments for the full date range of treatment.

# PII Rules:
* NEVER include a person's DOB, full address, or phone number.
* If possible, include the plaintiff's age as of {time.strftime("%B %d, %Y", time.localtime())}.
* Include city and state of residence when possible. NEVER include the full address for people.

# FORMAT STRUCTURE GUIDELINES:
* Concise summary of information from draft(s).
* List-like summary of the documents, one or two sentences only.
* List of **distinct** providers with addresses, dates and phone numbers, one provider per line with comma separated values.

Here is the draft report:\n\n{draft_report}
"""


reasoning_model = OpenAIModel(
    model_name="o1-mini",
    system_prompt_role="user",
)

short_version_agent = Agent[str, str](
    model=reasoning_model,
    deps_type=str,
    result_type=str,
    retries=3,
    system_prompt="You are a world class legal assistant AI.",
)


@short_version_agent.system_prompt
async def add_instructions(ctx: RunContext[str]) -> str:  # noqa: F811
    return get_short_version_prompt(long_version=ctx.deps)


short_version_exhibits_agent = Agent[str, str](
    model=reasoning_model,
    deps_type=str,
    result_type=str,
    retries=3,
    system_prompt="You are a world class legal assistant AI.",
)


@short_version_exhibits_agent.system_prompt
async def add_instructions(ctx: RunContext[str]) -> str:  # noqa: F811
    return get_short_version_exhibits_prompt(draft_report=ctx.deps)


def run_short_version(long_version: str) -> ShortVersionResult:
    result = asyncio.run(short_version_agent.run(user_prompt=None, deps=long_version))
    return ShortVersionResult(summary=result.data, usages=result.usage())


def run_short_version_exhibits(draft_report: str) -> ShortVersionResult:
    result = asyncio.run(short_version_exhibits_agent.run(user_prompt=None, deps=draft_report))
    return ShortVersionResult(summary=result.data, usages=result.usage())
