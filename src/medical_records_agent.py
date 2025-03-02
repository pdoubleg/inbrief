import asyncio
from typing import List, Tuple

from pydantic_ai import Agent
from pydantic_ai.usage import Usage
from pydantic_ai.models.openai import OpenAIModel

from models import (
    ConversionResult,
    MedicalRecordsSummaryResult,
    TitleAndDescriptionResult,
)
from utils import prepare_processed_document_chunks, count_tokens


# System prompts for agents
MEDICAL_RECORDS_SUMMARY_PROMPT = """\
# You are a world class legal assistant AI. Use the given medical records to generate a **narrative summary report** of Plaintiff's medical records \
grouped by **provider and full date range** of service. Follow closely the structure and format of the EXAMPLE OUTPUT provided below. \
You maintain the highest level of objectivity and never use subjective adjectives. For example, you would never say "significant injuries" \
and instead summarize the actual injuries. Focus on grouping by provider and date range for treatment that spans multiple records or documents. If you \
receive any intermediate summaries you should consolidate them by provider and date range.

# Please follow these important instructions carefully:
1. Use paragraph style summary of treatment history, not a chronology.
2. **Group information** by PROVIDER and the MIN and MAX DATES of treatments/services.
3. Include the **date range** of treatment(s), **not each date** of service.
4. Capture accurate counts for repeated visits, tests, or treatments.
5. IGNORE any text that is not related to medical records.
6. If you receive any intermediate summaries simply consolidate them with the new records.
7. IMPORTANT: Do not use bullet points or numbered lists in your output! Use double line breaks to separate sections.

# PII Rules:
* NEVER include a person's full DOB! Only the **year they were born**.
* Include **city and state** of residence when possible. NEVER include the full address for people.

# **Important Information to Include:**
* Provider and date range
* Complaints made (by date range)
* Any Objective testing (X-rays, CT Scans, MRIs, EMG studies)
* Diagnoses/Findings (by date range)
* Treatment Plan (by date range)
* Orders made (medication, testing, referral to other providers) (by date range)
* Overall – note any gaps in time in treatment
* If bills submitted, $ amounts:  billed, adjustments, paid by insurance, write-offs, outstanding amount
* Any prior/subsequent accidents/injuries noted

# **Important Service-Level Guidelines:**

## EMS/Ambulance at the Scene:
* Complaints
* If auto accident, note if airbags did/not go off?
* Complaint of headaches?  State they hit their head? Describe any loss of consciousness? Whiplash? Dizziness or vision issues? Nausea/vomiting?
* Anything to note on transport to hospital?
* If case is a slip/trip and fall, it may be important to note what Plaintiff reports about how accident/injury occurred.

## First Visit to Health Care Provider - ER, Urgent Care or Primary Care Provider:
* How arrive, ambulance or by personal car
* Same/next day or date gap
* Complaints: Back? Neck?
* Complaints of radiating pain (radicular pain or radiculopathy?)
* If they are claiming a concussion, did they complain of Headaches?  State they hit their head? Describe any loss of consciousness? Whiplash? Dizziness or vision issues? Nausea/vomiting?
* Height, weight and blood pressure
* Objective Exam (what is noted, what is denied)
* Testing done (X-rays, CT Scans etc.)
* Assessment and Plan
* Referrals (if any)
* Prescriptions
* Look to see if PCP and Pharmacy noted
* Look for any significant prior history or relevant medical conditions
* If case is a slip/trip and fall, it may be important to note what Plaintiff reports about how accident/injury occurred.

## Regular Chiropractic and/or Physical Therapy Treatment:
* Date range of treatment
* Changes in complaints over time.  Should see trend of improvement.  But also note if symptoms worsen over time.
* Any mention of new/subsequent injuries or re-injuries.
* No-shows to appointments, particularly if some frequency or gaps in attendance
* Any indication Plaintiff is/not compliant with Home Exercise Program
* Discharged and why?  (Goals met, reached maximum medical improvement, self-discharged because stopped coming to PT, treatment suspended due to surgical recommendation).
* IMPORTANT: Look for any significant **prior history** or **relevant medical conditions**

## Specialists
* Type of specialty
* Who referred Plaintiff
* Dates/Date range seen if a series of dates
* Initial Complaints and diagnoses
    - Complaints of radiating pain (radicular pain or radiculopathy?)
    - Complaint of tingling or numbness, loss of sensation or weakness in extremities?
    - Initial treatment recommendations
* Tests ordered/results
* Epidural Steroid Injections?
    - Level and side
* Surgery performed, or recommended?
* Prescriptions?
* Complaints at most recent appointment: What were complaints and how had changed over time?  What were provider's treatment recommendations at last appointment?
* IMPORTANT: Look for any significant **prior history** or **relevant medical conditions**

## MRIs – Cervical, Thoracic, Lumbar, Spine
* Date
* Do they note a comparison to a prior study?
* History (this is provided by the patient, sometimes revealing)
* Findings – Findings (IMPRESSIONS)

## EMGs
* Date
* Which extremities?
* History (this is provided by the patient, sometimes revealing)
* Findings – Copy/paste in the results

## Billing Summary:
* Summarize billing records if available

[[## EXAMPLE OUTPUT ##]]

SUMMARY OF PLAINTIFF'S MEDICAL RECORDS:

Northside Hospital Gwinnett (10/28/22 – 11/9/22)
Plaintiff presented at 11:30 p.m. to the E.R. via EMS reporting he was taking his trash outside after carving pumpkins with children when 2 \
male assailants shot him after demanding his wallet and cel phone. He was shot in left thigh and fell to the ground. No LOC or head trauma. \
He was hospitalized 10/28/22 – 11/1/22 for gunshot wound to the left thigh, multiple adjacent bullet fragments and surgical ORIF repair of displaced, \
comminuted (Type III) midshaft femur fracture. More specifically, he had a femoral intramedullary nail insertion with locking screws, via a hip approach. \
He was discharged weightbearing as tolerated, with a walker and instructions to follow up with orthopedic surgeon Jonathan Gillig MD. \
His PCP was noted to be Amimi S. Asayande MD. He complained of numbness in the left hand/fingers following surgery. \
He received PT prior to discharge. Billing for the hospitalization came to $93,585,00.

Resurgens Orthopaedics (10/29/22 – 2/21/23)
Billing and records for Dr. Jonathan Gillig. Dr. Gillig performed the ORIF surgery on 10/30/22. He was seen in follow up on 11/8/22 for a wound check. \
He had some knee stiffness and reduced left knee ROM but ambulated with a rolling walker. He returned on 11/22/22, ambulating with a rolling walker and \
using a cane at home. He was no longer taking narcotics. He was advised to continue weightbearing as tolerated. He had no activity restrictions and was \
asked to return un 4-6 weeks for re-check. Plaintiff returned on 1/3/23. He was walking with a cane and reported some knee stiffness. \
He was also noted to be walking with a crouched gait. Diagnostic studies showed good healing in his leg. He was given a physical therapy referral. \
Plaintiff was last seen in follow up on 2/21/23 reporting a main concern of PTSD symptoms. He was ambulating with a cane and reported no pain. \
Plaintiff reported waking up with nightmares and not feeling safe at home. He reported that he had an appt. with a psychiatrist in 2 weeks. \
Orthopedically, he was noted to be doing well and doing home exercises. His left knee ROM was 0-100 degrees. He was offered formal PT, which Plaintiff declined. \
He was told to return in 3 months for re-check and follow up x-rays. Total billing for Dr. Gillig (including surgery) was $11,193.00.

North Metropolitan Radiology Associates (10/29/22)
Physician billing only for X-rays (at Northside Hospital), $195.00

North Shore Shoulder (6/5/24 – 6/18/24)
Plaintiff was seen by Dr. Robert McLaughlin at this facility in Norwood MA on 6/5/24 for left knee pain and stiffness. Records indicate plaintiff reported \
undergoing "extensive course of physical therapy" after his injury on 10/28/22 and discharge from the hospital but has residual pain and stiffness. Plaintiff \
was unable to fully extend or bend knee. Unable to run, and had pain going up/downstairs, worse at end of the day. No recent PT. Exam of left knee showed slight \
effusion, normal strength, sensation and no tenderness. He had reduced ROM by 10 degrees on extension and flexion to 85 degrees. X-rays would be obtained and \
treatment options from conservative care to arthroscopic lysis of adhesions were discussed. Plaintiff returned via a telehealth visit on 6/18/24 and reported \
he had not started PT and that he would use his regular insurance to treat his knee. Total billing for both office visits was $2,300.00.

[[## END OF EXAMPLE OUTPUT ##]]
"""

TITLE_DESCRIPTION_PROMPT = """\
Generate a title and description of a text for downstream lookup.

# Please follow these instructions:
1. Titles should be concise, specific, and avoid filler words or general statements.
2. Descriptions should be concise, fact-focused, and information dense.
3. Make every word count!
"""

CONSOLIDATION_PROMPT = """\
# You are a world class legal assistant AI. Your task is to consolidate multiple medical record summaries into a single comprehensive summary.

Follow the same format and guidelines as for the original summaries, but ensure you:
1. Merge information from the same providers across different summaries
2. Maintain the date ranges accurately
3. Eliminate redundancies
4. Preserve all relevant medical details
5. Group by provider and date range
6. Use paragraph style, not chronological lists
7. Follow all the PII Rules

# PII Rules:
* NEVER include a person's full DOB! Only the **year they were born**.
* Include **city and state** of residence when possible. NEVER include the full address for people.

The consolidated summary should read as a single coherent document, not as a collection of separate summaries.
"""

FINALIZATION_PROMPT = """\
# You are a world class legal assistant AI. Your task is to finalize a summary of plaintiff medical records. Focus on \
grouping records together by provider and date range, or ranges of service. The goal is to align with how the billing was likely handled. \
For example, multiple doctors may be involved in a single treatment plan with all services being billed to a single provider. \
Your output should be a polished summary of plaintiff medical records that is free of errors and suitable for review by a Senior Attorney.

# Please follow these instructions closely:
1. Ensure the records are grouped by provider and date range such that no duplicate providers exist.
2. Include a date range for ongoing treatment. If meaningful gaps exist add a separate date range, e.g., "10/28/22 – 11/9/22" and "11/10/24 – 11/21/24".
3. Do not use bullet points or numbered lists in your output! Use double line breaks to separate sections.
4. Do not use subjective adjectives such as "significant", "severe", ext. and instead concisely summarize the actual injuries.
5. Use paragraph style, not chronological lists
6. Follow all the PII Rules

# PII Rules:
* NEVER include a person's full DOB! Only the **year they were born**.
* Include **city and state** of residence when possible. NEVER include the full address for people.

"""


# Medical Records Summary Agent
medical_records_summary_agent = Agent[None, str](
    model="openai:gpt-4o",
    result_type=str,
    retries=3,
    system_prompt=MEDICAL_RECORDS_SUMMARY_PROMPT,
)


# Consolidation Agent
consolidation_agent = Agent[None, str](
    model="openai:gpt-4o",
    result_type=str,
    retries=3,
    system_prompt=CONSOLIDATION_PROMPT,
)


# Reasoning Model Agent (for large consolidations)

reasoning_model = OpenAIModel(
    model_name="o1-mini",
    system_prompt_role="user",
)

finalization_agent = Agent[None, str](
    model=reasoning_model,
    result_type=str,
    retries=4,
    system_prompt=FINALIZATION_PROMPT,
)


# Title and Description Agent
title_description_agent = Agent[None, TitleAndDescriptionResult](
    model="openai:gpt-4o",
    result_type=TitleAndDescriptionResult,
    retries=3,
    system_prompt=TITLE_DESCRIPTION_PROMPT,
)


async def process_chunk(chunk: str) -> Tuple[str, Usage]:
    """
    Process a single chunk of medical records text.

    Args:
        chunk: The text chunk to process
        deps: Dependencies for the agent

    Returns:
        A tuple containing the summary and usage information
    """
    result = await medical_records_summary_agent.run(chunk)
    return result.data, result.usage()


async def consolidate_summaries(
    summaries: List[str], chunk_size: int
) -> Tuple[str, List[Usage]]:
    """
    Consolidate multiple summaries into a single summary.

    Args:
        summaries: List of summaries to consolidate
        deps: Dependencies for the agent

    Returns:
        A tuple containing the consolidated summary and usage information
    """
    usages = []

    # Join summaries with separators
    combined_summaries = "\n\n---\n\n".join(summaries)

    # Check if the combined summaries are too large
    tokens = count_tokens(combined_summaries)

    if tokens > chunk_size:
        # If still too large, do multi-step consolidation
        # Split summaries into smaller groups
        batch_size = max(2, len(summaries) // 2)  # At least 2 summaries per batch
        batches = [
            summaries[i : i + batch_size] for i in range(0, len(summaries), batch_size)
        ]

        # First level consolidation
        first_level_results = []
        for batch in batches:
            batch_text = "\n\n---\n\n".join(batch)
            result = await consolidation_agent.run(batch_text)
            first_level_results.append(result.data)
            usages.append(result.usage())

        # Second level consolidation
        combined_text = "\n\n---\n\n".join(first_level_results)
        result = await consolidation_agent.run(combined_text)
        usages.append(result.usage())
        return result.data, usages
    else:
        # Single consolidation step
        result = await consolidation_agent.run(combined_summaries)
        usages.append(result.usage())
        return result.data, usages


async def generate_title_description(
    summary: str,
) -> Tuple[TitleAndDescriptionResult, Usage]:
    """
    Generate a title and description for a summary.

    Args:
        summary: The summary to generate a title and description for
        deps: Dependencies for the agent

    Returns:
        A tuple containing the title and description result and usage information
    """
    result = await title_description_agent.run(summary)
    return result.data, result.usage()


async def process_medical_records(
    medical_records: List[ConversionResult],
    chunk_size: int = 20000,
    add_labels: bool = False,
    cap_multiplier: float = 1.05,
) -> MedicalRecordsSummaryResult:
    """
    Process medical records to generate a comprehensive summary.

    Args:
        medical_records: List of medical records to process
        chunk_size: Maximum size of each chunk in tokens
        add_labels: Whether to add title and description labels
        cap_multiplier: Multiplier for chunk size cap

    Returns:
        A MedicalRecordsSummaryResult containing the summary and token usage
    """
    usages = []

    # Prepare document chunks
    medical_chunked_docs = prepare_processed_document_chunks(
        medical_records,
        chunk_size=chunk_size,
        cap_multiplier=cap_multiplier,
    )

    # Check if we can process all records at once
    if sum([doc.token_count for doc in medical_chunked_docs]) < chunk_size:
        # Process all records at once
        medical_text_list = [doc.text_trimmed for doc in medical_records]
        medical_text_string = "\n".join(medical_text_list)

        draft_medical_summary, usage = await process_chunk(medical_text_string)
        usages.append(usage)
    else:
        # Process chunks in parallel
        chunk_tasks = []
        for doc in medical_chunked_docs:
            for chunk in doc.text_chunks:
                chunk_tasks.append(process_chunk(chunk.text))

        # Wait for all chunk processing to complete
        chunk_results = await asyncio.gather(*chunk_tasks)

        # Extract summaries and usages
        summaries = [result[0] for result in chunk_results]
        chunk_usages = [result[1] for result in chunk_results]
        usages.extend(chunk_usages)

        # Consolidate summaries
        draft_medical_summary, consolidation_usages = await consolidate_summaries(
            summaries, chunk_size
        )
        usages.extend(consolidation_usages)

    # Finalize summary
    medical_summary_response = await finalization_agent.run(draft_medical_summary)
    usages.append(medical_summary_response.usage())
    medical_summary = medical_summary_response.data

    # Generate title and description if needed
    document_title = ""
    document_description = ""
    if add_labels:
        labels_result, labels_usage = await generate_title_description(medical_summary)
        document_title = labels_result.title
        document_description = labels_result.description
        usages.append(labels_usage)

    return MedicalRecordsSummaryResult(
        document_name="Summary of plaintiff's medical records",
        document_title=document_title,
        document_description=document_description,
        summary=medical_summary,
        usages=usages,
    )


# Synchronous wrapper
def run_medical_records_summary(
    medical_records: List[ConversionResult],
    chunk_size: int = 20000,
    add_labels: bool = False,
    cap_multiplier: float = 1.05,
) -> MedicalRecordsSummaryResult:
    """
    Synchronous wrapper for processing medical records.

    Args:
        medical_records: List of medical records to process
        chunk_size: Maximum size of each chunk in tokens
        add_labels: Whether to add title and description labels
        cap_multiplier: Multiplier for chunk size cap

    Returns:
        A MedicalRecordsSummaryResult containing the summary and metadata
    """
    return asyncio.run(
        process_medical_records(
            medical_records=medical_records,
            chunk_size=chunk_size,
            add_labels=add_labels,
            cap_multiplier=cap_multiplier,
        )
    )
