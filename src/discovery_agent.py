import asyncio
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.usage import Usage
from pydantic_ai.models.openai import OpenAIModel

from utils import count_tokens


# Define the model for discovery summary results
class DiscoverySummaryResult(BaseModel):
    """Result of a discovery document summary task."""

    summary: str = Field(
        description="The detailed narrative summary of the discovery document"
    )
    usages: List[Usage] = Field(default_factory=list, description="Usage information")
    reasoning_model_flag: bool = Field(
        default=False, description="Whether the reasoning model was used"
    )
    reasoning_prompt_tokens: int = Field(
        default=0, description="Number of prompt tokens used by reasoning model"
    )
    reasoning_completion_tokens: int = Field(
        default=0, description="Number of completion tokens used by reasoning model"
    )


# System prompts for the discovery agents
DISCOVERY_SUMMARY_PROMPT = """\
You are a legal assistant for a high-power and very busy attorney. Summarize the following context, which is a DISCOVERY DOCUMENT \
containing interrogatories and their respective responses and optionally submitted SUPPORTING DOCUMENTS. You maintain the highest level of objectivity and never \
use subjective adjectives. For example, you would never say "significant injuries" and instead summarize the actual injuries you are referring to. Additionally, \
rather than describing a document as having "comprehensive" information, you would simply summarize the facts contained in the document. \
Aim for a clear, and professional narrative summary that captures the most relevant information for the attorney and that follows closely the \
EXAMPLE OUTPUT provided below.

Note the term 'interrogatories' refers generally to a formal set of questions. The context may contain a variety of legal documents, including but not limited to, \
interrogatories, requests for production of documents, admissions, and general discovery-related questions or requests. \
Please adapt your output to the specific DISCOVERY DOCUMENTS provided, and apply the following instructions for the type of legal document(s) you are summarizing.

# Please follow these instructions:
1. **Review the document carefully** to understand the context and purpose of the discovery document(s) and if provided supporting documents.
2. **Identify the key issues and topics** that the document(s) addresses.
3. **Present the summary in a clear and organized manner**, ensuring that all **important** details are captured.
4. **Focus on key interrogatories/questions/requests** related to **damages** such as medical expenses, lost wages/lost time from work and lost future earning capacity.

# **Important Notes:**
* **Follow closely the format** provided in the EXAMPLE OUTPUT.
* **Incorporate SUPPORTING DOCUMENTS** into the main narrative summary, do not summarize them separately.
* **Accuracy is key**, ensure that all information is explicitly stated in the document(s).
* Always **communicate uncertainty** when information is not clear.
* **NEVER** include overall statements, general thoughts or opinions, simply state the facts in a highly professional manner.
* On personal information, include the **Plaintiff's age and city of residence**, DO NOT LIST date of birth, full address, or phone number.
* Remember to be concise and to the point, focusing on the most relevant litigation-related information. Do NOT waste time by stating the obvious, i.e., 'this document covers a range of topics...'! **Your goal is to provide a legal style report suitable for a Senior Attorney.**

# NEVER USE BULLET POINTS OR NUMBERED LISTS IN YOUR RESPONSE!!!

[[## EXAMPLE OUTPUT 1 ##]]

CASE CAPTION : CAREY, MOLLIE V POSENCHEG, HANNAH

CLAIM NUMBER: LA830-051141399-0001

---

DISCOVERY SUMMARY:

PARTY RESPONDING / TITLE OF RESPONSE:  Plaintiff Mollie Jeanne Carey Responses to Defendant's Standard Discovery

HIGH LEVEL SUMMARY:  Plaintiff, age 30, currently resides in Wayne, PA.  She has been employed by Essent Guaranty in marketing since November, 2021.  She has health insurance though United Healthcare.  Plaintiff states she has not made any other claims/lawsuits in the last 10 years.  Her primary provider was Dr. Louise Maloney in Essex, CT until 2021 and then was seen at Mawr Family Practice.

Plaintiff was going to a restaurant in Conshocken and was stopped at a yield sign waiting for traffic to pass when she was rear-ended by Defendants' vehicle.

Plaintiff claims injuries to her neck, low back and right hand. 

She was initially seen at Bryn Mawr Family Practice two days after the accident.  She was next seen at the Chiropractic Spine Center by Theodore Glazer, DC (10/5/222 - 2/6/23) and Premier Orthopaedics at Broomall for physical therapy (10/4/22 -  12/7/23).  She was also seen at Main Line Health (Main Line Health Orthopedics and Rehab Associates of the Main Line). She underwent a lumbar MRI at Open MRI, ordered by Ted Glazer, on 12/13/22.  She had a cervical MRI at Main Line Health, ordered by Ajit Jada MD, on 1/22/23.  Main Line Health records indicate she was seen by Jeffrey Friedman (3/20/23 - 1/2/24).  Dr. Friedman noted a prior history of MVA at age 17.  In April, 2023 Plaintiff reported she had returned to jogging but not running. On 10/31/23, Plaintiff reported a recent fall from a height of 6 feet while in a corn maze, injuring her neck, back and right hand/thumb. She also reported sustaining a concussion from that fall and was wearing a hand splint and sling on her right arm.

Plaintiff states she still has neck and back pain that interfere with activities.  She claims she missed 3 days of work after the DOL.  

Plaintiff produced medical records, the police report, photographs and a pay stub information. She also produced a printout indicating 3 hours of time off from work on 9/15/22 and 8 hours off work on both 9/16/22 and 9/19/22.

[[## EXAMPLE OUTPUT 2 ##]]

CASE CAPTION :  LAURIE S. DESOUSA V. ANTHONY A. TAVAERAS, ET AL

CLAIM NUMBER:  LA179-047798050-0004

---

DISCOVERY SUMMARY:

PARTY RESPONDING / TITLE OF RESPONSE: Plt's Responses to Def 's Discovery

HIGH LEVEL SUMMARY: Plt is 56 years old and resides in Johnston, RI. PCP is Dr. Christopher Storey. No prior ailments or injury. Has Medicaid.

Plaintiff was traveling on Cherry Hill Road when she slowed to a stop sign to turn left onto Birch Tree Drive. While stopped she was rear-ended by the Defendant, Anthony Tavares. No known witnesses. Def 's father came up to her passenger side window and asked if she was okay and apologized for his son. He said that his son had rear ended somebody a month before while texting and driving.

The Plaintiff sustained bruising to her left shin, and sprains to her cervical, thoracic, and lumbar \
spine. No permanency. She is not making a claim for lost wages. She first sought treatment at \
Atmed Treatment Center on 12/07/21. Her complaints were of backside left leg below knee pain, \
right hand pain, left side rib/hip pain, lower back sore, and neck & shoulders. Assessed with a neck \
and back strain. Taking cyclobenzaprine and ibuprofen. Referred for chest and left leg x-rays. \
Consider PT. X-rays came back normal. She began treatment at RI Chiro on 12/08/21. Lumbar X-rays were performed 12/14/21. \
Mild degenerative changes at L3-4 were noted. Cervical x-rays \
showed a suspected muscular spasm. On January 4, 2022 she stated that her heart has been racing \
since the accident. She will see her PCP. Also recommended massage therapy. Lumbar MRI \
occurred 02/19/22. It revealed mild degenerative changes. No significant spinal canal or \
neuroforaminal stenosis at any levels. On March 9, 2022 she was seen by P. Canchis MD at Atmed \
stating that her heart has been fluttering and pounding. She attributes symptoms to the accident. \
Assessed with palpitations, hyperlipidemia, and anxiety disorder. Given lorazepam for anxiety and \
EKG ordered. On April 21, 2022 Plt stated that her symptoms had resolved.

The Plaintiff produced the police report, medical records, and medical bills.

[[## END OF EXAMPLE OUTPUT ##]]

**IMPORTANT REMINDER:** DO NOT USE BULLET POINTS OR NUMBERED LISTS!
"""

# Initialize the discovery summary agent
discovery_summary_agent = Agent[None, str](
    model="openai:gpt-4o",
    result_type=str,
    retries=3,
    system_prompt=DISCOVERY_SUMMARY_PROMPT,
)

# Reasoning Model Agent (for large document processing)
reasoning_model = OpenAIModel(
    model_name="o1-mini",
    system_prompt_role="user",
)

reasoning_agent = Agent[None, str](
    model=reasoning_model,
    result_type=str,
    retries=4,
    system_prompt=DISCOVERY_SUMMARY_PROMPT,
)


async def process_discovery_document(
    discovery_document: str,
    supporting_documents: Optional[str] = "",
    reasoning_model_threshold: int = 30000,
) -> DiscoverySummaryResult:
    """
    Process discovery documents to generate a comprehensive summary.

    Args:
        discovery_document: The main discovery document to summarize
        supporting_documents: Optional supporting documents
        reasoning_model_threshold: Token threshold for using reasoning model

    Returns:
        A DiscoverySummaryResult containing the summary and token usage
    """
    # Combine discovery document and supporting documents
    all_input_text = (
        discovery_document + "\n\n" + supporting_documents
        if supporting_documents
        else discovery_document
    )
    input_tokens = count_tokens(all_input_text)

    reasoning_model_flag = False
    reasoning_prompt_tokens = 0
    reasoning_completion_tokens = 0

    # Choose model based on token count
    if input_tokens < reasoning_model_threshold:
        # Use standard agent for smaller documents
        summary_response = await discovery_summary_agent.run(all_input_text)
        summary = summary_response.data
        usages = [summary_response.usage()]
    else:
        # Use reasoning model for larger documents
        reasoning_model_flag = True
        summary_response = await reasoning_agent.run(all_input_text)
        summary = summary_response.data
        usage = summary_response.usage()
        usages = [usage]

        # Track reasoning model token usage
        reasoning_prompt_tokens = usage.request_tokens
        reasoning_completion_tokens = usage.response_tokens

    return DiscoverySummaryResult(
        summary=summary,
        usages=usages,
        reasoning_model_flag=reasoning_model_flag,
        reasoning_prompt_tokens=reasoning_prompt_tokens,
        reasoning_completion_tokens=reasoning_completion_tokens,
    )


# Synchronous wrapper
def run_discovery_summary(
    discovery_document: str,
    supporting_documents: Optional[str] = "",
    reasoning_model_threshold: int = 30000,
) -> DiscoverySummaryResult:
    """
    Synchronous wrapper for processing discovery documents.

    Args:
        discovery_document: The main discovery document to summarize
        supporting_documents: Optional supporting documents
        reasoning_model_threshold: Token threshold for using reasoning model

    Returns:
        A DiscoverySummaryResult containing the summary and metadata
    """
    return asyncio.run(
        process_discovery_document(
            discovery_document=discovery_document,
            supporting_documents=supporting_documents,
            reasoning_model_threshold=reasoning_model_threshold,
        )
    )
