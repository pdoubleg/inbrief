"""
Example usage of the Discovery Document Summarization Agent

This script demonstrates how to use the pydantic-ai based discovery agent to summarize
legal discovery documents.
"""

import time

from discovery_agent import run_discovery_summary

# Test function to summarize discovery documents
def run_example():
    """
    Run an example discovery document summarization.
    
    This function demonstrates how to use the discovery agent to summarize
    a discovery document and optional supporting documents.
    """
    # Example discovery document (in a real scenario, this would be loaded from a file)
    discovery_document = """
CASE CAPTION : JOHNSON, MICHAEL V. SMITH CONSTRUCTION CO.

CLAIM NUMBER: LA592-084732164-0002

---

INTERROGATORIES:

1. Please state your full name, address, date of birth, and Social Security number.
ANSWER: Michael Johnson, 123 Oak Street, Seattle, WA. DOB: 05/12/1975. I object to providing my Social Security number on privacy grounds.

2. Please describe, in detail, how the incident that is the subject of this lawsuit occurred.
ANSWER: On June 15, 2023, I was working at the construction site at 456 Main Street, Seattle, WA. I was walking across the site when I stepped into an unmarked hole that had been covered with a thin piece of plywood. The plywood broke under my weight, and I fell approximately 4 feet into the hole, landing on my right side.

3. Please identify all injuries you claim to have sustained as a result of the incident that is the subject of this lawsuit.
ANSWER: As a result of the fall, I sustained a fractured right ankle, sprained right wrist, contusions to my right hip and shoulder, and lower back strain. I also experienced post-traumatic headaches for approximately 2 weeks following the incident.

4. Please identify all healthcare providers who have treated you for the injuries identified in your answer to the preceding interrogatory.
ANSWER: Dr. Jennifer Smith - Seattle Medical Center (06/15/2023 - 09/30/2023)
Dr. Robert Taylor - Pacific Orthopedics (06/17/2023 - 11/12/2023)
Physical Therapy Northwest - therapist Mark Wilson (07/01/2023 - 11/15/2023)

5. Please state the amount of medical expenses you have incurred to date for treatment of the injuries identified in your answer to Interrogatory No. 3.
ANSWER: To date, I have incurred the following medical expenses:
Seattle Medical Center: $4,850
Pacific Orthopedics: $6,725
Physical Therapy Northwest: $3,400
Prescription medications: $780
Total: $15,755

6. Have you ever been involved in any other accidents or incidents that resulted in injuries similar to those you claim to have sustained in the incident that is the subject of this lawsuit? If so, please describe each such accident or incident, including the date, location, circumstances, and injuries sustained.
ANSWER: In 2018, I twisted my ankle while hiking, resulting in a mild sprain. It was treated with rest and over-the-counter pain medication and resolved within 2 weeks. There was no fracture or long-term impact. I have had no other injuries to my right ankle, wrist, hip, shoulder, or lower back prior to this incident.

7. Please identify all employers for whom you have worked during the past 10 years, including the dates of employment, positions held, and reasons for leaving each position.
ANSWER: Urban Construction (2013-2018) - Construction Worker - Left for better pay
Northwest Builders (2018-2021) - Senior Construction Worker - Company downsized
Smith Construction Co. (2021-Present) - Construction Foreman - Currently employed

8. Please state the amount of income you claim to have lost as a result of the incident that is the subject of this lawsuit, and describe how that amount was calculated.
ANSWER: I was unable to work for 12 weeks following the incident. My weekly income as a Construction Foreman at Smith Construction Co. is $1,200. Therefore, my lost income is $14,400 (12 weeks × $1,200/week).

9. Do you claim any permanent disability, impairment, or loss of earning capacity as a result of the incident that is the subject of this lawsuit? If so, please describe the nature and extent of such disability, impairment, or loss of earning capacity.
ANSWER: According to Dr. Taylor at Pacific Orthopedics, I have a 5% permanent impairment of my right ankle due to the fracture. This impairment affects my ability to stand for extended periods, climb ladders, and walk on uneven surfaces, which are essential functions of my job as a Construction Foreman. Dr. Taylor has advised that I may need to transition to a less physically demanding role in the future, which could result in reduced earning capacity.

10. Please identify all documents you are producing in response to the Request for Production that accompanies these Interrogatories.
ANSWER: Medical records from all providers listed in response to Interrogatory No. 4
Medical bills from all providers
Pay stubs for the 3 months before and after the incident
Employment records from Smith Construction Co.
Photographs of the accident site taken on the day of the incident
Photographs of my injuries taken at various stages of recovery
Incident report filed with Smith Construction Co. on June 15, 2023
"""

    supporting_documents = """
SUPPORTING DOCUMENT 1: MEDICAL RECORD FROM SEATTLE MEDICAL CENTER

Patient: Michael Johnson
Date of Service: June 15, 2023
Provider: Dr. Jennifer Smith

Chief Complaint: Patient presents to ER following fall at construction site.

History of Present Illness: 48-year-old male construction worker who fell approximately 4 feet into a hole at work site today. Patient reports landing on his right side, with immediate pain in right ankle, wrist, hip, and shoulder. Also reports back pain and headache.

Physical Examination:
- General: Alert and oriented, in moderate distress due to pain.
- Vital Signs: BP 138/85, HR 92, RR 18, Temp 98.6°F, O2 Sat 99%
- Right Ankle: Marked swelling and tenderness over lateral malleolus. Limited range of motion due to pain. Neurovascular status intact.
- Right Wrist: Mild swelling and tenderness over dorsal aspect. Full range of motion with pain. No crepitus.
- Right Hip/Shoulder: Contusions visible. Tenderness to palpation. Range of motion limited by pain.
- Back: Paraspinal muscle tenderness in lumbar region. Negative straight leg raise.
- Neurological: Intact throughout.

Diagnostic Studies:
- X-ray Right Ankle: Distal fibula fracture
- X-ray Right Wrist: No fracture
- X-ray Lumbar Spine: No acute abnormality; mild degenerative changes at L4-L5

Assessment:
1. Right distal fibula fracture
2. Right wrist sprain
3. Contusions to right hip and shoulder
4. Lumbar strain
5. Post-traumatic headache

Plan:
1. Right ankle: Apply short leg cast. Non-weight-bearing for 4 weeks. Refer to orthopedics.
2. Right wrist: Wrist splint for comfort. NSAIDs for pain.
3. Contusions and lumbar strain: Rest, ice, NSAIDs.
4. Headache: Monitor symptoms. Return if worsening.
5. Prescribed: Ibuprofen 800mg TID, Tramadol 50mg q6h prn severe pain
6. Work status: No work for 2 weeks, then re-evaluate.
7. Follow-up with orthopedics in 1 week.

SUPPORTING DOCUMENT 2: EMPLOYMENT VERIFICATION

To Whom It May Concern:

This letter confirms that Michael Johnson has been employed by Smith Construction Co. since March 15, 2021, in the position of Construction Foreman.

Mr. Johnson's current salary is $62,400 per year ($1,200 per week). His regular work schedule is Monday through Friday, 7:00 AM to 3:30 PM.

Following the workplace incident on June 15, 2023, Mr. Johnson was unable to work for a period of 12 weeks. He returned to modified duty on September 7, 2023, with restrictions including no climbing, limited standing, and no lifting more than 20 pounds.

As of December 1, 2023, Mr. Johnson remains on modified duty based on his physician's recommendations.

Sincerely,
Barbara Wilson
Human Resources Director
Smith Construction Co.
"""

    print("Running discovery document summarization...")
    start_time = time.time()
    
    # Run the discovery summary
    result = run_discovery_summary(
        discovery_document=discovery_document,
        supporting_documents=supporting_documents,
        reasoning_model_threshold=200,
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\nSummarization completed in {elapsed_time:.2f} seconds")
    print("\nDISCOVERY SUMMARY:")
    print("="*80)
    print(result.summary)
    print("="*80)
    
    # Print usage statistics
    total_prompt_tokens = sum(usage.request_tokens for usage in result.usages)
    total_completion_tokens = sum(usage.response_tokens for usage in result.usages)
    total_tokens = total_prompt_tokens + total_completion_tokens
    
    print("\nUSAGE STATISTICS:")
    print(f"Total Prompt Tokens: {total_prompt_tokens}")
    print(f"Total Completion Tokens: {total_completion_tokens}")
    print(f"Total Tokens: {total_tokens}")
    
    if result.reasoning_model_flag:
        print("\nReasoning Model Usage:")
        print(f"Prompt Tokens: {result.reasoning_prompt_tokens}")
        print(f"Completion Tokens: {result.reasoning_completion_tokens}")


if __name__ == "__main__":
    run_example() 