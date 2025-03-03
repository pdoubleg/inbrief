DISCOVERY_SUMMARY_EXAMPLE_OUTPUT_1 = """
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

Plaintiff produced medical records, the police report, photographs and a pay stub information. She also produced a printout indicating 3 hours of time off from work on 9/15/22 and 8 hours off work on both 9/16/22 and 9/19/22."""


DISCOVERY_SUMMARY_EXAMPLE_OUTPUT_2 = """
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

The Plaintiff produced the police report, medical records, and medical bills."""


DISCOVERY_SUMMARY_EXAMPLES = [
    {
        "input": "...",
        "response": DISCOVERY_SUMMARY_EXAMPLE_OUTPUT_1,
    },
    {
        "input": "...",
        "response": DISCOVERY_SUMMARY_EXAMPLE_OUTPUT_2,
    },
]

DOCUMENTS_PRODUCED_EXAMPLE_1 = """\
smith_johnson_rogs.pdf: Plaintiff's first set of interrogatories.\\n
Records and Bills.pdf: Medical records from Happy Spine Institute and New Wave Yoga."""

DOCUMENTS_PRODUCED_EXAMPLES = [
    {
        "input": "...",
        "response": DOCUMENTS_PRODUCED_EXAMPLE_1,
    },
]


MEDICAL_RECORDS_SUMMARY_EXAMPLE_OUTPUT_1 = """\
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
he had not started PT and that he would use his regular insurance to treat his knee. Total billing for both office visits was $2,300.00."""

MEDICAL_RECORDS_SUMMARY_EXAMPLES = [
    {
        "input": "...",
        "response": MEDICAL_RECORDS_SUMMARY_EXAMPLE_OUTPUT_1,
    },
]

PROVIDER_LISTING_EXAMPLE_OUTPUT_1 = """\
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
Notes: Plaintiff was employed here and out on delivery at the time of the accident."""

PROVIDER_LISTING_EXAMPLES = [
    {
        "input": "...",
        "response": PROVIDER_LISTING_EXAMPLE_OUTPUT_1,
    },
]
