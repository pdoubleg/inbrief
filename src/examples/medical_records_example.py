"""Example usage of the refactored SummaryEngine for medical records.

This example demonstrates how to use the refactored SummaryEngine to process
medical records using the new pydantic-ai approach and the adapter pattern.
"""

import os
import sys
import time
from typing import List

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ConversionResult
from src.summary_engine import SummaryEngine


def create_mock_medical_records() -> List[ConversionResult]:
    """Create mock medical records for demonstration purposes.

    Returns:
        List of mock medical records
    """
    # In a real scenario, these would be loaded from files
    medical_record1 = ConversionResult(
        name="Patient Medical Record 1",
        text="Patient: John Doe\nDOB: 01/01/1970\nVisit Date: 05/15/2023\n"
        "Doctor: Dr. Smith\nClinic: City Medical Center\n"
        "Chief Complaint: Patient presents with persistent cough for 2 weeks.\n"
        "Assessment: Upper respiratory infection. Prescribed antibiotics.\n"
        "Follow-up: 2 weeks",
        text_trimmed="",
        page_text="",
        pages=[],
    )

    medical_record2 = ConversionResult(
        name="Patient Medical Record 2",
        text="Patient: John Doe\nDOB: 01/01/1970\nVisit Date: 06/01/2023\n"
        "Doctor: Dr. Johnson\nHospital: City General Hospital\n"
        "Reason for Visit: Follow-up for respiratory infection.\n"
        "Assessment: Patient's symptoms have improved. Infection clearing.\n"
        "Plan: Continue medication for 5 more days.",
        text_trimmed="",
        page_text="",
        pages=[],
    )

    return [medical_record1, medical_record2]


def run_medical_records_example():
    """Run an example of processing medical records with the refactored SummaryEngine."""
    print("=== Medical Records Processing Example ===")

    # Create mock medical records
    medical_records = create_mock_medical_records()
    print(f"Created {len(medical_records)} mock medical records.")

    # Initialize SummaryEngine for medical records
    print("\nInitializing SummaryEngine with traditional approach...")
    engine_traditional = SummaryEngine(
        supporting_docs=medical_records,
        has_medical_records=True,
        job_id="example-job-traditional",
        cache=False,
        use_pydantic_ai=False,  # Use traditional approach
    )

    # Process medical records with traditional approach
    print("Processing medical records with traditional approach...")
    start_time = time.perf_counter()
    traditional_summary = engine_traditional.run_summarization_process(
        medical_only=True
    )
    traditional_time = time.perf_counter() - start_time

    # Initialize SummaryEngine with pydantic-ai
    print("\nInitializing SummaryEngine with pydantic-ai approach...")
    engine_pydantic = SummaryEngine(
        supporting_docs=medical_records,
        has_medical_records=True,
        job_id="example-job-pydantic",
        cache=False,
        use_pydantic_ai=True,  # Use pydantic-ai approach
    )

    # Process medical records with pydantic-ai
    print("Processing medical records with pydantic-ai approach...")
    start_time = time.perf_counter()
    pydantic_summary = engine_pydantic.run_summarization_process(medical_only=True)
    pydantic_time = time.perf_counter() - start_time

    # Display results
    print("\n=== Results ===")
    print(f"Traditional approach time: {traditional_time:.2f} seconds")
    print(f"Pydantic-ai approach time: {pydantic_time:.2f} seconds")

    print("\n=== Traditional Summary ===")
    print(
        traditional_summary[:500] + "..."
        if len(traditional_summary) > 500
        else traditional_summary
    )

    print("\n=== Pydantic-AI Summary ===")
    print(
        pydantic_summary[:500] + "..."
        if len(pydantic_summary) > 500
        else pydantic_summary
    )


if __name__ == "__main__":
    run_medical_records_example()
