"""Document processing strategies.

This module contains various strategies for processing different document combinations.
"""

from .primary_and_supporting import PrimaryAndSupportingStrategy
from .primary_only import PrimaryOnlyStrategy
from .medical_only import MedicalOnlyStrategy
from .supporting_only import SupportingOnlyStrategy

__all__ = [
    "PrimaryAndSupportingStrategy",
    "PrimaryOnlyStrategy",
    "MedicalOnlyStrategy",
    "SupportingOnlyStrategy",
] 