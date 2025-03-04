"""Summary Engine V2.

This package implements an improved version of the summary engine
with better separation of concerns and adherence to the Strategy Pattern.
"""

from .context import DocumentProcessor
from .strategies import (
    PrimaryAndSupportingStrategy,
    PrimaryOnlyStrategy,
    MedicalOnlyStrategy,
    SupportingOnlyStrategy,
)

__all__ = [
    "DocumentProcessor",
    "PrimaryAndSupportingStrategy",
    "PrimaryOnlyStrategy",
    "MedicalOnlyStrategy",
    "SupportingOnlyStrategy",
] 