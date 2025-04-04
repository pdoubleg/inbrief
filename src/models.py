from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai.usage import Usage


class ProcessingType(str, Enum):
    """Processing types supported by the engine."""
    
    PRIMARY_AND_SUPPORTING = "primary_and_supporting"
    PRIMARY_ONLY = "primary_only"
    SUPPORTING_ONLY = "supporting_only"
    MEDICAL_ONLY = "medical_only"

class ConversionResult(BaseModel):
    """Represents a converted document with its metadata and content."""

    name: Optional[str] = Field(default=None, description="The name of the document")
    text: Optional[str] = Field(default=None, description="The full text content of the document") 
    text_trimmed: Optional[str] = Field(default=None, description="Trimmed text content of the document")
    page_text: Optional[str] = Field(default=None, description="Text content organized by pages")
    pages: Optional[List] = Field(default=None, description="List of Page objects containing text")
    token_count: Optional[int] = Field(default=0, description="The token count of the document")


class Page(BaseModel):
    """Represents a page in a document."""

    page_number: int = Field(description="The page number")
    text: str = Field(description="The text content of the page")
    table_headers: List[str] = Field(
        default_factory=list, description="Headers of tables found on the page"
    )


class TextChunk(BaseModel):
    """Represents a chunk of text from a document."""

    text: str = Field(description="The text content of the chunk")
    start_page: int = Field(description="The starting page number of the chunk")
    end_page: int = Field(description="The ending page number of the chunk")
    token_count: int = Field(description="The token count of the chunk")


class ProcessedDocument(ConversionResult):
    """Represents a processed document with text chunks."""

    processed_text: str = Field(description="The processed text content")
    token_count: int = Field(description="The token count of the processed text")
    text_chunks: List[TextChunk] = Field(description="List of text chunks")


class ContextSummary(BaseModel):
    """Represents a summary of a supporting document (exhibit)."""

    exhibit_number: int = Field(description="The exhibit number")
    file_name: str = Field(description="The file name of the exhibit")
    summary: str = Field(description="The summary of the exhibit")
    document_description: Optional[str] = Field(
        default=None, description="Description of the document"
    )
    document_title: Optional[str] = Field(
        default=None, description="Title of the document"
    )
    usages: List[Usage] = Field(default_factory=list, description="Usage information")

    def __str__(self) -> str:
        """String representation of the context summary."""
        exhibit_info = f"Exhibit {self.exhibit_number}: {self.file_name}"
        if self.document_title:
            exhibit_info += f"\nTitle: {self.document_title}"
        if self.document_description:
            exhibit_info += f"\nDescription: {self.document_description}"
        return f"{exhibit_info}\n\n{self.summary}"


class ContextSummaries(BaseModel):
    """Collection of context summaries."""

    summaries: List[ContextSummary] = Field(
        default_factory=list, description="List of context summaries"
    )

    def docs_info_string(self, number: bool = True) -> str:
        """Generate a string with information about all documents."""
        result = []
        for summary in self.summaries:
            prefix = f"Exhibit {summary.exhibit_number}: " if number else ""
            result.append(f"{prefix}{summary.file_name}")
        return "\n".join(result)

    def get_exhibit_string(self, exhibit_number: int) -> str:
        """Get the text of an exhibit by its number."""
        exhibit = next(
            (e for e in self.summaries if e.exhibit_number == exhibit_number), None
        )
        return str(exhibit) if exhibit else "Exhibit not found."


# Agent-specific models


class ExhibitsResearchItem(BaseModel):
    """Item from a discovery document that depends on information from the exhibits."""

    chain_of_thought: str = Field(
        description="The reasoning behind the research item in relation to the exhibits."
    )

    excerpt: str = Field(
        description="The verbatim excerpt from the discovery document that needs to be researched. Should include the item, its number, and response whenever possible."
    )

    question: str = Field(
        description="The question that requires information from the exhibits to answer."
    )


class ExhibitsResearchNotNeeded(BaseModel):
    """Use if research items are not needed or when exhibits are not relevant to the discovery document."""


class ExhibitsSelection(BaseModel):
    """Exhibit number(s) selected to answer a research item."""

    chain_of_thought: str = Field(
        description="The reasoning behind the selection of the exhibit number(s)."
    )

    exhibit_numbers: List[int] = Field(
        description="The exhibit number(s) selected to answer the research item. Must choose at least one exhibit number."
    )


class ExhibitsResearchResult(BaseModel):
    """Result of a research task."""

    result_string: str = Field(description="String of results")
    usages: List[Usage] = Field(description="Usage information")


class MedicalRecordsSummaryResult(BaseModel):
    """Result of a medical records summary task."""

    summary: str = Field(description="The medical records summary")
    usages: List[Usage] = Field(default_factory=list, description="Usage information")
    document_name: Optional[str] = Field(None, description="Name of the document")
    document_title: Optional[str] = Field(None, description="Title of the document")
    document_description: Optional[str] = Field(
        None, description="Description of the document"
    )


class TitleAndDescriptionResult(BaseModel):
    """Title and description for a medical records summary."""

    title: str = Field(description="The title of the text")
    description: str = Field(description="The description of the text")
    usages: List[Usage] = Field(default_factory=list, description="Usage information")


class EntityListing(BaseModel):
    """A listing of medical providers and other parties/entities found in discovery documents."""

    entity_listing: str = Field(
        description="A well formatted listing of providers, parties/entities."
    )


class ResolvedEntityListing(BaseModel):
    """A resolved listing of medical providers and other parties/entities with duplicates removed."""

    resolved_listings: str = Field(description="A resolved entity listing.")


class FinalizedEntityListing(BaseModel):
    """A finalized listing of medical providers and other parties/entities."""

    finalized_listing: str = Field(description="A finalized entity listing.")


class ProviderListingResult(BaseModel):
    """Result of provider listing generation."""

    resolved_listing: str = Field(
        description="The final resolved listing of providers and parties."
    )
    usages: List[Usage] = Field(default_factory=list, description="Usage information for the agents.")


class QCResult(BaseModel):
    """Quality control result on the finalized entity listing."""

    score: int = Field(
        gt=0,
        le=10,
        description="The score (0-10) of the quality control review with 10 being perfect and 0 being unacceptable.",
    )
    feedback: Optional[str] = Field(
        description="For any issues found, provide a detailed explanation of what needs to be changed."
    )


class DiscoverySummaryResult(BaseModel):
    """Result of a discovery document summary task."""

    summary: str = Field(
        description="The detailed narrative summary of the discovery document"
    )
    usages: Optional[List[Usage]] = Field(default_factory=list, description="Usage information")
    reasoning_model_flag: Optional[bool] = Field(
        default=False, description="Whether the reasoning model was used"
    )
    reasoning_prompt_tokens: Optional[int] = Field(
        default=0, description="Number of prompt tokens used by reasoning model"
    )
    reasoning_completion_tokens: Optional[int] = Field(
        default=0, description="Number of completion tokens used by reasoning model"
    )


class SummaryResult(BaseModel):
    """Result model for the entire summarization process."""

    long_version: Optional[str] = Field(
        default=None, description="The detailed long version summary."
    )
    short_version: Optional[str] = Field(
        default=None, description="The condensed short version summary."
    )
    cost: Optional[float] = Field(
        default=None, description="The total cost of the summarization process."
    )
    processing_time: Optional[float] = Field(
        default=None, description="The total processing time in seconds."
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the summarization process failed."
    )
    usages: List[Usage] = Field(
        default_factory=list, description="The usages of the summarization process."
    )


class ShortVersionResult(BaseModel):
    """Result of a short version generation task."""

    summary: str = Field(description="The condensed short version summary.")
    usages: Optional[Usage] = Field(None, description="Usage information")


class DocumentsProducedResult(BaseModel):
    """Result of a documents produced generation task."""

    summary: str = Field(description="The summary of the documents produced.")
    usages: Optional[Usage] = Field(None, description="Usage information")
