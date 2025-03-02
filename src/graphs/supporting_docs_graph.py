"""
Graph-based implementation of the supporting documents agent.

This module provides a Pydantic-AI graph implementation for processing
and summarizing legal discovery documents.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Union

from pydantic_ai import Agent
from pydantic_ai.usage import Usage
from pydantic_graph import BaseNode, Graph, GraphRunContext, End

from ..models import (
    ConversionResult,
    ContextSummary,
    ContextSummaries,
    ProcessedDocument,
    TitleAndDescriptionResult,
)
from ..utils import prepare_processed_document_chunks, count_tokens

# System prompts from original implementation
DOCUMENT_SUMMARY_PROMPT = """
You are a world class legal assistant AI. Summarize the context, which is a packet of discovery documents for a legal matter.

# Please follow these instructions:

1. **Review the document carefully** to identify and understand the information relevant to litigation.
2. **Summarize the document contents** in a clear, detailed, and fact-focused manner.
3. **Retain key details** regarding the timeline of events, parties and/or service providers involved.
4. **Focus on** information that will help the attorney understand the case.
5. **On medical records** include the **diagnosis** the doctor listed from the visit.
6. **Do NOT include** any general statements or overall thoughts, or otherwise unnecessary text, simply consolidate the facts.
7. Whenever possible, favor **paragraph format** with over lists to ensure readability.

# PII Rules:

* NEVER include a person's DOB, physical address, or phone number.
* Include city and state of residence when possible. NEVER include the full address for people.

# For medical related documents, include the following information:

* The Summary of medical record/bills
* Provider and date range
* Complaints made (by date)
* Any Objective testing (X-rays, CT Scans, MRIs, EMG studies)
* Diagnoses/Findings (by date)
* Treatment Plan (by date)
* Orders made (medication, testing, referral to other providers) (by date)
* Overall – note any gaps in time in treatment
* If bills submitted, $ amounts:  billed, adjustments, paid by insurance, write-offs, outstanding amount
* Any prior/subsequent accidents/injuries noted

# **Important Notes:**

* Include contact information for any parties and/or service providers when available.
* Consider all parties related to the Plaintiff even if not directly involved in the legal matter. Examples can include, but are not limited to employer and primary care physician.
* Always highlight the following items:
    * Prior and/or subsequent accidents
    * Prior and/or subsequent injuries
* **Medical Records** should always be grouped by provider.
* **Multiple documents** may be included. If so, maintain clear separation in your summary.
* Remember to include all information on **monetary amounts**, **diagnoses**, and **treatments**.
"""

INTERMEDIATE_SUMMARY_PROMPT = """
You are a world class legal assistant AI. Generate a detailed summary of the excerpt from a larger packet of discovery documents.

Please follow these instructions:

1. **Review the document carefully** to identify and understand the information relevant to litigation.
2. **Summarize the document** in a clear, detailed, and fact-focused manner.
3. **Retain details** regarding the timeline of events, parties involved, and specific **service providers**.
4. **Focus on** information that will help the attorney understand the case.
5. **On medical records** include the **diagnosis** the doctor listed from the visit.
6. **Do NOT include** any general statements or overall thoughts, or otherwise unnecessary text, simply consolidate the facts.

# For medical related documents, include the following information:

* The Summary of medical record/bills
* Provider and date range
* Complaints made (by date)
* Any Objective testing (X-rays, CT Scans, MRIs, EMG studies)
* Diagnoses/Findings (by date)
* Treatment Plan (by date)
* Orders made (medication, testing, referral to other providers) (by date)
* Overall – note any gaps in time in treatment
* If bills submitted, $ amounts:  billed, adjustments, paid by insurance, write-offs, outstanding amount
* Any prior/subsequent accidents/injuries noted

**Important Notes:**

* Include contact information for any parties and/or service providers when available.
* Consider all parties related to the Plaintiff even if not directly involved in the legal matter. Examples can include, but are not limited to employer and primary care physician.
* Always highlight the following items:
    * Prior and/or subsequent accidents
    * Prior and/or subsequent injuries
* **Medical Records** should always be grouped by provider.
* **Multiple documents** may be included. If so, maintain clear separation in your summary.
* Remember to include all information on **monetary amounts**, **diagnoses**, and **treatments**.
"""

CONSOLIDATED_SUMMARY_PROMPT = """
You are a world class legal assistant AI. Consolidate the context, which is a set of intermediate summaries for a large packet of discovery documents.

# Please follow these instructions:

1. **Review the intermediate summaries carefully** to identify and understand the information relevant to litigation.
2. **Consolidated and the document** in a clear, detailed, and fact-focused manner.
3. **Retain details** regarding the timeline of events, parties involved, and specific **service providers**.
4. **Focus on** information that will help the attorney understand the case.
5. **On medical records** include the **diagnosis** the doctor listed from the visit.
6. **Do NOT include** any general statements or overall thoughts, or otherwise unnecessary text, simply consolidate the facts.
7. Favor **paragraph format** to ensure readability.

# PII Rules:

* NEVER include a person's DOB, physical address, or phone number.
* Include city and state of residence when possible. NEVER include the full address for people.

# For medical related documents, include the following information:

* The Summary of medical record/bills
* Provider and date range
* Complaints made (by date)
* Any Objective testing (X-rays, CT Scans, MRIs, EMG studies)
* Diagnoses/Findings (by date)
* Treatment Plan (by date)
* Orders made (medication, testing, referral to other providers) (by date)
* Overall – note any gaps in time in treatment
* If bills submitted, $ amounts:  billed, adjustments, paid by insurance, write-offs, outstanding amount
* Any prior/subsequent accidents/injuries noted

# **Important Notes:**

* Include contact information for any parties and/or service providers when available.
* Consider all parties related to the Plaintiff even if not directly involved in the legal matter. Examples can include, but are not limited to employer and primary care physician.
* Always highlight the following items:
    * Prior and/or subsequent accidents
    * Prior and/or subsequent injuries
* **Medical Records** should always be grouped by provider.
* **Multiple documents** may be included. If so, maintain clear separation in your summary.
* Remember to include all information on **monetary amounts**, **diagnoses**, and **treatments**.
"""

TITLE_DESCRIPTION_PROMPT = """
# Please follow these instructions:
1. Titles should be concise, specific, and avoid filler words or general statements.
2. Descriptions should be concise, fact-focused, and information dense.
3. Make every word count!
"""

# Define the state object
@dataclass
class DocumentSummaryState:
    """
    State object for the Document Summary Graph.
    
    This state is passed between nodes to maintain context throughout the graph execution.
    
    Attributes:
        documents: List of ConversionResult documents to process
        processed_documents: List of processed documents with chunks
        current_doc_index: Index of the current document being processed
        intermediate_summaries: List of intermediate summaries for large documents
        document_summaries: List of completed document summaries
        document_title: Current document title
        document_description: Current document description
        usages: List of usage statistics from all agent calls
    """
    documents: List[ConversionResult] = field(default_factory=list)
    processed_documents: List[ProcessedDocument] = field(default_factory=list)
    current_doc_index: int = 0
    intermediate_summaries: Dict[int, List[str]] = field(default_factory=dict)
    document_summaries: List[ContextSummary] = field(default_factory=list)
    document_title: str = ""
    document_description: str = ""
    usages: List[Usage] = field(default_factory=list)
    
    # Settings
    chunk_size: int = 20000
    add_labels: bool = True
    cap_multiplier: float = 1.0


# Define dependencies
@dataclass
class DocumentSummaryDeps:
    """
    Dependencies for the Document Summary Graph.
    
    This class holds agents and other resources that will be injected into the graph.
    
    Attributes:
        document_summary_agent: Agent to summarize documents
        intermediate_summary_agent: Agent to summarize intermediate document excerpts
        consolidated_summary_agent: Agent to consolidate multiple summaries
        title_description_agent: Agent to generate titles and descriptions
    """
    document_summary_agent: Agent
    intermediate_summary_agent: Agent 
    consolidated_summary_agent: Agent
    title_description_agent: Agent


# Create agents with proper types
document_summary_agent = Agent[None, str](
    model="openai:gpt-4o",
    result_type=str,
    retries=3,
    system_prompt=DOCUMENT_SUMMARY_PROMPT,
)

intermediate_summary_agent = Agent[None, str](
    model="openai:gpt-4o",
    result_type=str,
    retries=3,
    system_prompt=INTERMEDIATE_SUMMARY_PROMPT,
)

consolidated_summary_agent = Agent[None, str](
    model="openai:gpt-4o",
    result_type=str,
    retries=3,
    system_prompt=CONSOLIDATED_SUMMARY_PROMPT,
)

title_description_agent = Agent[None, TitleAndDescriptionResult](
    model="openai:gpt-4o",
    result_type=TitleAndDescriptionResult,
    retries=3,
    system_prompt=TITLE_DESCRIPTION_PROMPT,
)


# Define nodes for the graph
@dataclass
class Initialize(BaseNode[DocumentSummaryState, DocumentSummaryDeps]):
    """
    Initial node in the Document Summary Graph.
    
    This node sets up the state and transitions to the PrepareDocuments node.
    """
    
    async def run(
        self, ctx: GraphRunContext[DocumentSummaryState, DocumentSummaryDeps]
    ) -> "ProcessDocument":
        """
        Initialize the document summarization process.
        
        Args:
            ctx: Graph run context with state and dependencies.
            
        Returns:
            ProcessDocument node to process documents for processing.
        """
        # Process all documents to create chunks
        ctx.state.processed_documents = prepare_processed_document_chunks(
            ctx.state.documents, 
            ctx.state.chunk_size, 
            ctx.state.cap_multiplier
        )
        
        return ProcessDocument()


@dataclass
class ProcessDocument(BaseNode[DocumentSummaryState, DocumentSummaryDeps]):
    """
    Node to process a single document.
    
    This node determines whether to process the document directly or in chunks.
    """
    
    async def run(
        self, ctx: GraphRunContext[DocumentSummaryState, DocumentSummaryDeps]
    ) -> Union["SummarizeDocument", "ProcessDocumentChunks", "Finalize"]:
        """
        Process the current document by determining the appropriate summarization approach.
        
        Args:
            ctx: Graph run context with state and dependencies.
            
        Returns:
            SummarizeDocument node for small documents or
            ProcessDocumentChunks node for large documents.
        """
        # Check if we have more documents to process
        if ctx.state.current_doc_index >= len(ctx.state.processed_documents):
            return Finalize()
        
        # Get the current document
        document = ctx.state.processed_documents[ctx.state.current_doc_index]
        
        # Check if the document is small enough to process directly
        if document.token_count <= ctx.state.chunk_size:
            return SummarizeDocument()
        
        # Document is too large, process in chunks
        return ProcessDocumentChunks()


@dataclass
class SummarizeDocument(BaseNode[DocumentSummaryState, DocumentSummaryDeps]):
    """
    Node to summarize a small document directly.
    
    This node processes a document that fits within the token limit.
    """
    
    async def run(
        self, ctx: GraphRunContext[DocumentSummaryState, DocumentSummaryDeps]
    ) -> Union["GenerateTitleDescription", "NextDocument"]:
        """
        Summarize the current document directly.
        
        Args:
            ctx: Graph run context with state and dependencies.
            
        Returns:
            GenerateTitleDescription node if labels are needed, 
            otherwise NextDocument node.
        """
        try:
            # Get the current document
            document = ctx.state.processed_documents[ctx.state.current_doc_index]
            
            # Run the agent
            result = await ctx.deps.document_summary_agent.run(document.processed_text)
            ctx.state.usages.append(result.usage())
            
            # Store the summary
            ctx.state.document_summary = result.data
            
            # Check if we need to generate title and description
            if ctx.state.add_labels:
                return GenerateTitleDescription()
            
            # Otherwise, create the context summary and move to the next document
            exhibit_number = ctx.state.current_doc_index + 1
            document_name = ctx.state.documents[ctx.state.current_doc_index].name
            
            ctx.state.document_summaries.append(
                ContextSummary(
                    exhibit_number=exhibit_number,
                    file_name=document_name,
                    summary=ctx.state.document_summary,
                    document_title="",
                    document_description="",
                    usages=ctx.state.usages
                )
            )
            
            return NextDocument()
            
        except Exception as e:
            # Handle errors
            error_message = f"Error processing document: {ctx.state.documents[ctx.state.current_doc_index].name}\nException: {str(e)}"
            print(error_message)
            
            # Create error summary
            exhibit_number = ctx.state.current_doc_index + 1
            document_name = ctx.state.documents[ctx.state.current_doc_index].name
            
            ctx.state.document_summaries.append(
                ContextSummary(
                    exhibit_number=exhibit_number,
                    file_name=document_name,
                    summary=error_message,
                    document_title="Unknown due to processing error",
                    document_description="Unknown due to processing error",
                    usages=[]
                )
            )
            
            return NextDocument()


@dataclass
class ProcessDocumentChunks(BaseNode[DocumentSummaryState, DocumentSummaryDeps]):
    """
    Node to process a large document in chunks.
    
    This node sets up chunk processing for documents that exceed the token limit.
    """
    
    async def run(
        self, ctx: GraphRunContext[DocumentSummaryState, DocumentSummaryDeps]
    ) -> "SummarizeChunks":
        """
        Set up processing for document chunks.
        
        Args:
            ctx: Graph run context with state and dependencies.
            
        Returns:
            SummarizeChunks node to process all chunks.
        """
        # Get the current document
        document = ctx.state.processed_documents[ctx.state.current_doc_index]
        
        # Initialize intermediate summaries list for this document
        ctx.state.intermediate_summaries[ctx.state.current_doc_index] = []
        
        # Initialize list of chunks to process
        ctx.state.chunks_to_process = [chunk.text for chunk in document.text_chunks]
        ctx.state.current_chunk_index = 0
        
        return SummarizeChunks()


@dataclass
class SummarizeChunks(BaseNode[DocumentSummaryState, DocumentSummaryDeps]):
    """
    Node to summarize individual document chunks.
    
    This node processes each chunk of a large document and collects intermediate summaries.
    """
    
    async def run(
        self, ctx: GraphRunContext[DocumentSummaryState, DocumentSummaryDeps]
    ) -> Union["SummarizeChunks", "ConsolidateSummaries"]:
        """
        Process the current chunk and move to the next one.
        
        Args:
            ctx: Graph run context with state and dependencies.
            
        Returns:
            SummarizeChunks node if more chunks to process,
            otherwise ConsolidateSummaries node.
        """
        # Check if we have more chunks to process
        if ctx.state.current_chunk_index < len(ctx.state.chunks_to_process):
            try:
                # Get the current chunk
                chunk_text = ctx.state.chunks_to_process[ctx.state.current_chunk_index]
                
                # Summarize this chunk
                result = await ctx.deps.intermediate_summary_agent.run(chunk_text)
                ctx.state.usages.append(result.usage())
                
                # Store the intermediate summary
                ctx.state.intermediate_summaries[ctx.state.current_doc_index].append(result.data)
                
            except Exception as e:
                # Handle errors
                error_message = f"Error processing chunk {ctx.state.current_chunk_index} of document {ctx.state.documents[ctx.state.current_doc_index].name}\nException: {str(e)}"
                print(error_message)
                
                # Add an error message as the summary
                ctx.state.intermediate_summaries[ctx.state.current_doc_index].append(
                    f"Error summarizing chunk {ctx.state.current_chunk_index}: {str(e)}"
                )
            
            # Move to next chunk
            ctx.state.current_chunk_index += 1
            return SummarizeChunks()
        
        # All chunks processed, move to consolidation
        return ConsolidateSummaries()


@dataclass
class ConsolidateSummaries(BaseNode[DocumentSummaryState, DocumentSummaryDeps]):
    """
    Node to consolidate multiple intermediate summaries.
    
    This node combines all intermediate summaries into a single summary.
    """
    
    async def run(
        self, ctx: GraphRunContext[DocumentSummaryState, DocumentSummaryDeps]
    ) -> Union["GenerateTitleDescription", "NextDocument"]:
        """
        Consolidate all intermediate summaries for the current document.
        
        Args:
            ctx: Graph run context with state and dependencies.
            
        Returns:
            GenerateTitleDescription node if labels are needed,
            otherwise NextDocument node.
        """
        try:
            # Get all intermediate summaries for this document
            intermediate_summaries = ctx.state.intermediate_summaries[ctx.state.current_doc_index]
            
            # Join summaries with separators
            combined_summaries = "\n\n---\n\n".join(intermediate_summaries)
            
            # Check if the combined summaries are too large
            tokens = count_tokens(combined_summaries)
            
            if tokens > ctx.state.chunk_size:
                # If too large, do multi-step consolidation
                batch_size = max(2, len(intermediate_summaries) // 2)
                batches = [
                    intermediate_summaries[i:i+batch_size]
                    for i in range(0, len(intermediate_summaries), batch_size)
                ]
                
                # First level consolidation
                first_level_results = []
                for batch in batches:
                    batch_text = "\n\n---\n\n".join(batch)
                    result = await ctx.deps.consolidated_summary_agent.run(batch_text)
                    ctx.state.usages.append(result.usage())
                    first_level_results.append(result.data)
                
                # Second level consolidation
                combined_text = "\n\n---\n\n".join(first_level_results)
                result = await ctx.deps.consolidated_summary_agent.run(combined_text)
                ctx.state.usages.append(result.usage())
                ctx.state.document_summary = result.data
            else:
                # Single consolidation step
                result = await ctx.deps.consolidated_summary_agent.run(combined_summaries)
                ctx.state.usages.append(result.usage())
                ctx.state.document_summary = result.data
            
            # Check if we need to generate title and description
            if ctx.state.add_labels:
                return GenerateTitleDescription()
            
            # Otherwise, create the context summary and move to the next document
            exhibit_number = ctx.state.current_doc_index + 1
            document_name = ctx.state.documents[ctx.state.current_doc_index].name
            
            ctx.state.document_summaries.append(
                ContextSummary(
                    exhibit_number=exhibit_number,
                    file_name=document_name,
                    summary=ctx.state.document_summary,
                    document_title="",
                    document_description="",
                    usages=ctx.state.usages
                )
            )
            
            return NextDocument()
            
        except Exception as e:
            # Handle errors
            error_message = f"Error consolidating summaries for document: {ctx.state.documents[ctx.state.current_doc_index].name}\nException: {str(e)}"
            print(error_message)
            
            # Create error summary
            exhibit_number = ctx.state.current_doc_index + 1
            document_name = ctx.state.documents[ctx.state.current_doc_index].name
            
            ctx.state.document_summaries.append(
                ContextSummary(
                    exhibit_number=exhibit_number,
                    file_name=document_name,
                    summary=error_message,
                    document_title="Unknown due to processing error",
                    document_description="Unknown due to processing error",
                    usages=[]
                )
            )
            
            return NextDocument()


@dataclass
class GenerateTitleDescription(BaseNode[DocumentSummaryState, DocumentSummaryDeps]):
    """
    Node to generate a title and description for a document.
    
    This node processes the document summary to create a title and description.
    """
    
    async def run(
        self, ctx: GraphRunContext[DocumentSummaryState, DocumentSummaryDeps]
    ) -> "NextDocument":
        """
        Generate a title and description for the current document.
        
        Args:
            ctx: Graph run context with state and dependencies.
            
        Returns:
            NextDocument node to process the next document.
        """
        try:
            # Get the document summary
            summary = ctx.state.document_summary
            
            # Generate title and description
            result = await ctx.deps.title_description_agent.run(summary)
            ctx.state.usages.append(result.usage())
            
            # Store the title and description
            ctx.state.document_title = result.data.title
            ctx.state.document_description = result.data.description
            
        except Exception as e:
            # Handle errors
            error_message = f"Error generating title/description for document: {ctx.state.documents[ctx.state.current_doc_index].name}\nException: {str(e)}"
            print(error_message)
            
            # Set default values
            ctx.state.document_title = "Error generating title"
            ctx.state.document_description = "Error generating description"
        
        # Create the context summary
        exhibit_number = ctx.state.current_doc_index + 1
        document_name = ctx.state.documents[ctx.state.current_doc_index].name
        
        ctx.state.document_summaries.append(
            ContextSummary(
                exhibit_number=exhibit_number,
                file_name=document_name,
                summary=ctx.state.document_summary,
                document_title=ctx.state.document_title,
                document_description=ctx.state.document_description,
                usages=ctx.state.usages
            )
        )
        
        return NextDocument()


@dataclass
class NextDocument(BaseNode[DocumentSummaryState, DocumentSummaryDeps]):
    """
    Node to move to the next document.
    
    This node increments the document index and transitions back to
    the document preparation stage.
    """
    
    async def run(
        self, ctx: GraphRunContext[DocumentSummaryState, DocumentSummaryDeps]
    ) -> "ProcessDocument":
        """
        Move to the next document for processing.
        
        Args:
            ctx: Graph run context with state and dependencies.
            
        Returns:
            ProcessDocument node to check for more documents to process.
        """
        # Move to next document
        ctx.state.current_doc_index += 1
        
        # Clear per-document state
        if hasattr(ctx.state, "document_summary"):
            delattr(ctx.state, "document_summary")
        
        if hasattr(ctx.state, "chunks_to_process"):
            delattr(ctx.state, "chunks_to_process")
            
        if hasattr(ctx.state, "current_chunk_index"):
            delattr(ctx.state, "current_chunk_index")
        
        ctx.state.document_title = ""
        ctx.state.document_description = ""
        ctx.state.usages = []
        
        # Return to start of document processing
        return ProcessDocument()


@dataclass
class Finalize(BaseNode[DocumentSummaryState, DocumentSummaryDeps]):
    """
    Final node in the Document Summary Graph.
    
    This node combines all document summaries into a final result.
    """
    
    async def run(
        self, ctx: GraphRunContext[DocumentSummaryState, DocumentSummaryDeps]
    ) -> End:
        """
        Finalize the document summarization process.
        
        Args:
            ctx: Graph run context with state and dependencies.
            
        Returns:
            None to end the graph execution.
        """
        # Create the final ContextSummaries object
        ctx.state.final_result = ContextSummaries(summaries=ctx.state.document_summaries)
        
        return End()


# Create the graph
document_summary_graph = Graph(
    nodes=[
        Initialize,
        ProcessDocument,
        SummarizeDocument,
        ProcessDocumentChunks,
        SummarizeChunks,
        ConsolidateSummaries,
        GenerateTitleDescription,
        NextDocument,
        Finalize
    ],
    state_type=DocumentSummaryState,
)

# Visualization of the graph
document_summary_graph.mermaid_save(
    "document_summary_graph.png",
    start_node=ProcessDocument,
    direction="LR",
    highlighted_nodes=[Finalize],
)


# Function to run the graph
async def process_documents_graph_async(
    documents: List[ConversionResult],
    chunk_size: int = 20000,
    add_labels: bool = True,
    cap_multiplier: float = 1.0,
) -> ContextSummaries:
    """
    Asynchronously process documents using the document summary graph.
    
    This function executes the document summary graph to process a list of documents
    and generate summaries for each.
    
    Args:
        documents: List of ConversionResult documents to process.
        chunk_size: Maximum chunk size for processing.
        add_labels: Whether to generate title and description.
        cap_multiplier: Multiplier for chunk size cap.
        
    Returns:
        ContextSummaries object containing all processed document summaries.
        
    Example:
        ```python
        result = await process_documents_graph_async(
            documents=[doc1, doc2, doc3],
            chunk_size=20000,
            add_labels=True
        )
        print(result.summaries[0].summary)
        ```
    """
    # Initialize state
    state = DocumentSummaryState(
        documents=documents,
        chunk_size=chunk_size,
        add_labels=add_labels,
        cap_multiplier=cap_multiplier
    )
    
    # Initialize dependencies
    deps = DocumentSummaryDeps(
        document_summary_agent=document_summary_agent,
        intermediate_summary_agent=intermediate_summary_agent,
        consolidated_summary_agent=consolidated_summary_agent,
        title_description_agent=title_description_agent
    )
    
    # Run the graph
    await document_summary_graph.run(Initialize(), state=state, deps=deps)
    
    # Return the final result
    return state.final_result


def run_documents_summary_graph(
    documents: List[ConversionResult],
    chunk_size: int = 20000,
    add_labels: bool = True,
    cap_multiplier: float = 1.0,
) -> ContextSummaries:
    """
    Synchronous wrapper for process_documents_graph_async.
    
    Args:
        documents: List of ConversionResult documents to process.
        chunk_size: Maximum chunk size for processing.
        add_labels: Whether to generate title and description.
        cap_multiplier: Multiplier for chunk size cap.
        
    Returns:
        ContextSummaries object containing all processed document summaries.
    """
    import asyncio
    
    return asyncio.run(
        process_documents_graph_async(
            documents=documents,
            chunk_size=chunk_size,
            add_labels=add_labels,
            cap_multiplier=cap_multiplier
        )
    ) 