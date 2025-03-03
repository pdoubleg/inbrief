import math
import tiktoken
from typing import List, Union
import logging
import traceback

from src.models import Page, TextChunk, ProcessedDocument, ConversionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using tiktoken.

    Args:
        text: The text to count tokens for.

    Returns:
        The number of tokens in the text.
    """
    if not text:
        return 0

    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}. Using fallback method.")
        # Fallback to an approximation if tiktoken fails
        return len(text) // 4  # Rough approximation


def has_common_column_name(list_a: List[str], list_b: List[str]) -> bool:
    """
    Check if two lists have any common elements.

    Args:
        list_a: First list of strings.
        list_b: Second list of strings.

    Returns:
        True if the lists have at least one common element, False otherwise.
    """
    return not set(list_a).isdisjoint(list_b)


def is_same_table(page_1: Page, page_2: Page) -> bool:
    """
    Check if two pages have the same table based on headers.

    Args:
        page_1: First page object.
        page_2: Second page object.

    Returns:
        True if the pages have the same table, False otherwise.
    """
    headers_1 = page_1.table_headers
    headers_2 = page_2.table_headers
    return has_common_column_name(headers_1, headers_2)


def create_dynamic_text_chunks(
    pages: Union[List[Page], List[str]],
    max_chunk_size: int = 20000,
    cap_multiplier: float = 1.0,
) -> List[TextChunk]:
    """
    Splits a list of pages into chunks of text with approximately equal token distribution
    while allowing additional token 'buffer' for multi-page tables.

    Args:
        pages: The list of Page objects or strings to be chunked.
        max_chunk_size: The maximum number of tokens allowed per chunk. Default: 20000.
        cap_multiplier: Multiplier for the target chunk size to allow
                        larger chunks when necessary for multi-page tables. Default: 1.0.

    Returns:
        A list of TextChunk objects representing the chunks of text.
    """
    # If pages is a list of strings, convert to a list of Page objects
    if isinstance(pages[0], str):
        pages = [
            Page(page_number=i, text=page) for i, page in enumerate(pages, start=1)
        ]

    # Get total tokens
    all_pages = "\n\n".join([page.text.strip() for page in pages if not page.skipped])
    total_tokens = count_tokens(all_pages)

    # If total tokens is less than max_chunk_size, return a single chunk
    if total_tokens <= max_chunk_size:
        return [
            TextChunk(
                text=all_pages,
                start_page=1,
                end_page=len(pages),
                token_count=total_tokens,
            )
        ]

    # Estimate number of chunks
    N = max(math.ceil(total_tokens / max_chunk_size), 1)

    # Initialize variables
    chunks = []
    current_tokens = 0
    current_text_list = []
    start_page_for_chunk = None
    remaining_tokens = total_tokens
    remaining_chunks = N

    for page_number, page in enumerate(pages, start=1):
        if page.skipped:
            continue

        page_tokens = count_tokens(page.text)

        if start_page_for_chunk is None:
            start_page_for_chunk = page_number

        # Determine the target chunk size based on the remaining tokens and chunks
        if remaining_chunks > 1:
            target_chunk_size = math.ceil(remaining_tokens / remaining_chunks)
        else:
            target_chunk_size = remaining_tokens

        # Adjust target_chunk_size to respect the max_chunk_size
        target_chunk_size = min(max_chunk_size, target_chunk_size)

        # Calculate the dynamic cap
        dynamic_cap = int(target_chunk_size * cap_multiplier)

        # Check if adding the current page exceeds the target chunk size
        if current_tokens + page_tokens > target_chunk_size:
            # If it is part of the same table, allow up to the dynamic cap
            if current_tokens + page_tokens <= dynamic_cap and (
                page_number > 1 and is_same_table(pages[page_number - 2], page)
            ):
                # Continue adding as it's part of the same table
                current_text_list.append(page.text)
                current_tokens += page_tokens
            elif current_tokens > 0:
                # finalize current chunk
                chunks.append(
                    TextChunk(
                        text="\n".join(current_text_list),
                        start_page=start_page_for_chunk,
                        end_page=page_number - 1,
                        token_count=current_tokens,
                    )
                )

                # start a new chunk
                current_text_list = [page.text]
                current_tokens = page_tokens
                start_page_for_chunk = page_number

                # Update remaining tokens and chunks
                remaining_tokens -= current_tokens
                remaining_chunks -= 1
        else:
            current_text_list.append(page.text)
            current_tokens += page_tokens

    # Add the last chunk
    if current_text_list:
        chunks.append(
            TextChunk(
                text="\n".join(current_text_list),
                start_page=start_page_for_chunk,
                end_page=page_number,
                token_count=current_tokens,
            )
        )

    return chunks


def prepare_processed_document_chunks(
    docs: List[ConversionResult], chunk_size: int = 20000, cap_multiplier: float = 1.0
) -> List[ProcessedDocument]:
    """
    Chunks documents into parts where each part's token count does not exceed chunk_size.

    Args:
        docs: A list of documents to be processed.
        chunk_size: The maximum number of tokens each chunk can have.
        cap_multiplier: Multiplier to allow larger chunks when necessary for multi-page tables.

    Returns:
        A list of processed documents with text chunks.
    """
    processed_docs = []

    for doc in docs:
        chunks = create_dynamic_text_chunks(
            pages=doc.pages,
            max_chunk_size=chunk_size,
            cap_multiplier=cap_multiplier,
        )

        text = doc.text_trimmed
        tokens = count_tokens(text)

        processed_doc = ProcessedDocument(
            processed_text=text,
            token_count=tokens,
            text_chunks=chunks,
            **doc.model_dump(),
        )

        processed_docs.append(processed_doc)

    return processed_docs


def friendly_error(e: Exception, error_message: str = None) -> str:
    """
    Create a friendly error message with traceback information.

    Args:
        e: The exception that was raised.
        error_message: Optional custom error message.

    Returns:
        A string containing the error message and traceback.
    """
    error_type = type(e).__name__
    error_msg = str(e)
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    tb_string = "".join(tb)

    full_error = f"{error_message or 'Error occurred'}: {error_type}: {error_msg}\n\nTraceback:\n{tb_string}"
    return full_error


def log_exception(
    job_id: str = None,
    model: str = None,
    error_message: str = None,
    error_category: str = None,
    error_traceback: str = None,
    module: str = None,
) -> None:
    """
    Log an exception to logger and optionally to a monitoring system.

    Args:
        job_id: Optional job ID for tracking.
        model: Optional model name.
        error_message: Error message.
        error_category: Category of the error.
        error_traceback: Error traceback.
    """
    # Log to logger
    logger.error(
        f"Error in job {job_id or 'unknown'}, model {model or 'unknown'}, "
        f"category {error_category or 'unknown'}: {error_message or 'Unknown error'}"
    )

    if error_traceback:
        logger.debug(f"Traceback: {error_traceback}")

    # Additional logging to monitoring systems can be added here if needed
