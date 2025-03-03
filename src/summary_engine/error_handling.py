"""Error handling utilities for the SummaryEngine.

This module provides decorators and functions for consistent error handling,
logging, and reporting throughout the SummaryEngine pipeline.
"""

import functools
import inspect
import time
import traceback
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    cast,
    get_origin,
    get_args,
    Union,
    get_type_hints,
)
from pydantic import BaseModel, create_model

from src.utils import log_exception

# Type variables for better type hinting with decorators
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


def friendly_error(error: Exception, message: str) -> str:
    """Creates a user-friendly error message with traceback.

    Args:
        error: The exception that was raised
        message: A user-friendly message explaining the error

    Returns:
        A formatted error message with traceback
    """
    trace = traceback.format_exc()
    return f"{message}\n\nError: {str(error)}\n\nTraceback:\n{trace}"


# A generic type for functions
# F = Callable[..., Any]


def create_error_model(model: Any) -> Any:
    """Convert all fields of a Pydantic model to Optional and adds error fields."""
    if not issubclass(model, BaseModel):
        raise ValueError("model must be a subclass of BaseModel")

    fields = {}
    for field_name, field in model.model_fields.items():
        annotation = field.annotation
        # Check if field is already optional
        if get_origin(annotation) is not Union or type(None) not in get_args(
            annotation
        ):
            fields[field_name] = (Optional[field.annotation], None)
        else:
            fields[field_name] = (field.annotation, None)

    NewModel = create_model(
        model.__name__ + "Optional",
        __base__=model,
        error_message=(Optional[str], None),
        error_traceback=(Optional[str], None),
        **fields,
    )
    return NewModel


def handle_llm_errors(
    task_name: str, error_category: str, default_message: Optional[str] = None
):
    """
    Decorator for handling errors in LLM operations, supporting both sync and async functions.

    This decorator catches exceptions, logs them, and returns a default value
    constructed from the function's annotated return type (which must be a subclass of BaseModel
    or a string).

    Args:
        task_name: Name of the task being performed (for logging).
        error_category: Category of the error (for classification).
        default_message: Default error message if none is provided.

    Returns:
        A decorator function.
    """

    def decorator(func: F) -> F:
        # Get type hints from the function to determine the return type
        hints = get_type_hints(func)
        return_type = hints.get("return", None)

        # Ensure that the return type is either a subclass of pydantic.BaseModel or str
        is_str_return = isinstance(return_type, type) and issubclass(return_type, str)
        if not return_type or not (
            is_str_return
            or (isinstance(return_type, type) and issubclass(return_type, BaseModel))
        ):
            raise TypeError(
                "The wrapped function must have a return type annotation that is either a subclass of pydantic.BaseModel or str"
            )

        # Async wrapper: used if the function is a coroutine function.
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(self, *args: Any, **kwargs: Any) -> Any:
                try:
                    # Await the coroutine
                    result = await func(self, *args, **kwargs)
                    return result

                except Exception as e:
                    error_message = (
                        f"{e}: {default_message}, "
                        f"\nCalling {func.__name__} with args: {args}, kwargs: {kwargs} "
                        f"\nStack: {traceback.format_exc()}"
                    )
                    log_exception(
                        job_id=getattr(self, "job_id", None),
                        model="OpenAIModel_GPT_4O",
                        module=task_name,
                        error_message=error_message,
                        error_category=error_category,
                        error_traceback=str(e),
                    )
                    if is_str_return:
                        return f"An error occurred: {error_message}. Details: {str(e)}"
                    else:
                        error_model = create_error_model(return_type)
                        error_info = {
                            "error_message": error_message,
                            "error_traceback": str(e),
                        }
                        return error_model.model_validate(error_info)

            return cast(F, async_wrapper)
        else:
            # Synchronous wrapper
            @functools.wraps(func)
            def sync_wrapper(self, *args: Any, **kwargs: Any) -> Any:
                try:
                    result = func(self, *args, **kwargs)
                    return result
                except Exception as e:
                    error_message = (
                        f"{e}: {default_message}, "
                        f"\nCalling {func.__name__} with args: {args}, kwargs: {kwargs} "
                        f"\nStack: {traceback.format_exc()}"
                    )
                    log_exception(
                        job_id=getattr(self, "job_id", None),
                        model="OpenAIModel_GPT_4O",
                        module=task_name,
                        error_message=error_message,
                        error_category=error_category,
                        error_traceback=str(e),
                    )
                    if is_str_return:
                        return f"An error occurred: {error_message}. Details: {str(e)}"
                    else:
                        error_model = create_error_model(return_type)
                        error_info = {
                            "error_message": error_message,
                            "error_traceback": str(e),
                        }
                        return error_model.model_validate(error_info)

            return cast(F, sync_wrapper)

    return decorator


def log_execution_time(logger: Optional[Callable[[str], None]] = None):
    """Decorator for logging execution time of functions.

    Args:
        logger: Function to use for logging (defaults to print)

    Returns:
        A decorator function

    Example:
        @log_execution_time()
        def expensive_operation(data):
            # Time-consuming operation
            return result
    """
    log_func = logger or print

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            log_func(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
            return result

        return cast(F, wrapper)

    return decorator
