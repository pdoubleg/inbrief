"""Base classes for prompt management using Pydantic"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai.format_as_xml import format_as_xml


class Example(BaseModel):
    """Model for prompt examples with input and expected output"""

    input: str = "..."
    response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BasePrompt(BaseModel):
    """Base model for all prompts"""

    name: str
    template: str
    examples: List[Example] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def format_with_examples(self) -> str:
        """Format the prompt template with given kwargs"""
        if not self.examples:
            raise ValueError("No examples provided for prompt")
        formatted_template = self.template.replace(
            "{{EXAMPLES}}", format_as_xml(self.examples)
        )
        return formatted_template

    def format_with_draft_input(self, draft_input: str) -> str:
        """Format the prompt template with a draft summary"""
        if not draft_input:
            raise ValueError("No draft input provided")
        formatted_template = self.template.replace("{DRAFT_INPUT}", draft_input)
        return formatted_template


class PromptLibrary(BaseModel):
    """Container for managing multiple prompts"""

    prompts: List[BasePrompt]

    def get_prompt(self, name: str) -> Optional[BasePrompt]:
        """Retrieve a prompt by name"""
        for prompt in self.prompts:
            if prompt.name == name:
                return prompt
        return None

    def add_prompt(self, prompt: BasePrompt) -> None:
        """Add a prompt to the library"""
        self.prompts.append(prompt)
