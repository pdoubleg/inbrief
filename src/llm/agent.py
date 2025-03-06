from dataclasses import dataclass
from typing import AsyncIterator, Generic, TypeVar, List

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.result import RunResult
from pydantic_ai.usage import Usage

@dataclass
class AgentDeps:
    usages: List[Usage]



DepsT = TypeVar("DepsT", bound=AgentDeps)


@dataclass
class AgentRunner(Generic[DepsT]):
    """
    Class which wraps an Agent to facilitate agent execution by:
    - Maintaining and managing message history
    - Providing dependencies
    """

    agent: Agent[DepsT, str]
    deps: DepsT
    message_history: list[ModelMessage] | None = None

    def clear_message_history(self) -> None:
        """Clear the message history."""
        self.message_history = None

    def run_sync(self, query: str) -> RunResult[str]:
        """Run a query and automatically provide dependencies and message history."""
        response = self.agent.run_sync(query, deps=self.deps, message_history=self.message_history)
        self.message_history = response.all_messages()
        return response

    async def run(self, query: str) -> RunResult[str]:
        """Run a query and automatically provide dependencies and message history."""
        response = await self.agent.run(query, deps=self.deps, message_history=self.message_history)
        self.message_history = response.all_messages()
        return response

    async def run_stream(self, query: str) -> AsyncIterator[str]:
        """Run a query and automatically provide dependencies and message history."""
        async with self.agent.run_stream(query, deps=self.deps, message_history=self.message_history) as result:
            async for message in result.stream_text():
                yield message

            self.message_history = result.all_messages()


def get_agent_runner(model: Model, deps: DepsT) -> AgentRunner[DepsT]:
    agent = Agent(
        model=model,
        deps_type=type(deps),
        system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    )
    return AgentRunner(agent, deps=deps)


