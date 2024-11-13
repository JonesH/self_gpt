from dataclasses import dataclass

from .agent_abc import AgentABC
from typing import Any


class DefaultAgent(AgentABC):
    class Config(AgentABC.Config):
        model: str = "gpt-4"
        name: str = "DefaultAgent"
        role: str = "Default Assistant for managing user queries."
        temperature: float = 0.7
        top_p: float = 1.0
        functions: list[str] | None = None
        caching: bool = True

    @property
    def state(self) -> dict[str, Any]:
        return {
            "status": "active",
            "description": "Default agent for managing queries and communication.",
        }

    def make_messages(self, prompt: str) -> list[dict[str, str]]:
        """
        Creates a message format compatible with OpenAI chat completions.
        """
        user_message = {"role": "user", "content": prompt}
        self.conversation_history.append(user_message)
        return self.conversation_history


# Register this agent as the default interaction point.
default_agent = DefaultAgent()
