import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generator, Callable

from openai import OpenAI

from ..cache import Cache
from ..config import cfg
from ..function import get_openai_schemas
from ..role import SystemRole
from .handler import Handler


base_url = cfg.get("API_BASE_URL")
use_litellm = cfg.get("USE_LITELLM") == "true"
additional_kwargs = {
    "timeout": int(cfg.get("REQUEST_TIMEOUT")),
    "api_key": cfg.get("OPENAI_API_KEY"),
    "base_url": None if base_url == "default" else base_url,
}


client = OpenAI(**additional_kwargs)  # type: ignore
completion: Callable[..., Any] = client.chat.completions.create
additional_kwargs = {}


@dataclass(frozen=True)
class AgentABC(Handler, ABC):
    class Config:
        model: str = 'gpt-4o'
        name: str
        role: str
        functions: list[str] | None
        temperature: float = 0.7
        top_p: float = 1.0
        caching: bool = False
        tool_choice: str = "auto"

    @classmethod
    @property
    def config(cls):
        return cls.Config()

    def __post_init__(self) -> None:
        super().__init__(self.system_role, markdown=True)

    @property
    @abstractmethod
    def state(self) -> dict[str, Any]:
        raise NotImplementedError("Must be implemented in subclass")

    @property
    def system_role(self) -> SystemRole:
        return SystemRole(self.config.name, self.config.role, self.state)

    @abstractmethod
    def make_messages(self, prompt: str) -> list[dict[str, str]]:
        """
        Abstract method to create messages for the specific role and functions.
        """
        raise NotImplementedError("Must be implemented in subclass")

    @Handler.cache
    def get_completion(
        self,
        model: str,
        temperature: float,
        top_p: float,
        messages: list[dict[str, Any]],
        functions: list[dict[str, Any]] | None,
    ) -> Generator[str, None, None]:
        if not (functions is None or self.config.functions is None):
            functions = [
                f for f in functions if f["function"]["name"] in self.config.functions
            ]
        additional_kwargs.update(
            tool_choice=self.config.tool_choice,
            tools=functions,
            parallel_tool_calls=False
        )

        return self.complete(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            functions=functions,
            **additional_kwargs,
        )

    def handle(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = Config.temperature,
        top_p: float = Config.top_p,
        caching: bool = Config.caching,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Handles a completion with a specific role and its corresponding functions.
        Returns plain string response to the caller.
        """
        model = self.config.model if model is None else model
        functions = get_openai_schemas()
        messages = self.make_messages(prompt.strip())
        return self.get_completion(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            functions=functions,
            caching=caching,
            **kwargs,
        )
