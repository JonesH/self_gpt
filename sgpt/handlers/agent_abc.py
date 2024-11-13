import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator, Callable

from openai import OpenAI

from ..config import cfg
from ..function import get_openai_schemas, get_function
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


@dataclass
class AgentABC(Handler, ABC):
    conversation_history: list[dict[str, Any]] = field(default_factory=list, init=False)

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
        self.conversation_history.append({"role": "system", "content": self.config.role})

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

    def get_function_schema(self) -> dict[str, Any]:
        """
        Returns a schema that defines the input parameters and output format.
        This schema can be used for integration with other agents or OpenAI functions.
        """
        return {
            "name": self.config.name,
            "description": f"Agent for {self.config.role}",
            "parameters": {
                "type": "object",
                "properties": {
                    "path_str": {
                        "type": "string",
                        "description": "The prompt or message to process."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt or message to process."
                    },
                    # Add additional parameters as required by your agent
                },
                "required": ["path_str", "prompt"],
            }
        }

    def __call__(self, input: str) -> str:
        """
        Makes the agent callable as a function.
        It processes the given input and returns the result.
        """
        messages = self.make_messages(input.strip())
        generator = self.get_completion(
            model=self.config.model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            messages=messages,
            functions=None,  # Assuming no additional functions for simplicity
            caching=self.config.caching,
        )
        return self._process_stream(generator)

    def _process_stream(self, generator: Generator[str, None, None]) -> str:
        """
        Processes the response generator into a full response text.
        """
        return "".join(generator)

    def handle_function_call(
            self,
            messages: list[dict[str, Any]],
            name: str,
            arguments: str,
    ) -> Generator[str, None, None]:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "function_call": {"name": name, "arguments": arguments},
            }
        )

        if messages and messages[-1]["role"] == "assistant":
            yield "\n"

        dict_args = json.loads(arguments)
        joined_args = ", ".join(f'{k}="{v}"' for k, v in dict_args.items())
        yield f"> @FunctionCall `{name}({joined_args})` \n\n"

        # Call the function
        result = get_function(name)(**dict_args)

        # Write the function output to the named pipe
        fifo_path = f"/tmp/sgpt_function_output-{os.getpid()}"
        os.mkfifo(fifo_path)
        with open(fifo_path, "w") as fifo:
            fifo.write(f"Function {name} output:\n{result}\n\n")

        # Optionally show a short summary in the main conversation if needed
        yield f"[Function {name} executed, output written to named pipe]\n"

        messages.append({"role": "function", "content": result, "name": name})

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
        """
        model = self.config.model if model is None else model
        functions = get_openai_schemas()
        messages = self.make_messages(prompt.strip())
        response = self.get_completion(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            functions=functions,
            caching=caching,
            **kwargs,
        )
        reply = ""
        for chunk in response:
            yield chunk
            reply += chunk
        self.conversation_history.append({"role": "assistant", "content": reply})
