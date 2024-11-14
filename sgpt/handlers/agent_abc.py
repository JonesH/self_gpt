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
class AgentABC(ABC):
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

        api_base_url = cfg.get("API_BASE_URL")
        base_url = None if api_base_url == "default" else api_base_url
        timeout = int(cfg.get("REQUEST_TIMEOUT"))

        markdown = True
        code_theme = cfg.get("CODE_THEME")
        color = cfg.get("DEFAULT_COLOR")

    @classmethod
    @property
    def config(cls):
        return cls.Config()

    def __post_init__(self) -> None:
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

    import time

    def handle_function_call(
            self,
            messages: list[dict[str, Any]],
            name: str,
            arguments: str,
    ) -> Generator[str, None, None]:
        # Append the initial function call to messages
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "function_call": {"name": name, "arguments": arguments},
            }
        )

        if messages and messages[-1]["role"] == "assistant":
            yield "\n"

        # Parse the arguments and prepare the function call
        dict_args = json.loads(arguments)
        joined_args = ", ".join(f'{k}="{v}"' for k, v in dict_args.items())
        yield f"> @FunctionCall `{name}({joined_args})` \n\n"

        # Call the function and process the structured output
        result = get_function(name)(**dict_args)

        if isinstance(result, dict):  # Handle structured output
            if result.get("status") == "error":
                content = result.get("message", "An unknown error occurred.")
            elif result.get("status") == "success":
                file_hash = result.get("file_hash", "")
                content = f"[File Hash: {file_hash}]\n\n{result.get('content', '')}"
            else:
                content = "Unexpected response structure received."
        else:  # Handle legacy or unstructured output
            content = str(result)

        with open('/tmp/funout.log', 'w') as f:
            f.write(f"Function {name} output:\n{content}\n\n")
        # Optionally show a short summary in the main conversation if needed
        yield f"[Function {name} executed, output written to named pipe]\n"

        # Append the final structured response to messages
        messages.append({"role": "function", "content": content, "name": name})

    def complete(
            self,
            model: str,
            temperature: float,
            top_p: float,
            messages: list[dict[str, Any]],
            functions: list[dict[str, str]] | None,
            **additional_kwargs,
    ) -> Generator[str, None, None]:
        name = arguments = ""

        try:
            response = completion(
                model=model,
                temperature=temperature,
                top_p=top_p,
                messages=messages,
                stream=True,
                **additional_kwargs,
            )
        except Exception as e:
            import typer
            typer.secho(str(e), fg=typer.colors.RED)
            raise

        try:
            for chunk in response:
                delta = chunk.choices[0].delta

                # LiteLLM uses dict instead of Pydantic object like OpenAI does.
                tool_calls = (
                    delta.get("tool_calls") if use_litellm else delta.tool_calls
                )
                if tool_calls:
                    for tool_call in tool_calls:
                        if tool_call.function.name:
                            name = tool_call.function.name
                        if tool_call.function.arguments:
                            arguments += tool_call.function.arguments
                if chunk.choices[0].finish_reason == "tool_calls":
                    yield from self.handle_function_call(messages, name, arguments)
                    yield from self.get_completion(
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        messages=messages,
                        functions=functions,
                        caching=False,
                    )
                    return

                yield delta.content or ""
        except KeyboardInterrupt:
            response.close()

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
