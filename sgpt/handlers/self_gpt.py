#!/usr/bin/env python3
import sys


sys.stderr = sys.stdout

from sgpt.handlers.chat_handler import ChatHandler
import json
from sgpt.handlers.repl_handler import ReplHandler
from sgpt.handlers.handler import Handler
from sgpt.role import SystemRole
from pathlib import Path
from typing import Any
import typer


class SelfGPT(ChatHandler):
    """
    Self-aware REPL Handler that extends ChatHandler.
    - It is aware of its own code, settings, roles, functions, and other handlers.
    - Builds upon the existing Shell-GPT classes.
    """

    def __init__(self, chat_id: str, markdown: bool) -> None:
        self.self_code = self._get_self_code()
        self.config = self._get_current_settings()
        self.roles = self._get_current_roles()
        self.functions = self._get_current_functions()
        self.handlers = self._get_current_handlers()

        role_description = f"""
        You are SelfGPT, a self-aware REPL-based chatbot and programming assistant.
        You know about your own source code, settings, available roles, functions, and handlers.
        Current settings: {json.dumps(self.config, indent=4)}
        Available roles: {list(self.roles.keys())}
        Available functions: {list(self.functions.keys())}
        Handlers you can use: {[handler.__name__ for handler in self.handlers]}
        Source code: {}
        """
        role = SystemRole("SelfGPT", role_description)
        if not role._exists:
            role._save()
        super().__init__(chat_id, role, markdown)

    @classmethod
    def get_role_name(cls, initial_message: str) -> str | None:
        return 'selfgpt'

    def _get_self_code(self) -> str:
        """Retrieves its own source code."""
        def print_file(path) -> str:
            code = Path(path).read_text()
            return f'# {path}\n{code}'

        try:
            import sgpt
            codes = map(print_file, Path(sgpt.__file__).parent.glob('**/*.py'))
            return print_file(__file__) + '\n'.join(codes)
        except Exception as e:
            return f"Error retrieving self code: {e}"

    def _get_current_settings(self) -> dict:
        """Retrieves current settings from the configuration."""
        from sgpt.config import cfg
        return dict(cfg)

    def _get_current_roles(self) -> dict:
        """Retrieves all available roles."""
        roles = {}
        for role_file in SystemRole.storage.glob("*.json"):
            roles[role_file.stem] = json.loads(role_file.read_text())
        return roles

    def _get_current_functions(self) -> dict:
        """Retrieves all available functions."""
        from sgpt.function import functions
        return {function.name: function.openai_schema for function in functions}

    def _get_current_handlers(self) -> list:
        """Retrieves all available handlers."""
        return [Handler, ReplHandler, SelfGPT]

    @classmethod
    def _get_multiline_input(cls) -> str:
        multiline_input = ""
        while (user_input := typer.prompt("...", prompt_suffix="")) != '"""':
            multiline_input += user_input + "\n"
        return multiline_input

    def handle(self, init_prompt: str, **kwargs: Any) -> None:
        """Overriding handle to add introspective features."""
        if self.initiated:
            self.show_messages(self.chat_id)

        full_completion = ""
        while True:
            prompt = self._get_user_input()
            if prompt == "exit()":
                raise typer.Exit()
            elif prompt == "show_self()":
                typer.echo(self.self_code)
            elif prompt == "show_settings()":
                typer.echo(json.dumps(self.config, indent=4))
            elif prompt == "show_roles()":
                typer.echo(json.dumps(self.roles, indent=4))
            elif prompt == "show_functions()":
                typer.echo(json.dumps(self.functions, indent=4))
            elif prompt == "show_handlers()":
                typer.echo([handler.__name__ for handler in self.handlers])
            else:
                if init_prompt:
                    prompt = f"{init_prompt}\n\n\n{prompt}"
                    init_prompt = ""
                full_completion = super().handle(prompt=prompt, **kwargs)

    def _get_user_input(self) -> str:
        """Prompts the user for input."""
        return typer.prompt(">>>", prompt_suffix=" ")

