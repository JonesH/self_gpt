from dataclasses import field, InitVar
from pathlib import Path
from typing import Any

from attr import dataclass

from sgpt.handlers.agent_abc import AgentABC

ROLE = """# Open file: {path}
# hash: {hash}
# content:

{content}

##############

{task}
"""


@dataclass
class FileEditorAgent(AgentABC):
    """
    Handler for managing file editing tasks, utilizing functions to read file contents and apply changes.
    """
    path_str: InitVar[str]
    prompt: str
    path: Path = field(init=False)

    class Config:
        model = "gpt-4o-mini"
        name = "FileEditor"
        role = ROLE
        functions = ["apply_file_changes"]
        tool_choice = 'required'

    def __post_init__(self, path_str) -> None:
        super().__post_init__()
        self.path = Path(path_str).expanduser().resolve()
        if not self.path.exists():
            self.path.mkdir(parents=True)
            self.path.touch()

    @property
    def state(self) -> dict[str, Any]:
        content = self.path.read_text()
        return dict(path=str(self.path),
                    hash=hash(content),
                    content=content,
                    task=self.prompt)

    def make_messages(self, prompt: str) -> list[dict[str, str]]:
        """
        Create messages for the file editor based on the prompt.
        """
        return [
            {"role": "system", "content": self.system_role.role},
            {"role": "user", "content": prompt},
        ]

    def handle_file_editing(self, file_path: str, task_description: str) -> str:
        """
        Handles file editing tasks by reading the file contents and applying changes if necessary.
        """
        # Example logic for handling file editing
        # This method can be expanded with specific logic for reading and applying changes
        prompt = f"Open file: {file_path}\nTask: {task_description}"
        return self.handle(prompt)


file_agent = FileEditorAgent()
