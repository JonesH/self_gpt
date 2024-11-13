from typing import Any
from sgpt.handlers.role_function_handler import AgentABC
from sgpt.role import SystemRole


ROLE = """# Open file: {path}
# hash: {hash}
# content:

{content}

##############

{task}
"""


class FileEditorAgent(AgentABC):
    """
    Handler for managing file editing tasks, utilizing functions to read file contents and apply changes.
    """

    class Config:
        model = "gpt-4o-mini"
        name = "FileEditor"
        role = ROLE
        functions = ["apply_file_changes"]

    @property
    def state(self) -> dict[str, Any]:
        content = Path(path).read_text()
        return dict(path="", hash=hash(content), content=content, task=self.task)

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
        return self.handle_with_role_and_functions(prompt)
