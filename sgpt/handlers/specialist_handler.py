from typing import Any
from sgpt.handlers.role_function_handler import AgentABC
from sgpt.role import SystemRole


class SpecialistABC(AgentABC):
    """
    Handler for managing specialists, which are specific role-function combinations.
    """

    def __init__(
        self, role: SystemRole, functions: list[dict[str, Any]], markdown: bool
    ) -> None:
        super().__init__(role, functions, markdown)

    def make_messages(self, prompt: str) -> list[dict[str, str]]:
        """
        Create messages for the specialist based on the prompt.
        """
        return [
            {"role": "system", "content": self.role.role},
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def create_specialist() -> None:
        """
        Interactive method to create a new specialist.
        """
        # Logic to create a new specialist interactively
        pass

    @staticmethod
    def show_specialist(name: str) -> None:
        """
        Show details of a specific specialist.
        """
        # Logic to show specialist details
        pass

    @staticmethod
    def list_specialists() -> None:
        """
        List all available specialists.
        """
        # Logic to list all specialists
        pass
