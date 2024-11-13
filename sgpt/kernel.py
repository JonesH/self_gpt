# Enhanced REPL Using ipykernel and DefaultAgent
from typing import Generator

from ipykernel.kernelbase import Kernel
from sgpt.handlers.agent_abc import AgentABC
from sgpt.handlers.default_agent import DefaultAgent


class EnhancedREPLKernel(Kernel):
    implementation = "Enhanced REPL"
    implementation_version = "1.0"
    language = "plaintext"
    language_version = "1.0"
    language_info = {
        "name": "shell-gpt",
        "mimetype": "text/plain",
        "file_extension": ".txt",
    }
    banner = "Enhanced REPL using ipykernel and ShellGPT agents"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_registry = {}
        self.default_agent = DefaultAgent()  # Initialize the default agent

    def register_agent(self, name: str, agent: AgentABC):
        """Register a new agent."""
        self.agent_registry[name] = agent

    def do_execute(
            self,
            code: str,
            silent: bool,
            store_history: bool = True,
            user_expressions=None,
            allow_stdin: bool = False,
    ):
        """Handle code execution and stream responses."""
        if silent:
            return {"status": "ok", "execution_count": self.execution_count}

        response_generator = self.route_to_agent(code)

        # Ensure live updates are sent to the client
        if response_generator:
            for chunk in response_generator:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stdout", "text": chunk}
                )

        return {"status": "ok", "execution_count": self.execution_count}

    def route_to_agent(self, code: str) -> Generator[str, None, None]:
        """
        Parse input and route to the appropriate agent.
        Use the default agent if no specific agent is specified.
        """
        if code.startswith("@"):  # If the input starts with '@', route to specific agent
            parts = code.split(" ", 1)
            agent_name = parts[0][1:]  # Remove the '@'
            message = parts[1] if len(parts) > 1 else ""

            agent = self.agent_registry.get(agent_name)
            if agent:
                return agent.handle(message)
            yield f"Agent '{agent_name}' not found."
            return

        # Use the default agent for plain input
        yield from self.default_agent.handle(code)


# Main entry point to initialize the kernel app
if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp


    class EnhancedREPLKernelApp(IPKernelApp):
        kernel_class = EnhancedREPLKernel


    EnhancedREPLKernelApp.launch_instance()
