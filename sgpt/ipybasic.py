from ipykernel.kernelbase import Kernel
from openai import OpenAI


class BasicREPLKernel(Kernel):
    implementation = "Basic REPL"
    implementation_version = "1.0"
    language = "plaintext"
    language_version = "1.0"
    language_info = {
        "name": "plaintext",
        "mimetype": "text/plain",
        "file_extension": ".txt",
    }
    banner = "Basic REPL for GPT Communication"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize OpenAI client with your API key
        self.client = OpenAI(api_key="YOUR_API_KEY")

    def do_execute(
            self, code: str, silent: bool, store_history: bool = True, user_expressions=None, allow_stdin: bool = False
    ):
        """Handle code execution from the REPL."""
        if silent:
            return {"status": "ok", "execution_count": self.execution_count}

        # Send the input to GPT and get a response
        response = self.send_to_gpt(code)

        # Display the GPT response
        stream_content = {"name": "stdout", "text": response}
        self.send_response(self.iopub_socket, "stream", stream_content)

        return {"status": "ok", "execution_count": self.execution_count}

    def send_to_gpt(self, prompt: str) -> str:
        """Send the prompt to GPT and return the response."""
        try:
            completion = self.client.completions.create(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
            )
            return completion["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error communicating with GPT: {e}"


# Main entry point to initialize the kernel app
if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp


    class BasicREPLKernelApp(IPKernelApp):
        kernel_class = BasicREPLKernel


    BasicREPLKernelApp.launch_instance()
