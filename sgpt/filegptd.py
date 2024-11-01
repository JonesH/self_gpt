import asyncio
import difflib
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from watchfiles import watch, Change
import typer
from sgpt.handlers.repl_handler import ReplHandler  # Import the base REPL handler
from sgpt.role import SystemRole

app = typer.Typer()


@dataclass
class ChangingFile:
    file_path: Path
    original_content: list[str] = field(init=False)
    changes: list[list[str]] = field(default_factory=list)

    def __post_init__(self):
        """
        Initialize the ChangingFile by reading the original file content.
        """
        self.file_path = Path(self.file_path)
        self.original_content = self._read_file()

    def _read_file(self) -> list[str]:
        """
        Reads the content of the file and returns it as a list of lines.
        """
        if self.file_path.exists():
            with self.file_path.open('r', encoding='utf-8') as file:
                return file.readlines()
        return []

    def record_change(self, new_content: list[str]) -> None:
        """
        Records a change by computing the diff between the current version and new content.
        """
        current_version = self.get_current_version()
        diff = list(difflib.unified_diff(current_version, new_content, lineterm=''))
        print(f'{self.file_path} changed: {diff}')
        with open('record.log', 'a') as f:
            f.write(f'{self.file_path} changed: {diff}')
        self.changes.append(diff)

    @lru_cache
    def get_version(self, version: int) -> list[str]:
        """
        Retrieves a specific version of the file by applying the first `version` diffs.
        `version` should be between 0 (original) and len(self.changes) (current).
        """
        content = self.original_content
        for change in self.changes[:version]:
            content = self.apply_diff(content, change)
        return content

    def get_current_version(self) -> list[str]:
        """
        Returns the current version by applying all diffs to the original content.
        """
        return self.get_version(len(self.changes))

    @staticmethod
    def apply_diff(original: list[str], diff: list[str]) -> list[str]:
        """
        Applies a unified diff to the original content and returns the updated content.
        """
        return list(difflib.restore(diff, 1))

    def __hash__(self):
        return hash(self.file_path)


class Watcher(ABC):
    @abstractmethod
    def notify(self) -> None:
        raise NotImplementedError()


@dataclass
class FilesView:
    file_paths: list[Path]
    files: dict[str, ChangingFile] = field(init=False)
    observers: list[Watcher] = field(default_factory=list)

    def __post_init__(self):
        """
        Initialize FilesView with ChangingFile instances for each file.
        """
        self.files = {str(file_path): ChangingFile(file_path) for file_path in self.file_paths}

    def notify_observers(self) -> None:
        for observer in self.observers:
            observer.notify()

    def record_change(self, file_path: str, new_content: list[str]) -> None:
        """
        Records a change for the specified file.
        """
        if file_path in self.files:
            self.files[file_path].record_change(new_content)
            self.notify_observers()
        else:
            raise ValueError(f"File {file_path} is not being tracked.")

    @property
    def file_history_summary(self) -> str:
        """
        Returns a summary of the files being tracked and their change history.
        """
        summary = "Watched Files and Their Change History:\n"
        for path, changing_file in self.files.items():
            summary += f"- {path}: {len(changing_file.changes)} changes recorded\n"
        return summary

    def register_observer(self, watcher: Watcher):
        self.observers.append(watcher)


class FileWatcherREPLHandler(ReplHandler, Watcher):

    def notify(self) -> None:
        self.role = self._role

    def __init__(self, files_view: FilesView, *args, model='gpt-4o-mini', **kwargs):
        """
        Initialize with a custom system prompt containing the files' history.
        """
        self.model = model
        self.files_view = files_view
        self.files_view.register_observer(self)

        super().__init__("temp", self._role, *args, **kwargs)
        self.role = self._role

    @property
    def _system_prompt(self) -> str:
        return (
            "You are FileGPT, a helpful assistant monitoring file changes.\n\n"
            f"{self.files_view.file_history_summary}\n"
            "Answer questions or provide insights based on these tracked files."
        )

    @property
    def _role(self) -> SystemRole:
        return SystemRole(name="FileGPT", role=self._system_prompt)

    def start_repl(self):
        """
        Starts the REPL with the custom system prompt including file history.
        """
        super().handle(init_prompt="Let's start the REPL. Ask me about the watched files!", model=self.model,
                        temperature=0.7, top_p=1.0, caching=False)

@dataclass
class FileGPTd:
    file_paths: list[str | Path]
    files_view: FilesView = field(init=False)

    def __post_init__(self):
        """
        Initialize FileGPTd by setting up FilesView.
        """
        self.file_paths = [Path.cwd() / Path(path).expanduser().resolve() for path in self.file_paths]
        self.files_view = FilesView(self.file_paths)


    def monitor_files(self):
        """
        Asynchronously monitors files for changes and updates the FilesView.
        """
        print(f'monitoring files: {self.files_view.files.keys()}')
        for changes in watch(*self.files_view.files.keys()):
            for change_type, file_path in changes:
                if change_type in {Change.modified, Change.added, Change.deleted}:
                    self.handle_change(change_type, file_path)

    def handle_change(self, change_type: Change, file_path: str) -> None:
        """
        Handles file changes by updating the corresponding ChangingFile instance.
        """
        cwd = Path.cwd()
        file_path = Path(file_path)
        file_path_str = str(file_path)

        # Read the updated file content
        new_content = self.files_view.files[file_path_str]._read_file()

        # Update the ChangingFile instance with the new content
        self.files_view.record_change(file_path_str, new_content)

    def start_services(self):
        """
        Runs the REPL and file monitoring services concurrently.
        """
        repl_handler = FileWatcherREPLHandler(self.files_view, markdown=True)
        await asyncio.gather(
            repl_handler.start_repl(),
            self.monitor_files()
        )


@app.command()
def start(file_paths: list[str]):
    """
    Starts the FileGPTd service to monitor file changes in specified files or directories.
    """
    if not file_paths:
        typer.echo("Please provide file paths or directories to monitor.")
        raise typer.Exit(1)

    file_gptd = FileGPTd(file_paths)
    typer.echo(f"Starting FileGPTd to watch changes in: {', '.join(file_paths)}")

    # Start the REPL and file monitoring asynchronously
    asyncio.run(file_gptd.start_services())



if __name__ == "__main__":
    app()
