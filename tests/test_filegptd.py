import pytest
from pathlib import Path
from sgpt.filegptd import ChangingFile, FilesView


@pytest.fixture
def temp_file(tmp_path):
    # Create a temporary file for testing purposes
    file = tmp_path / "test_file.txt"
    file.write_text("Line 1\nLine 2\nLine 3\n")
    return file


@pytest.fixture
def changing_file(temp_file):
    # Initialize ChangingFile with the temporary file's path
    return ChangingFile(temp_file)


def test_initial_content(changing_file, temp_file):
    # Test if the initial content of ChangingFile is read correctly
    assert changing_file.original_content == ["Line 1\n", "Line 2\n", "Line 3\n"]


def test_record_change(changing_file):
    # Simulate a change in the file content
    new_content = ["Line 1\n", "Line 2 changed\n", "Line 3\n"]
    changing_file.record_change(new_content)

    # Check if the change has been recorded
    assert len(changing_file.changes) == 1
    # Check if the current version reflects the change
    assert changing_file.get_current_version() == new_content


def test_get_version(changing_file):
    # Simulate multiple changes
    first_change = ["Line 1 changed\n", "Line 2\n", "Line 3\n"]
    second_change = ["Line 1 changed\n", "Line 2\n", "Line 3 changed\n"]
    changing_file.record_change(first_change)
    changing_file.record_change(second_change)

    # Test retrieval of different versions
    assert changing_file.get_version(0) == ["Line 1\n", "Line 2\n", "Line 3\n"]
    assert changing_file.get_version(1) == first_change
    assert changing_file.get_version(2) == second_change


def test_apply_diff(changing_file):
    # Simulate a change and test applying the diff
    new_content = ["Line 1\n", "Line 2 updated\n", "Line 3\n"]
    changing_file.record_change(new_content)

    # Retrieve and apply the diff
    diff = changing_file.changes[0]
    applied_content = ChangingFile.apply_diff(changing_file.original_content, diff)
    assert applied_content == new_content


# Tests for FilesView
@pytest.fixture
def files_view(tmp_path):
    # Create multiple temporary files and initialize FilesView
    file1 = tmp_path / "file1.txt"
    file1.write_text("File1 - Line 1\nFile1 - Line 2\n")
    file2 = tmp_path / "file2.txt"
    file2.write_text("File2 - Line 1\nFile2 - Line 2\n")
    return FilesView([file1, file2])


def test_files_view_initialization(files_view):
    # Test if FilesView initializes ChangingFile objects correctly
    assert len(files_view.files) == 2
    assert "file1.txt" in files_view.files
    assert "file2.txt" in files_view.files


def test_files_view_record_change(files_view, tmp_path):
    # Simulate changes for file1.txt
    file1_path = str(tmp_path / "file1.txt")
    new_content = ["File1 - Line 1 updated\n", "File1 - Line 2\n"]
    files_view.record_change(file1_path, new_content)

    # Check if the change was recorded
    assert len(files_view.files[file1_path].changes) == 1
    # Check if the current version reflects the change
    assert files_view.get_current_file_version(file1_path) == new_content


def test_files_view_get_file_version(files_view, tmp_path):
    # Simulate multiple changes for file1.txt
    file1_path = str(tmp_path / "file1.txt")
    first_change = ["File1 - Line 1\n", "File1 - Line 2 updated\n"]
    second_change = ["File1 - Line 1 updated\n", "File1 - Line 2 updated\n"]

    files_view.record_change(file1_path, first_change)
    files_view.record_change(file1_path, second_change)

    # Test retrieval of different versions
    assert files_view.get_file_version(file1_path, 0) == ["File1 - Line 1\n", "File1 - Line 2\n"]
    assert files_view.get_file_version(file1_path, 1) == first_change
    assert files_view.get_file_version(file1_path, 2) == second_change


def test_files_view_view_changes(files_view, tmp_path):
    # Simulate changes for file2.txt
    file2_path = str(tmp_path / "file2.txt")
    new_content = ["File2 - Line 1 updated\n", "File2 - Line 2\n"]
    files_view.record_change(file2_path, new_content)

    # Check if the changes are accessible in the view_changes method
    changes = files_view.view_changes(file2_path)
    assert len(changes) == 1
    assert changes[0] == files_view.files[file2_path].changes[0]
