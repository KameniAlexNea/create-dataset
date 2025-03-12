import pytest

from qageneratorllm.utils import (
    _select_pages,
    get_files,
    is_potential_title,
    is_valid_file,
    merge_sentences,
)


def test_merge_sentences():
    input_text = "First line-\nsecond part\nNew sentence."
    expected = "First linesecond part\nNew sentence."
    assert merge_sentences(input_text) == expected


def test_is_potential_title():
    assert is_potential_title("1. Chapter Title")
    assert is_potential_title("The Main Title")
    assert not is_potential_title("this is not a title.")


def test_is_valid_file():
    valid_content = "Normal text content\nwith multiple lines\nwith multiple lines\nwith multiple lines\nbut not too many."
    invalid_content = "".join([f"Line {i}...............\n" for i in range(10)])

    assert is_valid_file(valid_content)
    assert not is_valid_file(invalid_content)


def test_select_pages():
    pages = list(range(10))
    result = _select_pages(pages, n=2, m=3)

    assert len(result) <= 2
    for page_group in result:
        assert len(page_group) <= 3


@pytest.mark.parametrize(
    "files_content",
    [
        ["File 1 content", "File 2 content"],
        ["Invalid...........", "Valid content"],
    ],
)
def test_get_files(tmp_path, files_content):
    # Create temporary files
    files = []
    for i, content in enumerate(files_content):
        file_path = tmp_path / f"test_{i}.txt"
        file_path.write_text(content)
        files.append(str(file_path))

    result = get_files(files)
    assert isinstance(result, list)
    assert all(isinstance(content, str) for content in result)
