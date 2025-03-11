import os
import pytest

@pytest.fixture
def sample_context():
    return """
    Ancient Egypt was a civilization in Northeastern Africa that existed from about 3100 BC to 30 BC.
    The Nile River shaped Ancient Egyptian civilization.
    Pyramids were built as tombs for pharaohs and their consorts during the Old and Middle Kingdom periods.
    """

@pytest.fixture
def sample_qa_response():
    return {
        "questions": [
            {
                "question": "When did Ancient Egypt civilization exist?",
                "answer": "Ancient Egypt existed from about 3100 BC to 30 BC."
            }
        ]
    }

@pytest.fixture
def sample_mcq_response():
    return {
        "questions": [
            {
                "question": "What was the purpose of pyramids in Ancient Egypt?",
                "choices": {
                    "a": "Tombs for pharaohs and their consorts",
                    "b": "Storage facilities",
                    "c": "Military fortresses",
                },
                "answer": ["a"],
                "explanation": "Pyramids were built as tombs for pharaohs and their consorts during the Old and Middle Kingdom periods."
            }
        ]
    }

@pytest.fixture
def temp_text_file(tmp_path):
    content = "Sample text for testing.\nMultiple lines of content.\n"
    file_path = tmp_path / "test.txt"
    file_path.write_text(content)
    return str(file_path)
