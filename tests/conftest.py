import pytest

from qageneratorllm.qa_dataclass import (
    AnswerChoice,
    MCQBank,
    QABank,
    Question,
    QuestionAnswer,
)


@pytest.fixture
def sample_context():
    return """
    Ancient Egypt was a civilization in Northeastern Africa that existed from about 3100 BC to 30 BC.
    The Nile River shaped Ancient Egyptian civilization.
    Pyramids were built as tombs for pharaohs and their consorts during the Old and Middle Kingdom periods.
    """


@pytest.fixture
def sample_qa_response():
    return QABank(
        **{
            "questions": [
                QuestionAnswer(
                    **{
                        "question": "When did Ancient Egypt civilization exist?",
                        "answer": "Ancient Egypt existed from about 3100 BC to 30 BC.",
                    }
                )
            ]
        }
    )


@pytest.fixture
def sample_mcq_response():
    return MCQBank(
        **{
            "questions": [
                Question(
                    **{
                        "question": "What was the purpose of pyramids in Ancient Egypt?",
                        "choices": [
                            AnswerChoice(
                                **{
                                    "letter": "a",
                                    "text": "Tombs for pharaohs and their consorts",
                                }
                            ),
                            AnswerChoice(
                                **{"letter": "b", "text": "Storage facilities"}
                            ),
                            AnswerChoice(
                                **{"letter": "c", "text": "Military fortresses"}
                            ),
                        ],
                        "answer": ["a"],
                        "explanation": "Pyramids were built as tombs for pharaohs and their consorts during the Old and Middle Kingdom periods.",
                    }
                )
            ]
        }
    )


@pytest.fixture
def temp_text_file(tmp_path):
    content = "Sample text for testing.\nMultiple lines of content.\n"
    file_path = tmp_path / "test.txt"
    file_path.write_text(content)
    return str(file_path)
