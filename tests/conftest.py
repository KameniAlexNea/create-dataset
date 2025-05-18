import pytest

from qageneratorllm.qa_dataclass import (
    MultipleChoiceOption,
    MultipleChoiceQuestion,
    MultipleChoiceQuestionBank,
    OpenEndedQuestion,
    OpenEndedQuestionBank,
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
    return OpenEndedQuestionBank(
        open_ended_questions=[
            OpenEndedQuestion(
                question_prompt="When did Ancient Egypt civilization exist?",
                reference_answer="Ancient Egypt existed from about 3100 BC to 30 BC.",
            )
        ]
    )


@pytest.fixture
def sample_mcq_response():
    return MultipleChoiceQuestionBank(
        mcq_questions=[
            MultipleChoiceQuestion(
                question_text="What was the purpose of pyramids in Ancient Egypt?",
                answer_options=[
                    MultipleChoiceOption(
                        option_id="A",
                        option_text="Tombs for pharaohs and their consorts",
                    ),
                    MultipleChoiceOption(
                        option_id="B", option_text="Storage facilities"
                    ),
                    MultipleChoiceOption(
                        option_id="C", option_text="Military fortresses"
                    ),
                ],
                correct_option_ids=["A"],
                answer_explanation="Pyramids were built as tombs for pharaohs and their consorts during the Old and Middle Kingdom periods.",
            )
        ]
    )


@pytest.fixture
def temp_text_file(tmp_path):
    content = "Sample text for testing.\nMultiple lines of content.\n"
    file_path = tmp_path / "test.txt"
    file_path.write_text(content)
    return str(file_path)
