import pytest

from qageneratorllm.qa_dataclass import (
    MultipleChoiceOption,
    MultipleChoiceQuestion,
    MultipleChoiceQuestionBank,
    OpenEndedQuestion,
)


def test_multiple_choice_option():
    choice = MultipleChoiceOption(option_id="A", option_text="Test answer")
    assert choice.option_id == "A"
    assert choice.option_text == "Test answer"


def test_multiple_choice_question():
    question = MultipleChoiceQuestion(
        question_text="Test question?",
        answer_options=[
            MultipleChoiceOption(option_id="A", option_text="Choice A"),
            MultipleChoiceOption(option_id="B", option_text="Choice B"),
        ],
        correct_option_ids=["A"],
        answer_explanation="Test explanation",
    )
    assert question.question_text == "Test question?"
    assert len(question.answer_options) == 2
    assert question.correct_option_ids == ["A"]


def test_open_ended_question():
    qa = OpenEndedQuestion(
        question_prompt="Test question?", reference_answer="Test answer"
    )
    assert qa.question_prompt == "Test question?"
    assert qa.reference_answer == "Test answer"


def test_multiple_choice_question_bank():
    bank = MultipleChoiceQuestionBank(
        mcq_questions=[
            MultipleChoiceQuestion(
                question_text="Test?",
                answer_options=[
                    MultipleChoiceOption(option_id="A", option_text="Choice")
                ],
                correct_option_ids=["A"],
                answer_explanation="Test",
            )
        ]
    )
    assert len(bank.mcq_questions) == 1


def test_invalid_mcq():
    with pytest.raises(ValueError):
        MultipleChoiceQuestion(
            question_text="Test?",
            answer_options=[],  # Empty choices
            correct_option_ids=["A"],
            answer_explanation="Test",
        )
