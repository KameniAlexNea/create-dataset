import pytest

from qageneratorllm.qa_dataclass import (
    AnswerChoice,
    MCQBank,
    Question,
    QuestionAnswer,
)


def test_answer_choice():
    choice = AnswerChoice(letter="A", text="Test answer")
    assert choice.letter == "A"
    assert choice.text == "Test answer"


def test_mcq_question():
    question = Question(
        question="Test question?",
        choices=[
            AnswerChoice(letter="A", text="Choice A"),
            AnswerChoice(letter="B", text="Choice B"),
        ],
        answer=["A"],
        explanation="Test explanation",
    )
    assert question.question == "Test question?"
    assert len(question.choices) == 2
    assert question.answer == ["A"]


def test_qa_question():
    qa = QuestionAnswer(question="Test question?", answer="Test answer")
    assert qa.question == "Test question?"
    assert qa.answer == "Test answer"


def test_mcq_bank():
    bank = MCQBank(
        questions=[
            Question(
                question="Test?",
                choices=[AnswerChoice(letter="A", text="Choice")],
                answer=["A"],
                explanation="Test",
            )
        ]
    )
    assert len(bank.questions) == 1


def test_invalid_mcq():
    with pytest.raises(ValueError):
        Question(
            question="Test?",
            choices=[],  # Empty choices
            answer=["A"],
            explanation="Test",
        )
