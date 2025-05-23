import os
from typing import List

from pydantic import BaseModel, Field, field_validator


class MultipleChoiceOption(BaseModel):
    """Represents a single option in a multiple-choice question."""

    option_id: str = Field(
        description="The identifier for this option (e.g., 'A', 'B', 'C', 'D')"
    )
    option_text: str = Field(
        description="The text content of this multiple choice option"
    )


class MultipleChoiceQuestion(BaseModel):
    """A question with multiple predefined answer options where one or more options are correct."""

    question_text: str = Field(
        description="The complete text of the multiple-choice question"
    )
    answer_options: List[MultipleChoiceOption] = Field(
        description="List of all possible answer options for this question"
    )
    correct_option_ids: List[str] = Field(
        description="List of option_ids for all correct answers (e.g., ['A', 'C'] for multiple correct answers)"
    )
    answer_explanation: str = Field(
        description="Detailed explanation of why the marked answers are correct and others are incorrect"
    )

    @field_validator("answer_options")
    @classmethod
    def validate_unique_options(cls, options: list[MultipleChoiceOption]):
        for option in options:
            if not isinstance(option, MultipleChoiceOption):
                raise ValueError("Options must be of class MultipleChoiceOption")
        option_ids = {option.option_id for option in options}
        if len(option_ids) != len(options):
            raise ValueError("Answer options must have unique identifiers.")
        return options

    @field_validator("correct_option_ids")
    @classmethod
    def validate_correct_options(cls, correct_ids, values):
        available_ids = {
            option.option_id for option in values.data.get("answer_options", [])
        }
        for option_id in correct_ids:
            if option_id not in available_ids:
                raise ValueError(
                    f"Invalid answer option '{option_id}'. Must be one of {sorted(available_ids)}"
                )
        return correct_ids


class MultipleChoiceQuestionBank(BaseModel):
    """Collection of multiple-choice questions with predefined answer options."""

    mcq_questions: List[MultipleChoiceQuestion] = Field(
        description="Collection of all multiple-choice questions in this question bank"
    )

    @classmethod
    def example(cls) -> "MultipleChoiceQuestionBank":
        """Returns an example MultipleChoiceQuestionBank with sample questions."""
        return cls(
            mcq_questions=[
                MultipleChoiceQuestion(
                    question_text="What is the capital of France?",
                    answer_options=[
                        MultipleChoiceOption(option_id="A", option_text="London"),
                        MultipleChoiceOption(option_id="B", option_text="Paris"),
                        MultipleChoiceOption(option_id="C", option_text="Berlin"),
                        MultipleChoiceOption(option_id="D", option_text="Madrid"),
                    ],
                    correct_option_ids=["B"],
                    answer_explanation="Paris is the capital and largest city of France.",
                ),
                MultipleChoiceQuestion(
                    question_text="Which of the following are primary colors?",
                    answer_options=[
                        MultipleChoiceOption(option_id="A", option_text="Red"),
                        MultipleChoiceOption(option_id="B", option_text="Green"),
                        MultipleChoiceOption(option_id="C", option_text="Blue"),
                        MultipleChoiceOption(option_id="D", option_text="Purple"),
                    ],
                    correct_option_ids=["A", "C"],
                    answer_explanation="Red and Blue are primary colors. Green is also a primary color in the RGB color model, but not in the traditional RYB color model used in art.",
                ),
            ]
        )


class OpenEndedQuestion(BaseModel):
    """A single question requiring a free-form textual answer rather than selection from options."""

    question_prompt: str = Field(
        description="The complete text of the question that requires a detailed written response"
    )
    reference_answer: str = Field(
        description="The exemplar answer that completely addresses the question with factual information"
    )


class OpenEndedQuestionBank(BaseModel):
    """Collection of questions requiring free-form textual responses."""

    open_ended_questions: List[OpenEndedQuestion] = Field(
        description="Collection of all open-ended questions with model answers"
    )

    @classmethod
    def example(cls) -> "OpenEndedQuestionBank":
        """Returns an example OpenEndedQuestionBank with sample question-answer pairs."""
        return cls(
            open_ended_questions=[
                OpenEndedQuestion(
                    question_prompt="What is the capital of France?",
                    reference_answer="The capital of France is Paris.",
                ),
                OpenEndedQuestion(
                    question_prompt="Who wrote 'Romeo and Juliet'?",
                    reference_answer="William Shakespeare wrote 'Romeo and Juliet'.",
                ),
                OpenEndedQuestion(
                    question_prompt="What is the chemical symbol for water?",
                    reference_answer="The chemical formula for water is H₂O, consisting of two hydrogen atoms and one oxygen atom.",
                ),
            ]
        )


class ModelName:
    """Constants for default model names used by different LLM providers."""

    ANTHROPIC = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-sonnet-20240229")
    OLLAMA = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5")
    OPENAI = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    XAI = os.getenv("XAI_MODEL_NAME", "grok-beta")
    GROQ = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")


class LLMProviderType:
    """Constants identifying different LLM service providers."""

    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENAI = "openai"
    XAI = "xai"
    GROQ = "groq"


class QuestionType:
    """Constants identifying different types of questions that can be generated."""

    MCQ = "mcq"  # Multiple choice questions
    QA = "qa"  # Open-ended questions


class OutputType:
    """Constants identifying different output formats for LLM responses."""

    DATACLASS = "dataclass"  # Return a structured dataclass
    JSON = "json"  # Return raw JSON
