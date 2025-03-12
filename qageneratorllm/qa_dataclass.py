import os
from typing import List

from pydantic import BaseModel, Field, field_validator


class AnswerChoice(BaseModel):
    letter: str = Field(
        description="The letter identifier for the answer choice (e.g., 'A', 'B', 'C'...)"
    )
    text: str = Field(description="The actual text content of the answer choice")


class Question(BaseModel):
    question: str = Field(description="The actual text of the question being asked")
    choices: List[AnswerChoice] = Field(
        description="List of possible answer choices for the question"
    )
    answer: List[str] = Field(
        description="List of letters corresponding to the correct answer choices. Examples: ['A', 'C']"
    )
    explanation: str = Field(
        description="Factual detailed explanation of why the marked answers are correct"
    )

    @field_validator("choices")
    @classmethod
    def validate_unique_choices(cls, choices: list[AnswerChoice]):
        for choice in choices:
            if not isinstance(choice, AnswerChoice):
                raise ValueError("Choices must be of class AnswerChoice")
        letters = {choice.letter for choice in choices}
        if len(letters) != len(choices):
            raise ValueError("Choices must have unique letter identifiers.")
        return choices

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, answers, values):
        choice_letters = {choice.letter for choice in values.data.get("choices", [])}
        for ans in answers:
            if ans not in choice_letters:
                raise ValueError(
                    f"Invalid answer choice '{ans}'. Must be one of {sorted(choice_letters)}"
                )
        return answers


class MCQBank(BaseModel):
    questions: List[Question] = Field(
        description="Collection of all questions in the question bank"
    )


class QuestionAnswer(BaseModel):
    question: str = Field(description="The actual text of the question being asked")
    answer: str = Field(description="The correct answer to the question")


class QABank(BaseModel):
    questions: List[QuestionAnswer] = Field(
        description="Collection of all questions in the question bank"
    )


class ModelName:
    ANTHROPIC = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-sonnet-20240229")
    OLLAMA = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5")
    OPENAI = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    XAI = os.getenv("XAI_MODEL_NAME", "grok-beta")


class ChatLLMType:
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENAI = "openai"
    XAI = "xai"


class QuestionType:
    MCQ = "mcq"
    QA = "qa"
