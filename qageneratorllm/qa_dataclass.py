import os
from typing import List

from pydantic import BaseModel, Field


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
        description="List of letters corresponding to the correct answer choices. examples : ['A', 'C']"
    )
    explanation: str = Field(
        description="Factual detailed explanation of why the marked answers are correct"
    )


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
    OLLAMA = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1")  # "qwen2.5:3b"
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
