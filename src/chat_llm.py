import os
from typing import Union

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI

from .qa_dataclass import MCQBank, QABank


class ModelName:
    ANTHROPIC = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-sonnet-20240229")
    OLLAMA = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
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


class ChatLLM:
    def __init__(
        self,
        chat_type: ChatLLMType = ChatLLMType.OLLAMA,
        question_type: QuestionType = QuestionType.QA,
    ):
        if chat_type == ChatLLMType.ANTHROPIC:
            chat = ChatAnthropic(model_name=ModelName.ANTHROPIC)
        elif chat_type == ChatLLMType.OLLAMA:
            chat = ChatOllama(model_name=ModelName.OLLAMA)
        elif chat_type == ChatLLMType.OPENAI:
            chat = ChatOpenAI(model=ModelName.OPENAI)
        elif chat_type == ChatLLMType.XAI:
            chat = ChatXAI(model=ModelName.XAI)
        else:
            raise ValueError("Invalid chat type")
        if question_type == QuestionType.MCQ:
            from .prompts.mcq_prompt import human, system

            qa_type = MCQBank
        elif question_type == QuestionType.QA:
            from .prompts.qa_prompt import human, system

            qa_type = QABank
        self.human, self.system = human, system

        self.structured_llm = chat.with_structured_output(qa_type)

    def invoke(self, prompt) -> Union[MCQBank, QABank]:
        return self.structured_llm.invoke(prompt)
