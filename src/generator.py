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
        n_questions: int = 5,
    ):
        if chat_type == ChatLLMType.ANTHROPIC:
            chat = ChatAnthropic(model=ModelName.ANTHROPIC)
        elif chat_type == ChatLLMType.OLLAMA:
            chat = ChatOllama(model=ModelName.OLLAMA)
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
        self.n_questions = n_questions

        self.structured_llm = chat.with_structured_output(qa_type, include_raw=True)

    def prepare(self, context, source: str, n_questions: int) -> str:
        return [
            ("system", self.system),
            (
                "user",
                self.human.format(
                    SOURCE=source, N_QUESTION=n_questions, CONTEXT=context
                ),
            ),
        ]

    def invoke(
        self, prompt: str, source: str = None, n_questions: int = None
    ) -> Union[MCQBank, QABank]:
        source = source if source else "africa history"
        return self.structured_llm.invoke(
            self.prepare(prompt, source, n_questions or self.n_questions)
        )

    def batch_invoke(
        self, prompts: list[str], sources: list[str] = None, n_questions: int = None
    ) -> list[Union[MCQBank, QABank]]:
        sources = sources if sources else ["africa history"] * len(prompts)
        return self.structured_llm.batch(
            [
                self.prepare(prompt, source, n_questions or self.n_questions)
                for prompt, source in zip(prompts, sources)
            ]
        )

    def get_content(self, file_path: str) -> str:
        with open(file_path, "r") as file:
            context = file.read()
            root, _ = os.path.splitext(file_path)
            source = os.path.basename(root)
            return context, source

    def invoke_from_file(
        self, file_path: str, n_questions: int = None
    ) -> Union[MCQBank, QABank]:
        with open(file_path, "r") as file:
            context, source = self.get_content(file_path)
            return self.invoke(context, source, n_questions)

    def batch_invoke_from_files(
        self, file_paths: list[str], n_questions: int = None
    ) -> list[Union[MCQBank, QABank]]:
        contexts, sources = zip(
            *[self.get_content(file_path) for file_path in file_paths]
        )
        return self.batch_invoke(contexts, sources, n_questions)


if __name__ == "__main__":
    chat = ChatLLM()
    print(
        chat.invoke_from_file(
            "data/texts/CAD Antériorité des Civilisations Nègres/0d8ab37478294a4bb8d6932e19900ce8.txt"
        )
    )
