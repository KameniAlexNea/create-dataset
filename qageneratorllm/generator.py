import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI

from .qa_dataclass import ChatLLMType, MCQBank, ModelName, QABank, QuestionType


class ChatLLM:
    def __init__(
        self,
        chat_type: str = ChatLLMType.OLLAMA,
        question_type: str = QuestionType.QA,
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
            from .prompts.mcq_prompt import FORMAT, HUMAN, SYSTEM

            self.qa_type = MCQBank
        elif question_type == QuestionType.QA:
            from .prompts.qa_prompt import FORMAT, HUMAN, SYSTEM

            self.qa_type = QABank

        self.human, self.system, self.format = HUMAN, SYSTEM, FORMAT
        self.n_questions = n_questions
        self.structured_llm = chat.with_structured_output(self.qa_type)

    def prepare(
        self, context: str, source: str, n_questions: int
    ) -> List[Tuple[str, str]]:
        return [
            ("system", self.system),
            (
                "user",
                self.human.format(
                    SOURCE=source,
                    N_QUESTION=n_questions,
                    CONTEXT=context,
                    FORMAT=self.format,
                ),
            ),
        ]

    def invoke(
        self, prompt: str, source: str = None, n_questions: int = None
    ) -> Union[MCQBank, QABank]:
        source = source if source else "general knowledge"
        return self.structured_llm.invoke(
            self.prepare(prompt, source, n_questions or self.n_questions)
        )

    def batch_invoke(
        self, prompts: list[str], sources: list[str] = None, n_questions: int = None
    ):
        sources = sources if sources else ["africa history"] * len(prompts)
        results = self.structured_llm.batch(
            [
                self.prepare(prompt, source, n_questions or self.n_questions)
                for prompt, source in zip(prompts, sources)
            ]
        )
        return results

    def _get_content(self, file_path: str) -> str:
        with open(file_path, "r") as file:
            context = file.read()
            root, _ = os.path.splitext(file_path)
            source = os.path.basename(root)
            return context, source

    def invoke_from_file(self, file_path: str, n_questions: int = None) -> str:
        context, source = self._get_content(file_path)
        return self.invoke(context, source, n_questions)

    def batch_invoke_from_files(self, file_paths: list[str], n_questions: int = None):
        contexts, sources = zip(
            *[self._get_content(file_path) for file_path in file_paths]
        )
        return self.batch_invoke(contexts, sources, n_questions)

    def save_result(self, result: Union[MCQBank, QABank], output_path: str):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    def batch_invoke_from_folder(self, folder_path: str, n_questions: int = None):
        folder = Path(folder_path)
        file_paths = [str(f) for f in folder.rglob("*.txt")]
        return self.batch_invoke_from_files(file_paths, n_questions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA pairs from text files")
    parser.add_argument("--input", "-i", help="Input file or folder path")
    parser.add_argument("--output", "-o", help="Output file path", default=None)
    parser.add_argument(
        "--batch", "-b", action="store_true", help="Process input as folder"
    )
    parser.add_argument(
        "--questions", "-n", type=int, default=5, help="Number of questions to generate"
    )

    args = parser.parse_args()
    chat = ChatLLM(n_questions=args.questions)

    if args.batch:
        results = chat.batch_invoke_from_folder(args.input)
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, result in enumerate(results):
                output_path = output_dir / f"qa_{i}.json"
                chat.save_result(result, str(output_path))
        else:
            print(
                json.dumps(
                    [r.model_dump() for r in results], ensure_ascii=False, indent=2
                )
            )
    else:
        result = chat.invoke_from_file(args.input)
        if args.output:
            chat.save_result(result, args.output)
        else:
            print(json.dumps(result.dict(), ensure_ascii=False, indent=2))
