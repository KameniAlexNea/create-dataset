import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from llm_output_parser import parse_json

from .qa_dataclass import (
    LLMProviderType,
    ModelName,
    MultipleChoiceQuestionBank,
    OpenEndedQuestionBank,
    OutputType,
    QuestionType,
)


def get_example_json(
    example: Union[MultipleChoiceQuestionBank, OpenEndedQuestionBank],
) -> str:
    """Get a JSON string representation of the example dataclass."""
    return json.dumps(example.model_dump(), indent=2)


class QuestionGenerator:
    """A class that generates multiple-choice or open-ended questions using various LLM providers."""

    def __init__(
        self,
        provider_type: str = LLMProviderType.OLLAMA,
        question_type: str = QuestionType.QA,
        n_questions: int = 5,
        model_name: str = None,
        output_type: str = OutputType.DATACLASS,
        num_ctx: int = 4096,
    ):
        if provider_type == LLMProviderType.ANTHROPIC:
            anthropic_kwargs = {"model": model_name or ModelName.ANTHROPIC}
            if num_ctx is not None:
                anthropic_kwargs["num_ctx"] = num_ctx
            self.llm = ChatAnthropic(**anthropic_kwargs)
        elif provider_type == LLMProviderType.OLLAMA:
            ollama_kwargs = {"model": model_name or ModelName.OLLAMA}
            if num_ctx is not None:
                ollama_kwargs["num_ctx"] = num_ctx
            self.llm = ChatOllama(**ollama_kwargs)
        elif provider_type == LLMProviderType.OPENAI:
            openai_kwargs = {"model": model_name or ModelName.OPENAI}
            if num_ctx is not None:
                openai_kwargs["num_ctx"] = num_ctx
            self.llm = ChatOpenAI(**openai_kwargs)
        elif provider_type == LLMProviderType.XAI:
            xai_kwargs = {"model": model_name or ModelName.XAI}
            if num_ctx is not None:
                xai_kwargs["num_ctx"] = num_ctx
            self.llm = ChatXAI(**xai_kwargs)
        elif provider_type == LLMProviderType.GROQ:
            groq_kwargs = {"model": model_name or ModelName.GROQ}
            if num_ctx is not None:
                groq_kwargs["num_ctx"] = num_ctx
            self.llm = ChatGroq(**groq_kwargs)
        else:
            raise ValueError("Invalid LLM provider type")

        if question_type == QuestionType.MCQ:
            from .prompts.mcq_prompt import HUMAN, SYSTEM

            self.qa_type = MultipleChoiceQuestionBank
        elif question_type == QuestionType.QA:
            from .prompts.qa_prompt import HUMAN, SYSTEM

            self.qa_type = OpenEndedQuestionBank

        self.human, self.system, self.template_format = (
            HUMAN,
            SYSTEM,
            get_example_json(self.qa_type.example()),
        )
        self.n_questions = n_questions
        self.output_type = output_type

        # Only create structured_llm if using dataclass output
        if self.output_type == OutputType.DATACLASS:
            self.structured_llm = self.llm.with_structured_output(self.qa_type)

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
                    FORMAT=self.template_format,
                ),
            ),
        ]

    def invoke(
        self, prompt: str, source: str = None, n_questions: int = None
    ) -> Union[MultipleChoiceQuestionBank, OpenEndedQuestionBank, Dict[str, Any]]:
        source = source if source else "general knowledge"
        prepared_messages = self.prepare(
            prompt, source, n_questions or self.n_questions
        )

        if self.output_type == OutputType.DATACLASS:
            return self.structured_llm.invoke(prepared_messages)
        else:
            # For JSON output, parse the raw response
            raw_response = self.llm.invoke(prepared_messages)
            content = parse_json(raw_response.content)
            return content

    def batch_invoke(
        self, prompts: list[str], sources: list[str] = None, n_questions: int = None
    ):
        sources = sources if sources else ["africa history"] * len(prompts)
        prepared_messages = [
            self.prepare(prompt, source, n_questions or self.n_questions)
            for prompt, source in zip(prompts, sources, strict=False)
        ]

        if self.output_type == OutputType.DATACLASS:
            return self.structured_llm.batch(prepared_messages)
        else:
            # For JSON output, process each response
            raw_responses = self.llm.batch(prepared_messages)
            results = []
            for response in raw_responses:
                content = parse_json(response.content)
                results.append(content)
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
            *[self._get_content(file_path) for file_path in file_paths], strict=False
        )
        return self.batch_invoke(contexts, sources, n_questions)

    def save_result(
        self,
        result: Union[
            MultipleChoiceQuestionBank, OpenEndedQuestionBank, Dict[str, Any]
        ],
        output_path: str,
    ):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            if self.output_type == OutputType.DATACLASS:
                json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
            else:
                json.dump(result, f, ensure_ascii=False, indent=2)

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
    parser.add_argument(
        "--output-type",
        choices=[OutputType.DATACLASS, OutputType.JSON],
        default=OutputType.DATACLASS,
        help="Type of output format",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Context window size (num_ctx) for LLMs qui le supportent (ex: Ollama)",
    )

    args = parser.parse_args()
    generator = QuestionGenerator(
        n_questions=args.questions, output_type=args.output_type, num_ctx=args.num_ctx
    )

    if args.batch:
        results = generator.batch_invoke_from_folder(args.input)
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, result in enumerate(results):
                output_path = output_dir / f"qa_{i}.json"
                generator.save_result(result, str(output_path))
        else:
            print(
                json.dumps(
                    [r.model_dump() for r in results], ensure_ascii=False, indent=2
                )
            )
    else:
        result = generator.invoke_from_file(args.input)
        if args.output:
            generator.save_result(result, args.output)
        else:
            print(json.dumps(result.dict(), ensure_ascii=False, indent=2))
