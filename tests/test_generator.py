import json

import pytest

from qageneratorllm import LLMProviderType, QuestionGenerator, QuestionType
from qageneratorllm.qa_dataclass import (
    MultipleChoiceQuestionBank,
    OpenEndedQuestionBank,
    OutputType,
)


def test_question_generator_initialization():
    generator = QuestionGenerator()
    assert generator.qa_type == OpenEndedQuestionBank
    assert generator.n_questions == 5


def test_question_generator_invalid_type():
    with pytest.raises(ValueError):
        QuestionGenerator(provider_type="invalid")


def test_invoke_qa(monkeypatch, sample_context, sample_qa_response):
    class MockStructuredLLM:
        def invoke(self, _):
            return sample_qa_response

    def mock_init(self, *args, **kwargs):
        self.qa_type = OpenEndedQuestionBank
        self.n_questions = 5
        self.human, self.system, self.template_format = "", "", ""
        self.structured_llm = MockStructuredLLM()
        self.output_type = OutputType.DATACLASS

    monkeypatch.setattr(QuestionGenerator, "__init__", mock_init)

    generator = QuestionGenerator(question_type=QuestionType.QA)
    result = generator.invoke(sample_context)

    assert isinstance(result, OpenEndedQuestionBank)
    assert len(result.open_ended_questions) == 1


def test_invoke_mcq(monkeypatch, sample_context, sample_mcq_response):
    class MockLLM:
        def with_structured_output(self):
            return self

        def invoke(self, _):
            return sample_mcq_response

    def mock_init(self, *args, **kwargs):
        self.qa_type = kwargs.get("question_type")
        self.n_questions = 5
        self.human, self.system, self.template_format = "", "", ""
        self.structured_llm = MockLLM()
        self.output_type = OutputType.DATACLASS

    monkeypatch.setattr(QuestionGenerator, "__init__", mock_init)

    generator = QuestionGenerator(
        provider_type=LLMProviderType.OPENAI, question_type=QuestionType.MCQ
    )
    result = generator.invoke(sample_context)

    assert isinstance(result, MultipleChoiceQuestionBank)
    assert len(result.mcq_questions) == 1


def test_invoke_from_file(monkeypatch, temp_text_file):
    class MockLLM:
        def invoke(self, _):
            pass

    def mock_init(self, *args, **kwargs):
        self.qa_type = OpenEndedQuestionBank
        self.n_questions = 5
        self.human, self.system, self.template_format = "", "", ""
        self.structured_llm = MockLLM()
        self.output_type = OutputType.DATACLASS

    monkeypatch.setattr(QuestionGenerator, "__init__", mock_init)

    generator = QuestionGenerator()
    generator.invoke_from_file(temp_text_file)


def test_batch_invoke(monkeypatch, sample_context):
    class MockLLM:
        def batch(self, _):
            pass

    def mock_init(self, *args, **kwargs):
        self.qa_type = OpenEndedQuestionBank
        self.n_questions = 5
        self.human, self.system, self.template_format = "", "", ""
        self.structured_llm = MockLLM()
        self.output_type = OutputType.DATACLASS

    monkeypatch.setattr(QuestionGenerator, "__init__", mock_init)

    generator = QuestionGenerator()
    contexts = [sample_context] * 3
    generator.batch_invoke(contexts)


def test_invoke_json_qa(monkeypatch, sample_context, sample_qa_response):
    """Test invoking the generator with JSON output format for QA questions."""
    json_response = '{"result": "success"}'

    class MockLLM:
        def invoke(self, _):
            class MockResponse:
                content = f"```json\n{json_response}\n```"

            return MockResponse()

    def mock_init(self, *args, **kwargs):
        self.qa_type = OpenEndedQuestionBank
        self.n_questions = 5
        self.human, self.system, self.template_format = "", "", ""
        self.llm = MockLLM()
        self.output_type = OutputType.JSON

    # Get sample data as JSON
    expected_data = {"result": "success"}

    monkeypatch.setattr(QuestionGenerator, "__init__", mock_init)
    monkeypatch.setattr(
        "llm_output_parser.parse_json",
        lambda _: (
            json_response
            if isinstance(json_response, str)
            else json.dumps(json_response)
        ),
    )

    generator = QuestionGenerator(output_type=OutputType.JSON)
    result = generator.invoke(sample_context)

    assert isinstance(result, dict)
    assert result == expected_data


def test_invoke_json_mcq(monkeypatch, sample_context, sample_mcq_response):
    """Test invoking the generator with JSON output format for MCQ questions."""
    # Convert sample response to JSON string
    json_response = json.dumps(sample_mcq_response.model_dump())

    class MockLLM:
        def invoke(self, _):
            class MockResponse:
                content = f"```json\n{json_response}\n```"

            return MockResponse()

    def mock_init(self, *args, **kwargs):
        self.qa_type = MultipleChoiceQuestionBank
        self.n_questions = 5
        self.human, self.system, self.template_format = "", "", ""
        self.llm = MockLLM()
        self.output_type = OutputType.JSON

    monkeypatch.setattr(QuestionGenerator, "__init__", mock_init)
    monkeypatch.setattr("llm_output_parser.parse_json", lambda _: json_response)

    generator = QuestionGenerator(
        question_type=QuestionType.MCQ, output_type=OutputType.JSON
    )
    result = generator.invoke(sample_context)

    assert isinstance(result, dict)
    assert "mcq_questions" in result
    assert len(result["mcq_questions"]) == 1
    assert (
        result["mcq_questions"][0]["question_text"]
        == "What was the purpose of pyramids in Ancient Egypt?"
    )


def test_batch_invoke_json(monkeypatch, sample_context, sample_qa_response):
    """Test batch invoking the generator with JSON output format."""
    # Use model_dump to get JSON representation
    json_response = json.dumps(sample_qa_response.model_dump())

    class MockLLM:
        def batch(self, _):
            class MockResponse:
                content = f"```json\n{json_response}\n```"

            return [MockResponse(), MockResponse()]

    def mock_init(self, *args, **kwargs):
        self.qa_type = OpenEndedQuestionBank
        self.n_questions = 5
        self.human, self.system, self.template_format = "", "", ""
        self.llm = MockLLM()
        self.output_type = OutputType.JSON

    monkeypatch.setattr(QuestionGenerator, "__init__", mock_init)
    monkeypatch.setattr("llm_output_parser.parse_json", lambda _: json_response)

    generator = QuestionGenerator(output_type=OutputType.JSON)
    contexts = [sample_context] * 2
    results = generator.batch_invoke(contexts)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(result, dict) for result in results)
    assert all("open_ended_questions" in result for result in results)
