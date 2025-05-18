import pytest

from qageneratorllm import QuestionGenerator, ChatLLMType, QuestionType
from qageneratorllm.qa_dataclass import MultipleChoiceQuestionBank, OpenEndedQuestionBank


def test_question_generator_initialization():
    generator = QuestionGenerator()
    assert generator.qa_type == OpenEndedQuestionBank
    assert generator.n_questions == 5


def test_question_generator_invalid_type():
    with pytest.raises(ValueError):
        QuestionGenerator(chat_type="invalid")


def test_invoke_qa(monkeypatch, sample_context, sample_qa_response):
    class MockStructuredLLM:
        def invoke(self, _):
            return sample_qa_response

    def mock_init(self, *args, **kwargs):
        self.qa_type = OpenEndedQuestionBank
        self.n_questions = 5
        self.human, self.system, self.format = "", "", ""
        self.structured_llm = MockStructuredLLM()

    monkeypatch.setattr(QuestionGenerator, "__init__", mock_init)

    generator = QuestionGenerator(question_type=QuestionType.QA)
    result = generator.invoke(sample_context)

    assert isinstance(result, OpenEndedQuestionBank)
    assert len(result.open_ended_questions) == 1


def test_invoke_mcq(monkeypatch, sample_context, sample_mcq_response):
    class MockChat:
        def with_structured_output(self):
            return self

        def invoke(self, _):
            return sample_mcq_response

    def mock_init(self, *args, **kwargs):
        self.qa_type = kwargs.get("question_type")
        self.n_questions = 5
        self.human, self.system, self.format = "", "", ""
        self.structured_llm = MockChat()

    monkeypatch.setattr(QuestionGenerator, "__init__", mock_init)

    generator = QuestionGenerator(chat_type=ChatLLMType.OPENAI, question_type=QuestionType.MCQ)
    result = generator.invoke(sample_context)

    assert isinstance(result, MultipleChoiceQuestionBank)
    assert len(result.questions) == 1


def test_invoke_from_file(monkeypatch, temp_text_file):
    class MockLLM:
        def invoke(self, _):
            pass

    def mock_init(self, *args, **kwargs):
        self.qa_type = OpenEndedQuestionBank
        self.n_questions = 5
        self.human, self.system, self.format = "", "", ""
        self.structured_llm = MockLLM()

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
        self.human, self.system, self.format = "", "", ""
        self.structured_llm = MockLLM()

    monkeypatch.setattr(QuestionGenerator, "__init__", mock_init)

    # monkeypatch.setattr(ChatLLM, "structured_llm", MockLLM())

    generator = QuestionGenerator()
    contexts = [sample_context] * 3
    generator.batch_invoke(contexts)
