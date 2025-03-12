import pytest

from qageneratorllm import ChatLLM, ChatLLMType, QuestionType
from qageneratorllm.qa_dataclass import MCQBank, QABank


def test_chat_llm_initialization():
    llm = ChatLLM()
    assert llm.qa_type == QABank
    assert llm.n_questions == 5


def test_chat_llm_invalid_type():
    with pytest.raises(ValueError):
        ChatLLM(chat_type="invalid")


def test_invoke_qa(monkeypatch, sample_context, sample_qa_response):
    class MockStructuredLLM:
        def invoke(self, _):
            return sample_qa_response

    def mock_init(self, *args, **kwargs):
        self.qa_type = QABank
        self.n_questions = 5
        self.human, self.system, self.format = "", "", ""
        self.structured_llm = MockStructuredLLM()

    monkeypatch.setattr(ChatLLM, "__init__", mock_init)

    llm = ChatLLM(question_type=QuestionType.QA)
    result = llm.invoke(sample_context)

    assert isinstance(result, QABank)
    assert len(result.questions) == 1


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

    monkeypatch.setattr(ChatLLM, "__init__", mock_init)

    llm = ChatLLM(chat_type=ChatLLMType.OPENAI, question_type=QuestionType.MCQ)
    result = llm.invoke(sample_context)

    assert isinstance(result, MCQBank)
    assert len(result.questions) == 1


def test_invoke_from_file(monkeypatch, temp_text_file):
    class MockLLM:
        def invoke(self, _):
            pass

    def mock_init(self, *args, **kwargs):
        self.qa_type = QABank
        self.n_questions = 5
        self.human, self.system, self.format = "", "", ""
        self.structured_llm = MockLLM()

    monkeypatch.setattr(ChatLLM, "__init__", mock_init)

    llm = ChatLLM()
    llm.invoke_from_file(temp_text_file)


def test_batch_invoke(monkeypatch, sample_context):
    class MockLLM:
        def batch(self, _):
            pass

    def mock_init(self, *args, **kwargs):
        self.qa_type = QABank
        self.n_questions = 5
        self.human, self.system, self.format = "", "", ""
        self.structured_llm = MockLLM()

    monkeypatch.setattr(ChatLLM, "__init__", mock_init)

    # monkeypatch.setattr(ChatLLM, "structured_llm", MockLLM())

    llm = ChatLLM()
    contexts = [sample_context] * 3
    llm.batch_invoke(contexts)
