from unittest.mock import patch

import pytest

from qageneratorllm import ChatLLM, ChatLLMType, QuestionType
from qageneratorllm.qa_dataclass import MCQBank, QABank


def test_chat_llm_initialization():
    llm = ChatLLM()
    assert llm.question_type == QuestionType.QA
    assert llm.n_questions == 5


def test_chat_llm_invalid_type():
    with pytest.raises(ValueError):
        ChatLLM(chat_type="invalid")


@patch("langchain_ollama.ChatOllama")
def test_invoke_qa(mock_chat, sample_context, sample_qa_response):
    mock_chat.return_value.with_structured_output.return_value.invoke.return_value = (
        QABank(**sample_qa_response)
    )

    llm = ChatLLM(question_type=QuestionType.QA)
    result = llm.invoke(sample_context)

    assert isinstance(result, QABank)
    assert len(result.questions) == 1


@patch("langchain_openai.ChatOpenAI")
def test_invoke_mcq(mock_chat, sample_context, sample_mcq_response):
    mock_chat.return_value.with_structured_output.return_value.invoke.return_value = (
        MCQBank(**sample_mcq_response)
    )

    llm = ChatLLM(chat_type=ChatLLMType.OPENAI, question_type=QuestionType.MCQ)
    result = llm.invoke(sample_context)

    assert isinstance(result, MCQBank)
    assert len(result.questions) == 1


def test_invoke_from_file(temp_text_file):
    with patch.object(ChatLLM, "invoke") as mock_invoke:
        llm = ChatLLM()
        llm.invoke_from_file(temp_text_file)

        mock_invoke.assert_called_once()


def test_batch_invoke(sample_context):
    with patch.object(ChatLLM, "structured_llm") as mock_llm:
        llm = ChatLLM()
        contexts = [sample_context] * 3
        llm.batch_invoke(contexts)

        mock_llm.batch.assert_called_once()
