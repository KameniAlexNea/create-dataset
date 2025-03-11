# QAGeneratorLLM

A Python package for generating questions and answers using various LLM providers.

## Features

- Support for multiple LLM providers (Anthropic, Ollama, OpenAI, XAI)
- Generate both Multiple Choice Questions (MCQ) and Question-Answer (QA) pairs
- Structured output using Pydantic models
- Batch processing support
- File-based context input

## Installation

```bash
pip install qageneratorllm
```

## Usage

```python
from qageneratorllm import ChatLLM, ChatLLMType, QuestionType

# Initialize with default settings (Ollama + QA format)
llm = ChatLLM()

# Generate QA from text
result = llm.invoke("Your context text here")

# Generate MCQ using OpenAI
llm = ChatLLM(chat_type=ChatLLMType.OPENAI, question_type=QuestionType.MCQ)
result = llm.invoke("Your context text here")

# Generate from file
result = llm.invoke_from_file("path/to/your/file.txt")
```

## Environment Variables

- `ANTHROPIC_MODEL_NAME`: Anthropic model name (default: claude-3-sonnet-20240229)
- `OLLAMA_MODEL_NAME`: Ollama model name (default: deepseek-r1)
- `OPENAI_MODEL_NAME`: OpenAI model name (default: gpt-4)
- `XAI_MODEL_NAME`: XAI model name (default: grok-beta)

## License

MIT
