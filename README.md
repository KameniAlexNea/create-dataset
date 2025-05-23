# QAGeneratorLLM

A Python package for generating educational questions and answers using various LLM providers.

## Features

- Support for multiple LLM providers (Anthropic, Ollama, OpenAI, XAI)
- Generate both Multiple Choice Questions (MCQ) and Open-Ended Questions
- Choose between structured dataclass or raw JSON output formats
- Batch processing support
- File-based context input

## Installation

```bash
pip install qageneratorllm
```

## Usage

```python
from qageneratorllm import QuestionGenerator, LLMProviderType, QuestionType, OutputType

# Initialize with default settings (Ollama + Open-Ended Questions)
generator = QuestionGenerator()

# Generate open-ended questions from text
result = generator.invoke("Your context text here")

# Generate Multiple Choice Questions using OpenAI
generator = QuestionGenerator(
    provider_type=LLMProviderType.OPENAI,
    question_type=QuestionType.MCQ
)
result = generator.invoke("Your context text here")

# Generate from file
result = generator.invoke_from_file("path/to/your/file.txt")

# Generate with JSON output instead of dataclass
generator = QuestionGenerator(
    question_type=QuestionType.MCQ,
    output_type=OutputType.JSON
)
result = generator.invoke("Your context text here")
# result will be a dictionary instead of a Pydantic model
```

## Command Line Usage

```bash
# Basic usage
python -m qageneratorllm.generator --input input.txt --output questions.json

# Generate MCQs in batch mode from a folder of text files
python -m qageneratorllm.generator --input texts_folder/ --output results/ --batch --questions 5 --output-type json
```

## Gradio Interface

The package includes a Gradio web interface for interactive question generation:

```bash
# Launch the Gradio app
python -m qageneratorllm.gradio_app
```

With the Gradio interface, you can:
- Upload documents (.txt, .md, .pdf)
- View document chunks separated by headers
- Filter chunks by header level
- Select specific chunks for question generation
- Choose question type (MCQ or open-ended)
- Select LLM provider
- Generate questions interactively

## Environment Variables

- `ANTHROPIC_MODEL_NAME`: Anthropic model name (default: claude-3-sonnet-20240229)
- `OLLAMA_MODEL_NAME`: Ollama model name (default: qwen2.5)
- `OPENAI_MODEL_NAME`: OpenAI model name (default: gpt-4o)
- `XAI_MODEL_NAME`: XAI model name (default: grok-beta)

## License

MIT
