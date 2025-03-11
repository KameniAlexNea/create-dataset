from setuptools import find_packages, setup

setup(
    name="qageneratorllm",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain-anthropic",
        "langchain-ollama",
        "langchain-openai",
        "langchain-xai",
        "pydantic",
    ],
    python_requires=">=3.8",
    author="Original Author",
    description="A package for generating Q&A and MCQ using various LLM providers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
