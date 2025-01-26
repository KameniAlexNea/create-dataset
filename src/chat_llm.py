from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI


class ChatLLMType:
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENAI = "openai"
    XAI = "xai"


class ChatLLM:
    def __init__(self, chat_type: ChatLLMType):
        if chat_type == ChatLLMType.ANTHROPIC:
            self.chat = ChatAnthropic()
        elif chat_type == ChatLLMType.OLLAMA:
            self.chat = ChatOllama()
        elif chat_type == ChatLLMType.OPENAI:
            self.chat = ChatOpenAI()
        elif chat_type == ChatLLMType.XAI:
            self.chat = ChatXAI()
        else:
            raise ValueError("Invalid chat type")
        self.chat.with_structured_output()

    def invoke(self, prompt):
        return self.chat.invoke(prompt)
