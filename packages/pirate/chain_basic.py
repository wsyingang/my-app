import os

from dotenv import load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder


class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        # 处理 BaseMessage 对象
        if hasattr(text, 'content'):
            text = text.content
        return str(text).strip()


template = """You are a helpful assistant who speaks like a pirate"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = "{input}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                MessagesPlaceholder(variable_name="history"),
                                                human_message_prompt])


def get_llm() -> ChatOpenAI:
    load_dotenv()
    openai_api_key = os.environ.get('DEEPSEEK_API_KEY')
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=openai_api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.7
    )
