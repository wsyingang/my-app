from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough

from packages.pirate.basic_chain import debug_print
from packages.pirate.chain_basic import chat_prompt, CommaSeparatedListOutputParser, get_llm

memories = {}


def get_memory(session_id: str) -> ConversationBufferMemory:
    """根据session_id获取或创建对应的memory"""
    if session_id not in memories:
        memories[session_id] = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
    return memories[session_id]


llm_model = get_llm()


def create_conversation_chain(session_id: str):
    memory = get_memory(session_id)
    chain_with_memory = (
            RunnablePassthrough.assign(
                history=lambda x: memory.load_memory_variables({})["history"]
            )
            | chat_prompt
            | debug_print
            | llm_model
            | CommaSeparatedListOutputParser()
    )
    return chain_with_memory, memory
