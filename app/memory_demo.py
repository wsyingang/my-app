from langchain.memory import ConversationBufferMemory


memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")

if __name__=="__main__":
    print(memory.chat_memory)