from .chain_basic import chat_prompt, CommaSeparatedListOutputParser
from packages.pirate.chain_basic import get_llm

deep_seek_chat = get_llm()


def debug_print(x):
    print("\n=== [DEBUG] chat_prompt 节点输出的 Prompt ===")
    # 通常chat_prompt的输出是一个或多个PromptValue对象
    if hasattr(x, 'to_messages'):
        # 如果是ChatPromptValue，转换为消息列表查看
        messages = x.to_messages()
        for i, msg in enumerate(messages):
            print(f"[{i}] {msg.type}: {msg.content}")
    elif hasattr(x, 'to_string'):
        # 如果是StringPromptValue，查看字符串
        print(x.to_string())
    else:
        # 其他格式直接打印
        print(f"格式: {type(x)}")
        print(f"内容: {x}")
    print("=== [DEBUG] 结束 ===\n")
    # 重要：将原数据返回，确保链能继续执行
    return x


chain = chat_prompt | debug_print | deep_seek_chat | CommaSeparatedListOutputParser()
