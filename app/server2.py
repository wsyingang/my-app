from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from langserve import add_routes
from packages.pirate import chain
import uvicorn
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import time
import uuid

from packages.pirate.conversation_chain import create_conversation_chain

app = FastAPI()


# OpenAI兼容的请求模型
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None


# OpenAI兼容的响应模型
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionUsage


@app.get("/")
async def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")


# 原有的LangServe路由
add_routes(app, chain, path="/pirate")

class ChatCompletionRequestWithSession(ChatCompletionRequest):
    session_id: Optional[str] = "default_session"


# 新的OpenAI兼容端点
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequestWithSession):
    try:
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        user_input = user_messages[-1].content
        conversation_chain, memory = create_conversation_chain(request.session_id)
        result = conversation_chain.invoke({"input": user_input})
        memory.save_context({"input": user_input}, {"output": str(result)})
        print(f"[Memory Updated] Session: {request.session_id}")
        return await handle_streaming_response_with_memory(request, conversation_chain, user_input, result)

    except Exception as e:
        import traceback
        print(f"Error detail: {traceback.format_exc()}")  # 打印详细错误栈
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def handle_streaming_response_with_memory(request: ChatCompletionRequest, conversation_chain, user_input, fallback_result):
    """使用带记忆的链进行流式响应"""
    async def generate_stream():
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())

        # 发送开始chunk
        start_chunk = {
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': created_time,
            'model': request.model,
            'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]
        }
        yield f"data: {json.dumps(start_chunk)}\n\n"

        full_response = ""
        try:
            for chunk in conversation_chain.stream({"input": user_input}):
                chunk_content = extract_content_from_chunk(chunk)  # 您需要实现这个辅助函数
                if chunk_content.startswith(full_response):
                    new_content = chunk_content[len(full_response):]
                else:
                    new_content = chunk_content

                full_response = chunk_content

                if new_content:
                    content_chunk = {
                        'id': chunk_id,
                        'object': 'chat.completion.chunk',
                        'created': created_time,
                        'model': request.model,
                        'choices': [{'index': 0, 'delta': {'content': new_content}, 'finish_reason': None}]
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n"

        except Exception as e:
            print(f"Streaming failed, using fallback result: {e}")
            # 流式失败时，使用非流式生成的结果作为备选
            for char in str(fallback_result):
                content_chunk = {
                    'id': chunk_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': request.model,
                    'choices': [{'index': 0, 'delta': {'content': char}, 'finish_reason': None}]
                }
                yield f"data: {json.dumps(content_chunk)}\n\n"

        # 发送结束chunk
        end_chunk = {
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': created_time,
            'model': request.model,
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]
        }
        yield f"data: {json.dumps(end_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

def extract_prompt_text(messages: List[ChatMessage]) -> str:
    """从消息列表中提取文本用于token估算"""
    return " ".join([msg.content for msg in messages])


def estimate_tokens(text: str) -> int:
    """简单的token估算（实际使用时可以用tiktoken等库）"""
    return len(text.split())  # 按单词数粗略估算


def extract_content_from_chunk(chunk):
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, 'content'):
        return chunk.content
    if isinstance(chunk, dict):
        for key in ['output', 'result', 'text', 'content', 'response']:
            if key in chunk and chunk[key]:
                content = chunk[key]
                return content if isinstance(content, str) else str(content)
    return str(chunk)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)