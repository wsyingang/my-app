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
        # 提取用户消息（取最后一条用户消息）
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        user_input = user_messages[-1].content
        chain_input = user_input

        conversation_chain, memory = create_conversation_chain(request.session_id)
        result = conversation_chain.invoke({"text": user_input})
        memory.save_context({"input": user_input}, {"output": result})
        print(f"[Memory Updated] Session: {request.session_id}, Input saved.")

        return await handle_streaming_response(request, chain_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def handle_streaming_response(request: ChatCompletionRequest, chain_input):
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

        # 使用真正的流式输出
        full_response = ""
        try:
            # 使用 chain.stream 而不是 chain.invoke
            for chunk in chain.stream({"text": chain_input}):
                # 提取每个chunk的内容
                if hasattr(chunk, 'content'):
                    chunk_content = chunk.content
                elif isinstance(chunk, dict) and 'output' in chunk:
                    chunk_content = chunk['output']
                elif isinstance(chunk, dict) and 'result' in chunk:
                    chunk_content = chunk['result']
                elif isinstance(chunk, dict) and 'text' in chunk:
                    chunk_content = chunk['text']
                else:
                    chunk_content = str(chunk)

                # 只发送新增的内容
                if chunk_content.startswith(full_response):
                    new_content = chunk_content[len(full_response):]
                else:
                    new_content = chunk_content

                full_response = chunk_content

                # 发送内容chunk
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
            # 如果流式输出失败，回退到非流式
            print(f"Streaming error: {e}")
            result = chain.invoke({"text": chain_input})

            # 提取响应内容
            if hasattr(result, 'content'):
                response_content = result.content
            elif isinstance(result, dict) and 'output' in result:
                response_content = result['output']
            elif isinstance(result, dict) and 'result' in result:
                response_content = result['result']
            elif isinstance(result, dict) and 'text' in result:
                response_content = result['text']
            else:
                response_content = str(result)

            # 模拟流式输出
            for i, char in enumerate(response_content):
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)