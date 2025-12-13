import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from langserve import add_routes
from packages.pirate import chain
import uvicorn
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import json
import time
import uuid
import numpy as np
from numpy.linalg import norm

from packages.pirate.conversation_chain import create_conversation_chain

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加嵌入相关的导入
try:
    from sentence_transformers import SentenceTransformer

    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Embedding functions will not work.")

app = FastAPI()

# 全局嵌入模型实例
embedding_model = None

# 简单内存存储本地信息
local_info_store = []


# 初始化嵌入模型
def init_embedding_model():
    """初始化嵌入模型"""
    global embedding_model
    if EMBEDDING_AVAILABLE and embedding_model is None:
        try:
            # 使用一个轻量级且效果好的模型
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded successfully")

            # 添加示例信息
            add_example_info()

        except Exception as e:
            print(f"Failed to load embedding model: {e}")


def add_example_info():
    """添加示例信息到知识库"""
    example_info = "xw is a prate"

    # 检查是否已存在
    if not any(info["content"] == example_info for info in local_info_store):
        if embedding_model is not None:
            # 生成嵌入向量
            embedding = embedding_model.encode(example_info, convert_to_numpy=True)

            # 添加到存储
            local_info_store.append({
                "id": len(local_info_store) + 1,
                "content": example_info,
                "embedding": embedding,
                "metadata": {"type": "example", "source": "manual"}
            })
            print(f"Added example info to knowledge base: {example_info}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    return float(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-10))


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
    use_local_info: Optional[bool] = True  # 是否使用本地信息
    similarity_threshold: Optional[float] = 0.2  # 相似度阈值


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


# 嵌入相关模型
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = "text-embedding-ada-002"
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    index: int
    embedding: List[float]
    object: str = "embedding"


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]


class LocalInfo(BaseModel):
    content: str
    metadata: Optional[Dict] = None


class InfoSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    threshold: Optional[float] = 0.7


@app.get("/")
async def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")


# 原有的LangServe路由
add_routes(app, chain, path="/pirate")


class ChatCompletionRequestWithSession(ChatCompletionRequest):
    session_id: Optional[str] = "default_session"


def retrieve_relevant_info(query: str, threshold: float = 0.2) -> List[Dict]:
    """检索与查询相关的本地信息"""
    if not embedding_model or not local_info_store:
        return []

    try:
        # 生成查询的嵌入向量
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)

        # 计算相似度
        similarities = []
        for info in local_info_store:
            similarity = cosine_similarity(query_embedding, info["embedding"])
            if similarity >= threshold:
                similarities.append({
                    "content": info["content"],
                    "similarity": similarity,
                    "metadata": info["metadata"],
                    "id": info["id"]
                })

        # 按相似度排序
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities

    except Exception as e:
        print(f"Error retrieving info: {e}")
        return []


def enhance_prompt_with_local_info(user_input: str, local_info_results: List[Dict]) -> str:
    """使用本地信息增强提示"""
    if not local_info_results:
        return user_input

    # 构建上下文
    context_parts = ["以下是相关的本地信息："]
    for i, result in enumerate(local_info_results, 1):
        context_parts.append(f"{i}. {result['content']} (相似度: {result['similarity']:.2f})")

    context = "\n".join(context_parts)

    # 组合增强后的提示
    enhanced_prompt = f"""{context}

基于以上信息，请回答以下问题：
{user_input}

请确保回答时参考上述信息，如果信息相关就使用，不相关则忽略。"""

    return enhanced_prompt


# 新的OpenAI兼容端点（增强版）
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequestWithSession):
    try:
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        user_input = user_messages[-1].content

        # 如果启用本地信息检索
        local_info_context = ""
        retrieved_info = []

        if request.use_local_info and embedding_model and local_info_store:
            retrieved_info = retrieve_relevant_info(user_input, request.similarity_threshold)

            if retrieved_info:
                # 使用本地信息增强提示
                enhanced_input = enhance_prompt_with_local_info(user_input, retrieved_info)
                local_info_context = "\n\n[使用的本地信息："
                local_info_context += "; ".join([info['content'] for info in retrieved_info])
                local_info_context += "]"
                print(f"[Info Retrieval] Found {len(retrieved_info)} relevant pieces")
            else:
                enhanced_input = user_input
                local_info_context = "\n\n[未找到相关本地信息]"
                print(f"[Info Retrieval] No relevant info found for query: {user_input}")
        else:
            enhanced_input = user_input
            if not request.use_local_info:
                print(f"[Info Retrieval] Local info retrieval disabled for this request")

        # 创建对话链
        conversation_chain, memory = create_conversation_chain(request.session_id)

        # 使用增强后的输入
        result = conversation_chain.invoke({"input": enhanced_input})

        # 保存到记忆（保存原始输入和增强后的输出）
        memory.save_context(
            {"input": user_input},
            {"output": str(result) + local_info_context}
        )

        print(f"[Memory Updated] Session: {request.session_id}")

        return await handle_streaming_response_with_memory(
            request, conversation_chain, enhanced_input, result, retrieved_info
        )

    except Exception as e:
        import traceback
        print(f"Error detail: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# 添加本地信息管理端点
@app.post("/local-info/add")
async def add_local_info(info: LocalInfo):
    """添加新的本地信息到知识库"""
    if not embedding_model:
        raise HTTPException(status_code=501, detail="Embedding model not available")

    try:
        # 检查是否已存在
        for existing_info in local_info_store:
            if existing_info["content"] == info.content:
                return {
                    "id": existing_info["id"],
                    "content": info.content,
                    "status": "already_exists",
                    "message": "Info already in store"
                }

        # 生成嵌入向量
        embedding = embedding_model.encode(info.content, convert_to_numpy=True)

        # 添加到存储
        new_id = len(local_info_store) + 1
        local_info_store.append({
            "id": new_id,
            "content": info.content,
            "embedding": embedding,
            "metadata": info.metadata or {}
        })

        print(f"[Local Info] Added new info: {info.content}")

        return {
            "id": new_id,
            "content": info.content,
            "status": "added",
            "embedding_dim": len(embedding),
            "store_size": len(local_info_store)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding info: {str(e)}")


@app.get("/local-info/list")
async def list_local_info():
    """列出所有本地信息"""
    # 返回时移除嵌入向量以节省带宽
    simplified_list = []
    for info in local_info_store:
        simplified_list.append({
            "id": info["id"],
            "content": info["content"],
            "metadata": info["metadata"],
            "embedding_dim": len(info["embedding"]) if "embedding" in info else 0
        })

    return {
        "count": len(local_info_store),
        "items": simplified_list
    }


@app.post("/local-info/search")
async def search_local_info(request: InfoSearchRequest):
    """搜索相关的本地信息"""
    if not embedding_model or not local_info_store:
        raise HTTPException(status_code=501, detail="Embedding system not available")

    try:
        # 生成查询嵌入向量
        query_embedding = embedding_model.encode(request.query, convert_to_numpy=True)

        # 计算相似度
        results = []
        for info in local_info_store:
            similarity = cosine_similarity(query_embedding, info["embedding"])
            if similarity >= request.threshold:
                results.append({
                    "id": info["id"],
                    "content": info["content"],
                    "similarity": similarity,
                    "metadata": info["metadata"]
                })

        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # 取前k个
        results = results[:request.top_k]

        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "store_size": len(local_info_store)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching info: {str(e)}")


@app.delete("/local-info/clear")
async def clear_local_info():
    """清除所有本地信息"""
    global local_info_store
    count = len(local_info_store)
    local_info_store = []

    # 重新添加示例信息
    add_example_info()

    return {
        "status": "cleared",
        "removed_count": count,
        "current_count": len(local_info_store)
    }


@app.delete("/local-info/{info_id}")
async def delete_local_info(info_id: int):
    """删除指定ID的本地信息"""
    global local_info_store

    initial_count = len(local_info_store)
    local_info_store = [info for info in local_info_store if info["id"] != info_id]

    removed = initial_count - len(local_info_store)

    if removed > 0:
        return {
            "status": "deleted",
            "id": info_id,
            "removed": True,
            "current_count": len(local_info_store)
        }
    else:
        raise HTTPException(status_code=404, detail=f"Info with id {info_id} not found")


# 嵌入端点保持不变
@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    """创建文本嵌入向量"""
    if not EMBEDDING_AVAILABLE or embedding_model is None:
        raise HTTPException(
            status_code=501,
            detail="Embedding model not available. Please install sentence-transformers: pip install sentence-transformers torch"
        )

    try:
        # 处理输入：可以是字符串或字符串列表
        inputs = [request.input] if isinstance(request.input, str) else request.input

        if not inputs:
            raise HTTPException(status_code=400, detail="Input cannot be empty")

        # 生成嵌入向量
        embeddings = embedding_model.encode(inputs, convert_to_numpy=True)

        # 转换为列表格式
        embeddings_list = embeddings.tolist()

        # 构建响应数据
        data = []
        for i, embedding in enumerate(embeddings_list):
            data.append(EmbeddingData(
                index=i,
                embedding=embedding,
                object="embedding"
            ))

        # 估算token使用量（简单估算：按单词数）
        total_tokens = sum(len(text.split()) for text in inputs)

        return EmbeddingResponse(
            data=data,
            model="all-MiniLM-L6-v2",  # 实际使用的模型名称
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


async def handle_streaming_response_with_memory(
        request: ChatCompletionRequest,
        conversation_chain,
        user_input,
        fallback_result,
        retrieved_info: List[Dict] = None
):
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
                chunk_content = extract_content_from_chunk(chunk)
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


# 应用启动时初始化嵌入模型
@app.on_event("startup")
async def startup_event():
    init_embedding_model()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
