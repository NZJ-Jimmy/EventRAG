import asyncio
from eventrag.llm import jina_embedding

async def test_jina_embedding():
    texts = ["测试文本1", "测试文本2"]
    result = await jina_embedding(texts)
    print("jina_embedding 返回:", result)
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == len(texts)

if __name__ == "__main__":
    asyncio.run(test_jina_embedding())
