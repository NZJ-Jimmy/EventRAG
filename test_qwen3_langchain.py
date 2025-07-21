#!/usr/bin/env python3
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

async def main():
    # 1. 初始化LLM
    llm = ChatOpenAI(
        model="Qwen3-32B",
        openai_api_key="EMPTY",
        openai_api_base="http://10.10.202.242:2099/v1",
        temperature=0.7
    )

    # 2. 异步调用测试
    print("=== 异步测试 ===")
    try:
        response = await llm.ainvoke([HumanMessage(content="请说'pong'")])
        print(f"成功: {response.content}")  # 或 response.text
    except Exception as e:
        print(f"失败: {e}")

    # 3. 同步调用测试
    print("\n=== 同步测试 ===")
    try:
        sync_response = llm.invoke([HumanMessage(content="请说'ping'")])
        print(f"成功: {sync_response.content}")
    except Exception as e:
        print(f"失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())