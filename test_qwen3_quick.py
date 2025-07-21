#!/usr/bin/env python3
"""
快速测试 Qwen3 API 连接
"""
import os
import asyncio
from eventrag.llm import openai_complete_if_cache

async def quick_test_qwen3():
    """快速测试 Qwen3 API"""
    print("🧪 快速测试 Qwen3 API...")
    
    # 检查 API 密钥
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置 QWEN_API_KEY 环境变量")
        print("请运行: export QWEN_API_KEY='your_api_key'")
        return False
    
    print(f"✅ API Key: {api_key[:20]}...")
    
    try:
        # 测试简单对话
        response = await openai_complete_if_cache(
            model="qwen-plus",  # 使用稳定的 qwen-plus 模型
            prompt="Hello, how are you?",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            max_tokens=50
        )
        
        print(f"✅ 成功! 响应: {response}")
        return True
        
    except Exception as e:
        print(f"❌ 失败! 错误: {e}")
        print("\n可能的原因:")
        print("1. API 密钥无效")
        print("2. 网络连接问题")
        print("3. API 配额已用完")
        print("4. 模型名称不正确")
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test_qwen3())
    if success:
        print("\n🎉 Qwen3 API 测试成功!")
    else:
        print("\n💥 Qwen3 API 测试失败!")
    exit(0 if success else 1)
