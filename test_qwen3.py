#!/usr/bin/env python3
"""
测试 Qwen3 API 连接和功能
"""
import os
import asyncio
from eventrag.llm import openai_complete_if_cache
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

# Qwen-3 API configuration
async def qwen3_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
    """Qwen-3 API wrapper using OpenAI-compatible interface"""
    return await openai_complete_if_cache(
        model="Qwen3-32B",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        # base_url="http://10.10.202.242:2099/v1",  # Automatically use the base URL from environment variables
        # api_key=os.getenv("QWEN_API_KEY") or "sk-",  # Automatically use the API key from environment variables
        **kwargs
    )

async def test_qwen3_basic():
    """测试基本的文本生成功能"""
    print("🧪 测试 1: 基本文本生成")
    try:
        response = await qwen3_complete("你好，请介绍一下你自己。")
        print(f"✅ 成功! 响应: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ 失败! 错误: {e}")
        return False

async def test_qwen3_with_system_prompt():
    """测试带系统提示的功能"""
    print("\n🧪 测试 2: 带系统提示的文本生成")
    try:
        system_prompt = "你是一个友善的AI助手，请用简洁明了的方式回答问题。"
        response = await qwen3_complete(
            "什么是人工智能？", 
            system_prompt=system_prompt
        )
        print(f"✅ 成功! 响应: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ 失败! 错误: {e}")
        return False

async def test_qwen3_with_history():
    """测试带对话历史的功能"""
    print("\n🧪 测试 3: 带对话历史的文本生成")
    try:
        history_messages = [
            {"role": "user", "content": "我的名字是小明"},
            {"role": "assistant", "content": "你好小明！很高兴认识你。"}
        ]
        response = await qwen3_complete(
            "你还记得我的名字吗？", 
            history_messages=history_messages
        )
        print(f"✅ 成功! 响应: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ 失败! 错误: {e}")
        return False

async def test_qwen3_english():
    """测试英文对话功能"""
    print("\n🧪 测试 4: 英文对话")
    try:
        response = await qwen3_complete(
            "What are the main themes in Charles Dickens' novels?",
            system_prompt="You are a literature expert. Please provide detailed and insightful answers."
        )
        print(f"✅ 成功! 响应: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ 失败! 错误: {e}")
        return False

async def test_qwen3_parameters():
    """测试不同参数配置"""
    print("\n🧪 测试 5: 参数配置")
    try:
        response = await qwen3_complete(
            "写一首关于春天的诗",
            max_tokens=200,
            temperature=0.7
        )
        print(f"✅ 成功! 响应: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ 失败! 错误: {e}")
        return False

async def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("🚀 开始测试 Qwen3 API 功能")
    print("=" * 50)
    
    # 检查 API 密钥
    # api_key = os.getenv("QWEN_API_KEY")
    # if not api_key or api_key.startswith("sk-be98534"):
    #     print("⚠️  警告: 请设置正确的 QWEN_API_KEY 环境变量")
    #     print("   export QWEN_API_KEY='your_actual_api_key'")
    #     print()
    
    tests = [
        test_qwen3_basic,
        test_qwen3_with_system_prompt,
        test_qwen3_with_history,
        test_qwen3_english,
        test_qwen3_parameters
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! Qwen3 API 工作正常")
    elif passed > 0:
        print("⚠️  部分测试通过，请检查失败的测试")
    else:
        print("💥 所有测试失败，请检查 API 配置")
    
    print("=" * 50)
    
    return passed == total

def main():
    """主函数"""
    try:
        success = asyncio.run(run_all_tests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
        exit(1)
    except Exception as e:
        print(f"\n\n💥 测试运行失败: {e}")
        exit(1)

if __name__ == "__main__":
    main()
