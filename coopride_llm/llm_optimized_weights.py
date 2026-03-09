"""
CoopRide LLM API Test Utilities

Standalone script for testing OpenAI and Zhipu AI API connectivity.
The actual LLM weight optimization logic is in agent/ module.

Usage:
    cd coopride_llm
    python llm_optimized_weights.py
"""

import os
import traceback

# ==================== API Defaults ====================
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_OPENAI_BASE_URL = "https://api.chatanywhere.tech/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

DEFAULT_ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
DEFAULT_ZHIPU_MODEL = "glm-4.7"


def test_openai():
    """Test OpenAI API connectivity."""
    from openai import OpenAI

    print("=" * 70)
    print("Testing OpenAI API")
    print("=" * 70)
    print(f"  Base URL: {DEFAULT_OPENAI_BASE_URL}")
    print(f"  Model: {DEFAULT_OPENAI_MODEL}")

    try:
        client = OpenAI(api_key=DEFAULT_OPENAI_API_KEY, base_url=DEFAULT_OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, please introduce yourself briefly."}
            ],
            temperature=0.7,
            max_tokens=256
        )
        print("\n[SUCCESS]")
        print("-" * 40)
        print(response.choices[0].message.content)
        print("-" * 40)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()

    print("=" * 70)


def test_zhipu():
    """Test Zhipu AI API connectivity."""
    from zai import ZhipuAiClient

    print("=" * 70)
    print("Testing Zhipu API")
    print("=" * 70)
    print(f"  Model: {DEFAULT_ZHIPU_MODEL}")

    try:
        client = ZhipuAiClient(api_key=DEFAULT_ZHIPU_API_KEY)
        response = client.chat.completions.create(
            model=DEFAULT_ZHIPU_MODEL,
            messages=[
                {"role": "system", "content": "你是一个有用的AI助手。"},
                {"role": "user", "content": "你好，请介绍一下自己。"}
            ],
            temperature=0.6
        )
        print("\n[SUCCESS]")
        print("-" * 40)
        print(response.choices[0].message.content)
        print("-" * 40)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()

    print("=" * 70)


if __name__ == "__main__":
    # test_openai()
    test_zhipu()
