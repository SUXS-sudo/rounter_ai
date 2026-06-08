"""Verify MiMo API connectivity and configuration."""

from __future__ import annotations

import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from models.config import settings


def verify() -> bool:
    print("=" * 50)
    print("MiMo API 连通性验证")
    print("=" * 50)
    print()

    # 1. Check config
    print(f"  Base URL : {settings.mimo_base_url}")
    print(f"  Model    : {settings.mimo_model}")
    print(f"  API Key  : {settings.mimo_api_key[:8]}...{'*' * 4}")
    print()

    if not settings.mimo_api_key or settings.mimo_api_key == "your-api-key-here":
        print("[FAIL] API Key 未配置，请在 .env 文件中设置 ROUTE_PLANNER_MIMO_API_KEY")
        return False

    # 2. Try API call
    from core.mimo_client import chat

    print("  正在测试 API 连接...")
    start = time.time()
    try:
        reply = chat(
            system_prompt="You are a test assistant. Reply with exactly: OK",
            user_prompt="ping",
            temperature=0,
            max_tokens=16,
        )
        elapsed = time.time() - start
    except Exception as exc:
        elapsed = time.time() - start
        print()
        print(f"[FAIL] API 调用失败 ({elapsed:.1f}s)")
        print(f"       {exc}")
        return False

    print(f"  响应耗时 : {elapsed:.1f}s")
    print(f"  模型回复 : {reply[:100]}")
    print()

    if reply:
        print("[PASS] API 连接正常，配置正确！")
        return True
    else:
        print("[WARN] API 返回为空，请检查模型配置")
        return False


if __name__ == "__main__":
    ok = verify()
    sys.exit(0 if ok else 1)
