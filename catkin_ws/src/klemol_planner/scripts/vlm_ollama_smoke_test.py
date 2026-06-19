#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Ollama Python API smoke test.")
    parser.add_argument("--model", default="gemma3", help="Ollama model name, exactly as shown by `ollama list`.")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--prompt", default="Hello! Reply with one short sentence.", help="Prompt text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[VLM_SMOKE] importing ollama client...")
    try:
        from ollama import Client
    except ImportError as exc:
        raise SystemExit("Python package `ollama` is not installed. Try: pip install ollama") from exc

    print(f"[VLM_SMOKE] host={args.host}")
    print(f"[VLM_SMOKE] model={args.model}")
    print(f"[VLM_SMOKE] prompt={args.prompt!r}")

    client = Client(host=args.host)
    start = time.time()
    print("[VLM_SMOKE] sending chat request...")
    response = client.chat(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        options={"temperature": 0.0},
    )
    elapsed = time.time() - start

    message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    print(f"[VLM_SMOKE] response_time_sec={elapsed:.2f}")
    print("[VLM_SMOKE] response:")
    print(content)


if __name__ == "__main__":
    main()
