#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
DGX Spark / Qwen3.5-122B hard word-puzzle benchmark.

The default prompt is the Albert/Bernard/Cheryl word puzzle used for the local
reasoning timing run. It records elapsed time and prints the final solve time.
You can pass --prompt-file to run a different exact puzzle text.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


DEFAULT_PROMPT = """A teacher writes six words on a board: "cat dog has max dim tag." She gives three students, Albert, Bernard and Cheryl each a piece of paper with one letter from one of the words. Then she asks, "Albert, do you know the word?" Albert immediately replies yes. She asks, "Bernard, do you know the word?" He thinks for a moment and replies, "Yes." Then, she asks Cheryl the same question. She thinks and then replies, "Yes." What is the word?"""


COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "reset": "\033[0m",
    "dim": "\033[2m",
}


def c(text, color):
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def load_prompt(path):
    if path is None:
        return DEFAULT_PROMPT
    return Path(path).read_text(encoding="utf-8")


def stream_chat(base_url, model, prompt, max_tokens, temperature, timeout):
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    first_token_at = None
    started_at = time.perf_counter()
    text_parts = []
    usage = {}

    with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data = decoded[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            token = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
                or ""
            )
            if isinstance(token, list):
                token = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in token
                )
            if token:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                print(token, end="", flush=True)
                text_parts.append(token)

    ended_at = time.perf_counter()
    return {
        "text": "".join(text_parts),
        "elapsed": ended_at - started_at,
        "ttft": (first_token_at - started_at) if first_token_at else None,
        "usage": usage,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the Qwen3.5-122B hard word-puzzle timing check.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Cogni-Brain")
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--hide-prompt", action="store_true", help="Do not print the prompt before sending it")
    args = parser.parse_args()

    prompt = load_prompt(args.prompt_file)

    print(c("DGX Spark Qwen3.5-122B Puzzle Benchmark", "bold"))
    print(c(f"Endpoint: {args.base_url}  model={args.model}", "dim"))
    print()
    if not args.hide_prompt:
        print(c("PROMPT", "cyan"))
        print(prompt)
        print(c("MODEL OUTPUT", "cyan"))
        print()

    try:
        result = stream_chat(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
    except requests.exceptions.ConnectionError:
        print(c("\nFAILED Cannot reach endpoint. Is spark-brain running?", "red"))
        sys.exit(1)
    except Exception as exc:
        print(c(f"\nFAILED {exc}", "red"))
        sys.exit(1)

    elapsed = result["elapsed"]
    print()
    print()
    print(c("SUMMARY", "cyan"))
    print(f"  Final solve time: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    if result["ttft"] is not None:
        print(f"  TTFT:    {result['ttft'] * 1000:.0f}ms")
    if result["usage"]:
        print(f"  Usage:   {result['usage']}")


if __name__ == "__main__":
    main()
