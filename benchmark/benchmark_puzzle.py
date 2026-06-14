#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
DGX Spark / Qwen3.5-122B hard puzzle benchmark.

The default prompt is an epistemic logic puzzle harness. It records elapsed time
and prints a <= 8 minute target verdict. You can pass --prompt-file to run the
exact puzzle text used for a screenshot or video.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


TARGET_SECONDS = 8 * 60

DEFAULT_PROMPT = """Solve this carefully. Track what each person knows after each statement.

Albert, Bernard, and Cheryl are trying to determine a secret date from a list of
possible dates. Albert is told only the month, Bernard is told only the day, and
Cheryl is told only whether the date is in the first half or second half of the
month. They then make public statements about whether they know the date and
what they can infer from the others' statements.

Give the final date if it is determined. If the information is insufficient,
explain exactly where ambiguity remains. Keep the reasoning explicit and do not
guess.
"""


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
            delta = chunk["choices"][0].get("delta", {})
            token = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
                or ""
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
    parser = argparse.ArgumentParser(description="Run the Qwen3.5-122B hard puzzle timing check.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Cogni-Brain")
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    prompt = load_prompt(args.prompt_file)

    print(c("DGX Spark Qwen3.5-122B Puzzle Benchmark", "bold"))
    print(c(f"Target: <= {TARGET_SECONDS} seconds", "dim"))
    print(c(f"Endpoint: {args.base_url}  model={args.model}", "dim"))
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
    print(f"  Elapsed: {elapsed:.1f}s")
    if result["ttft"] is not None:
        print(f"  TTFT:    {result['ttft'] * 1000:.0f}ms")
    if result["usage"]:
        print(f"  Usage:   {result['usage']}")
    if elapsed <= TARGET_SECONDS:
        print(c("  Verdict: PASS, within 8 minute target", "green"))
    else:
        print(c("  Verdict: OVER TARGET", "yellow"))


if __name__ == "__main__":
    main()
