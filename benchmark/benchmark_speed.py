#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
DGX Spark / Qwen3.5-122B speed and context benchmark.

Tests TPS, TTFT, concurrent sessions, endpoint health, and the 262K context
target for the spark-vllm-docker recipe setup.
"""

import argparse
import json
import statistics
import sys
import threading
import time
from datetime import datetime

import requests


TARGET_TPS = 40
TARGET_CONTEXT = 262144

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


def header(title):
    line = "-" * 60
    print(f"\n{c(line, 'cyan')}")
    print(f"{c('  ' + title, 'bold')}")
    print(f"{c(line, 'cyan')}")


def result_line(label, value, unit="", color="green"):
    print(f"  {c(label.ljust(30), 'dim')} {c(str(value), color)} {unit}")


def make_prompt(n_words):
    base = ("The quick brown fox jumps over the lazy dog. " * 50).split()
    words = (base * ((n_words // len(base)) + 1))[:n_words]
    return " ".join(words) + "\n\nSummarize the above text in one sentence."


def count_tokens_approx(text):
    return int(len(text.split()) * 1.33)


def stream_completion(host, port, model, prompt, max_tokens=200, timeout=120, debug=False):
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first = None
    full_text = ""
    usage_tokens = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return None, None, 0, "", f"HTTP {resp.status_code}: {resp.text[:300]}"

            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                if debug:
                    print(f"  RAW: {data[:200]}")
                try:
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage_tokens = chunk["usage"].get("completion_tokens")
                    delta = chunk["choices"][0].get("delta", {})
                    combined = (
                        delta.get("content")
                        or delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or ""
                    )
                    if combined:
                        if t_first is None:
                            t_first = time.perf_counter()
                        full_text += combined
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except requests.exceptions.Timeout:
        return None, None, 0, "", "Timeout"
    except requests.exceptions.ConnectionError:
        return None, None, 0, "", "Connection refused - is spark-brain running?"
    except Exception as exc:
        return None, None, 0, "", str(exc)

    t_end = time.perf_counter()
    if t_first is None:
        return None, None, 0, full_text, "No tokens generated"

    ttft_ms = (t_first - t_start) * 1000
    generation_time = t_end - t_first
    tokens = usage_tokens if usage_tokens and usage_tokens > 0 else max(1, len(full_text) // 4)
    tps = tokens / generation_time if generation_time > 0 else 0
    return round(ttft_ms), round(tps, 1), tokens, full_text, None


def get_served_models(host, port):
    try:
        response = requests.get(f"http://{host}:{port}/v1/models", timeout=5)
        if response.status_code != 200:
            return [], f"HTTP {response.status_code}: {response.text[:200]}"
        data = response.json().get("data", [])
        models = [item.get("id") for item in data if item.get("id")]
        return models, None
    except Exception as exc:
        return [], str(exc)


def validate_served_model(host, port, model):
    models, error = get_served_models(host, port)
    if error:
        print(c(f"\nWARN Could not read /v1/models: {error}", "yellow"))
        return
    if not models:
        print(c("\nWARN /v1/models did not report any served models.", "yellow"))
        return
    if model not in models:
        print(c(f"\nFAILED Requested model '{model}' is not served by this endpoint.", "red"))
        print(c(f"  Available model(s): {', '.join(models)}", "dim"))
        sys.exit(1)
    print(c(f"  OK model is served ({model})", "green"))


def test_baseline_tps(host, port, model, debug=False):
    header("TEST 1 - Baseline TPS")
    prompt = "Explain speculative decoding in local LLM serving in practical terms."
    runs = 3
    results = []

    print("  Running warmup request...")
    warmup_ttft, warmup_tps, warmup_tokens, _, warmup_err = stream_completion(
        host, port, model, prompt, max_tokens=300, debug=debug
    )
    if warmup_err:
        print(c(f"  WARN Warmup failed: {warmup_err}", "yellow"))
    else:
        print(f"  Warmup: TTFT={warmup_ttft}ms TPS={warmup_tps} tokens={warmup_tokens}")

    print(f"  Running {runs} measured requests...")
    for run in range(runs):
        ttft, tps, tokens, _, err = stream_completion(
            host, port, model, prompt, max_tokens=300, debug=debug
        )
        if err:
            print(c(f"  FAILED Run {run + 1}: {err}", "red"))
            continue
        results.append((ttft, tps, tokens))
        color = "green" if tps >= TARGET_TPS else "yellow" if tps >= 30 else "red"
        print(f"  Run {run + 1}: TTFT={ttft}ms TPS={c(str(tps), color)} tokens={tokens}")
        time.sleep(1)

    if not results:
        return 0, 0

    avg_tps = round(statistics.mean(item[1] for item in results), 1)
    avg_ttft = round(statistics.mean(item[0] for item in results))
    peak_tps = max(item[1] for item in results)
    result_line("Average TPS", avg_tps, "tok/s", "green" if avg_tps >= TARGET_TPS else "yellow")
    result_line("Peak TPS", peak_tps, "tok/s", "green" if peak_tps >= TARGET_TPS else "yellow")
    result_line("Average TTFT", avg_ttft, "ms", "yellow")
    return avg_tps, peak_tps


def test_tps_vs_length(host, port, model):
    header("TEST 2 - TPS vs Output Length")
    lengths = [50, 150, 300, 600, 1000]
    prompt = "Write a detailed but practical explanation of how MoE routing affects local inference."

    print(f"  {'Output tokens'.ljust(18)} {'TPS'.ljust(12)} {'TTFT'}")
    print(f"  {'-' * 44}")
    for max_tok in lengths:
        ttft, tps, tokens, _, err = stream_completion(host, port, model, prompt, max_tokens=max_tok)
        if err:
            print(f"  {str(max_tok).ljust(18)} {c('FAILED: ' + err, 'red')}")
        else:
            color = "green" if tps >= TARGET_TPS else "yellow" if tps >= 30 else "red"
            print(f"  {(str(tokens) + ' tok').ljust(18)} {c(str(tps) + ' tok/s', color).ljust(20)} {ttft}ms")
        time.sleep(1)


def test_concurrent(host, port, model, max_concurrent=4):
    header("TEST 3 - Concurrent Sessions")
    prompts = [
        "Explain why prefix caching matters for long-context agents.",
        "Summarize the tradeoffs between speed and context length in local serving.",
        "Give a debugging checklist for vLLM memory pressure.",
        "Explain how tool-calling reliability should be evaluated.",
    ]

    for concurrency in range(1, max_concurrent + 1):
        results = [None] * concurrency
        errors = []

        def run_request(idx):
            ttft, tps, tokens, _, err = stream_completion(
                host, port, model, prompts[idx % len(prompts)], max_tokens=200
            )
            if err:
                errors.append(err)
            else:
                results[idx] = (tokens, tps, ttft)

        threads = [threading.Thread(target=run_request, args=(idx,)) for idx in range(concurrency)]
        t_start = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - t_start

        valid = [row for row in results if row is not None]
        if valid:
            total_tokens = sum(tokens for tokens, _, _ in valid)
            total_tps = round(total_tokens / elapsed, 1) if elapsed > 0 else 0
            per_session = round(total_tps / concurrency, 1)
            color = (
                "red"
                if errors
                else "green"
                if per_session >= 20
                else "yellow"
                if per_session >= 10
                else "red"
            )
            print(
                f"  {(str(concurrency) + ' session(s)').ljust(14)} "
                f"total={c(str(total_tps) + ' tok/s', color).ljust(22)} "
                f"per-session={c(str(per_session) + ' tok/s', color)}"
                f"{c(' partial failures=' + str(len(errors)), 'red') if errors else ''}"
            )
        else:
            print(f"  {(str(concurrency) + ' session(s)').ljust(14)} {c('ALL FAILED: ' + str(errors[0]), 'red')}")
        time.sleep(3)


def test_context_window(host, port, model):
    header("TEST 4 - 262K Context Target")
    sizes = [4096, 8192, 16384, 32768, 65536, 100000, 131072, 160000, 200000, 231000, 262144]
    last_working = 0

    print(f"  {'Context tokens'.ljust(20)} {'Result'.ljust(20)} {'TPS'}")
    print(f"  {'-' * 50}")
    for size in sizes:
        prompt = make_prompt(int(size / 1.33) + 2)
        actual_tokens = count_tokens_approx(prompt)
        ttft, tps, _, _, err = stream_completion(host, port, model, prompt, max_tokens=96, timeout=300)
        if err:
            print(f"  ~{(str(actual_tokens) + ' tok'):15} {c('FAILED ' + err[:28], 'red')}")
            break
        last_working = actual_tokens
        color = "green" if actual_tokens >= TARGET_CONTEXT else "yellow"
        tps_color = "green" if tps >= 20 else "yellow" if tps >= 10 else "red"
        print(f"  ~{(str(actual_tokens) + ' tok'):15} {c('OK', color):20} {c(str(tps) + ' tok/s', tps_color)}")
        time.sleep(2)

    if last_working:
        result_line("Max working context", f"~{last_working:,}", "tokens", "green" if last_working >= TARGET_CONTEXT else "yellow")
    return last_working


def test_health(host, port):
    header("TEST 5 - Health and Metrics")
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        result_line(
            "Health endpoint",
            "OK" if response.status_code == 200 else f"HTTP {response.status_code}",
            color="green" if response.status_code == 200 else "red",
        )
    except Exception as exc:
        result_line("Health endpoint", f"FAILED: {exc}", color="red")

    try:
        response = requests.get(f"http://{host}:{port}/metrics", timeout=5)
        if response.status_code == 200:
            found = False
            for line in response.text.splitlines():
                if "gpu_cache_usage_perc" in line and not line.startswith("#"):
                    found = True
                    pct = round(float(line.split()[-1]) * 100, 1)
                    result_line("KV cache used", f"{pct}%", color="green" if pct < 85 else "yellow")
                if "num_requests_running" in line and not line.startswith("#"):
                    found = True
                    result_line("Requests running", line.split()[-1])
            if not found:
                result_line("Metrics", "available, no known vLLM counters found", color="yellow")
        else:
            result_line("Metrics endpoint", f"HTTP {response.status_code}", color="yellow")
    except Exception:
        result_line("Metrics endpoint", "not available", color="yellow")


def print_summary(avg_tps, peak_tps, max_context, host, port, model):
    header("SUMMARY")
    result_line("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_line("Endpoint", f"http://{host}:{port}")
    result_line("Model", model)
    result_line("Average TPS", avg_tps, "tok/s", "green" if avg_tps >= TARGET_TPS else "yellow")
    result_line("Peak TPS", peak_tps, "tok/s", "green" if peak_tps >= TARGET_TPS else "yellow")
    result_line(
        "Max usable context",
        f"~{max_context:,}" if max_context is not None else "skipped",
        "tokens",
        "green" if max_context is not None and max_context >= TARGET_CONTEXT else "yellow",
    )
    print()
    if max_context is None:
        if avg_tps >= TARGET_TPS:
            print(c("  PASS speed target. Context target was skipped.", "green"))
        else:
            print(c("  REVIEW speed target not fully reached. Context target was skipped.", "yellow"))
    elif avg_tps >= TARGET_TPS and max_context >= TARGET_CONTEXT:
        print(c("  PASS target profile: 40 TPS class with 262K context.", "green"))
    else:
        print(c("  REVIEW target profile not fully reached in this run.", "yellow"))
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark DGX Spark Qwen3.5-122B spark-vllm-docker setup.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", default="Cogni-Brain")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument("--skip-concurrent", action="store_true")
    args = parser.parse_args()

    print(f"\n{c('DGX Spark Qwen3.5-122B Benchmark', 'bold')}")
    print(f"{c('Target:', 'dim')} 40 tok/s, 262K context  endpoint=http://{args.host}:{args.port}  model={args.model}")

    try:
        response = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
        if response.status_code != 200:
            print(c(f"\nFAILED endpoint not healthy (HTTP {response.status_code}). Is spark-brain running?", "red"))
            sys.exit(1)
    except Exception as exc:
        print(c(f"\nFAILED Cannot reach endpoint: {exc}", "red"))
        print(c("  Make sure spark-brain is running and port 8000 is open.", "dim"))
        sys.exit(1)

    print(c("  OK endpoint is reachable", "green"))
    validate_served_model(args.host, args.port, args.model)

    avg_tps, peak_tps = test_baseline_tps(args.host, args.port, args.model, debug=args.debug)
    test_tps_vs_length(args.host, args.port, args.model)

    if not args.skip_concurrent:
        test_concurrent(args.host, args.port, args.model)

    max_context = None
    if not args.skip_context:
        max_context = test_context_window(args.host, args.port, args.model)

    test_health(args.host, args.port)
    print_summary(avg_tps, peak_tps, max_context, args.host, args.port, args.model)


if __name__ == "__main__":
    main()
