# DFlash Entrpi Runtime Experiment

This was a follow-up DFlash experiment for Qwen3.5-122B on DGX Spark, using the
Entrpi [`qwen3.5-122B-A10B-on-spark`](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark)
runtime instead of the earlier `spark-vllm-docker` DFlash run.

The important change: this run kept tool calling intact. It reached a stronger
short-output speed profile while preserving **100/100** on `tool-eval-bench`
short mode.

## Result

| Check | Result |
|---|---:|
| Average TPS | 47.7 tok/s |
| Peak TPS | 50.3 tok/s |
| Average TTFT | 158 ms |
| Max usable context | ~261,497 tokens |
| Tool-eval-bench short mode | 100 / 100 |
| Tool scenarios | 15 passed, 0 partial, 0 failed |
| Quality | 100 / 100 |
| Responsiveness | 49 / 100, median turn 3.1s |
| Deployability | 85 / 100 |
| Token usage | 35,885 tokens |
| Tool benchmark runtime | 110.3s |

Working read: this runtime is a viable DFlash path for the local `Cogni-Brain`
use case. Unlike the first DFlash experiment, tool calling did not collapse.

## Runtime Notes

The served model was exposed as `Cogni-Brain` on `localhost:8000`.

Tool-eval reported:

| Runtime detail | Value |
|---|---|
| Model | `bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid` |
| Alias | `Cogni-Brain` |
| Engine | `vLLM 0.23.0+aeon.sm121a.dflash` |
| Quantization | INT4 |
| Max context | 262,144 tokens |

The upstream runtime describes this as Qwen3.5-122B-A10B on DGX Spark with
DFlash speculative decode, a prebuilt hybrid INT4+FP8 checkpoint, and a
ready-to-run Docker/vLLM installer.

## Commands

### First Time

```bash
git clone https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark.git ~/cogni-brain
cd ~/cogni-brain
sed -i 's/--served-model-name qwen/--served-model-name "Cogni-Brain"/' runtime/serve.sh
grep "served-model-name" runtime/serve.sh
REPO_DIR="$HOME/cogni-brain" ./install.sh --start
```

The fresh install downloads the runtime image, model checkpoint, and DFlash
drafter once. In this run, the server was ready about three minutes after
downloads completed.

### Verify

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

The model list should show `Cogni-Brain`.

### Test Inference

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Cogni-Brain",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64
  }'
```

### Subsequent Starts

```bash
cd ~/cogni-brain
REPO_DIR="$HOME/cogni-brain" ./install.sh --start --no-pull --no-download
```

With everything already downloaded, startup was about three minutes to READY.

### Stop and Restart

```bash
docker stop qwen-spark
docker rm qwen-spark

cd ~/cogni-brain
REPO_DIR="$HOME/cogni-brain" ./install.sh --start --no-pull --no-download
```

### Logs

```bash
docker logs -f qwen-spark
```

## Speed Details

| Test | Result |
|---|---:|
| Warmup | 47.0 tok/s, 186 ms TTFT |
| Baseline run 1 | 50.3 tok/s, 207 ms TTFT |
| Baseline run 2 | 46.4 tok/s, 133 ms TTFT |
| Baseline run 3 | 46.4 tok/s, 135 ms TTFT |
| 50-token output | 69.8 tok/s |
| 150-token output | 49.8 tok/s |
| 300-token output | 50.5 tok/s |
| 600-token output | 38.5 tok/s |
| 1000-token output | 38.7 tok/s |

## Concurrency

| Sessions | Total TPS | Per-session TPS |
|---:|---:|---:|
| 1 | 42.4 tok/s | 42.4 tok/s |
| 2 | 53.7 tok/s | 26.9 tok/s |
| 3 | 63.7 tok/s | 21.2 tok/s |
| 4 | 61.0 tok/s | 15.2 tok/s |

## Long Context

| Approx context | TPS |
|---:|---:|
| ~4,095 tokens | 80.1 tok/s |
| ~8,180 tokens | 67.6 tok/s |
| ~16,352 tokens | 65.7 tok/s |
| ~32,695 tokens | 61.9 tok/s |
| ~65,381 tokens | 54.5 tok/s |
| ~99,759 tokens | 48.7 tok/s |
| ~130,753 tokens | 41.5 tok/s |
| ~159,609 tokens | 40.4 tok/s |
| ~199,509 tokens | 34.0 tok/s |
| ~230,431 tokens | 34.0 tok/s |
| ~261,497 tokens | 31.5 tok/s |

## Tool Calling

`tool-eval-bench --short` passed all 15 scenarios:

| Category | Score |
|---|---:|
| Tool Selection | 100% |
| Parameter Precision | 100% |
| Multi-Step Chains | 100% |
| Restraint & Refusal | 100% |
| Error Recovery | 100% |

This is the key practical result: the Entrpi DFlash runtime improved the speed profile without the
tool-call failure pattern seen in the first DFlash attempt.

## Screenshot Evidence

<p align="center">
  <img src="./assets/dflash_entrpi_speed_test_1-3.png" width="650" alt="DFlash Entrpi speed benchmark tests 1 to 3">
</p>

<p align="center">
  <img src="./assets/dflash_entrpi_speed_test_4-5.png" width="650" alt="DFlash Entrpi context and health benchmark tests 4 to 5">
</p>

<p align="center">
  <img src="./assets/dflash_entrpi_tool_smarts_1.png" width="650" alt="DFlash Entrpi tool benchmark startup and first scenario">
</p>

<p align="center">
  <img src="./assets/dflash_entrpi_tool_smarts_2.png" width="650" alt="DFlash Entrpi tool benchmark scenario pass list">
</p>

<p align="center">
  <img src="./assets/dflash_entrpi_tool_smarts_3.png" width="650" alt="DFlash Entrpi tool benchmark summary showing 100 out of 100">
</p>
