# AutoRound Baseline Experiment

This was the previous baseline setup, preserved as a reference run. It used
**[Intel/Qwen3.5-122B-A10B-int4-AutoRound](https://hfviewer.com/Intel/Qwen3.5-122B-A10B-int4-AutoRound)**
through the community [`spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)
recipe stack with MTP speculative decoding.

The baseline reached around **40 tok/s**, preserved **262K context**, and scored
**100/100 Tool-Eval**. It is no longer the primary setup; the primary path is the
Entrpi DFlash dense run documented in the README.

## Spark Arena

<p align="center">
  <img src="./assets/spark_arena_qwen3_5_122b.png" width="850" alt="Spark Arena community benchmark for Qwen3.5-122B on single DGX Spark">
  <br><i><a href="https://spark-arena.com/benchmark/sub1781472573286">spark-arena community benchmark</a> for Qwen3.5-122B on single DGX Spark.</i>
</p>

Selected `llama-benchy` results for the `Cogni-Brain` served model:

| Test | c1 | c2 total / req | c4 total / req |
|---|---:|---:|---:|
| `pp2048` | 1844.87 tok/s | 2146.64 / 1075.50 tok/s | 2175.14 / 585.89 tok/s |
| `tg128` | 39.61 tok/s | 65.11 / 33.42 tok/s | 82.85 / 22.44 tok/s |
| `tg128 @ d65535` | 25.10 tok/s | 30.31 / 16.29 tok/s | 34.63 / 10.61 tok/s |
| `tg128 @ d131072` | 18.27 tok/s | 20.26 / 11.01 tok/s | 1.67 / 3.70 tok/s |
| `tg128 @ d200000` | 14.65 tok/s | 14.36 / 8.11 tok/s | 0.78 / 3.27 tok/s |

## Setup Notes

The reliable path was to use the community `spark-vllm-docker` recipe through
this repo's AutoRound reference scripts instead of hand-assembling a long
`docker run`.

The helpers are idempotent by default:

* `setup/autoround_install.sh` reuses an existing `spark-vllm-docker` checkout unless `FORCE_BUILD=1` is set
* `setup/autoround_download_model.sh` skips the model download if it is already present unless `FORCE_DOWNLOAD=1` is set
* `docker/autoround_start.sh` only launches the recipe

## Runtime Stack

| Layer | Choice |
|---|---|
| Hardware | NVIDIA DGX Spark |
| Runtime | `spark-vllm-docker` |
| Recipe | `qwen3.5-122b-int4-autoround` |
| Model | `Intel/Qwen3.5-122B-A10B-int4-AutoRound` |
| Served name | `Cogni-Brain` |
| API shape | OpenAI-compatible vLLM endpoint |
| Main clients | Claude Code, NemoHermes, Open WebUI, local agents |
| Stable target | 262K context, ~40 tok/s, 100/100 Tool-Eval |

## Quick Start

```bash
# 1. Verify prerequisites and clone/build spark-vllm-docker if needed.
bash setup/autoround_install.sh

# 2. Download the model through the community helper.
bash setup/autoround_download_model.sh

# 3. Launch the recipe.
bash docker/autoround_start.sh

# 4. Follow logs.
docker logs -f spark-brain
```

Stop:

```bash
bash docker/autoround_stop.sh
```

`docker/autoround_start.sh` is the AutoRound baseline launch path.

| Setting | Default |
|---|---|
| `SPARK_VLLM_DIR` | `../spark-vllm-docker` |
| `MODEL_ID` | `Intel/Qwen3.5-122B-A10B-int4-AutoRound` |
| `RECIPE` | `qwen3.5-122b-int4-autoround` |
| `CONTAINER_NAME` | `spark-brain` |
| `SERVED_MODEL_NAME` | `Cogni-Brain` |
| `PORT` | `8000` |
| `SPECULATIVE_CONFIG` | `{"method":"qwen3_next_mtp","num_speculative_tokens":2}` |

Examples:

```bash
PORT=8001 CONTAINER_NAME=spark-brain bash docker/autoround_start.sh
SERVED_MODEL_NAME=Cogni-Brain-Qwen122 bash docker/autoround_start.sh
FORCE_BUILD=1 bash setup/autoround_install.sh
FORCE_DOWNLOAD=1 bash setup/autoround_download_model.sh
```

Extra arguments after `--` are passed through to vLLM. The AutoRound baseline
benchmarked profile relies on the recipe defaults plus the served model name and
speculative config from `docker/autoround_start.sh`.

For controlled experiments only:

```bash
bash docker/autoround_start.sh -- --max-model-len 262144 --gpu-memory-utilization 0.8
```

Do not treat higher memory utilisation, larger batching, or changed
speculative-token settings as stable until tool calling is re-tested.

The documented AutoRound baseline run used:

```text
max_model_len=262144
gpu_memory_utilization=0.8
tensor_parallel=1
max_num_batched_tokens=8192
served_model_name=Cogni-Brain
speculative_config={"method":"qwen3_next_mtp","num_speculative_tokens":2}
```

## Results

| Check | Result |
|---|---:|
| Single-stream generation | ~40 tok/s |
| Usable context | 262,144 tokens |
| Tool-eval-bench short mode | 100 / 100 |
| Open WebUI reasoning example | 5-minute reasoning trace |
| llama-benchy shallow `tg128` | 39.61 tok/s single stream; 65.11 tok/s total at c2; 82.85 tok/s total at c4 |
| llama-benchy long-context `tg128` | 25.10 tok/s at 65K; 18.27 tok/s at 131K; 14.65 tok/s at 200K |
| Runtime path | `spark-vllm-docker` recipe |
| Served model name | `Cogni-Brain` |

## Speed and Context

<p align="center">
  <img src="./assets/benchmark_speed_test_1-3.png" width="650" alt="Qwen3.5-122B speed benchmark tests 1 to 3">
</p>

<p align="center">
  <img src="./assets/benchmark_speed_test_4-5.png" width="650" alt="Qwen3.5-122B context and health benchmark tests 4 to 5">
</p>

## Tool Score

<p align="center">
  <img src="./assets/benchmark_smarts_1.png" width="650" alt="Qwen3.5-122B tool benchmark summary">
</p>

<p align="center">
  <img src="./assets/benchmark_smarts_2.png" width="650" alt="Qwen3.5-122B tool benchmark details">
</p>

<p align="center">
  <img src="./assets/benchmark_smarts_3.png" width="650" alt="Qwen3.5-122B tool benchmark deployability and responsiveness">
</p>

## Open WebUI Reasoning Example

<p align="center">
  <img src="./assets/puzzle_solution_qwen3_5_122b.png" width="850" alt="Qwen3.5-122B reasoning example in Open WebUI">
  <br><i>Open WebUI reasoning example showing a 5-minute reasoning trace before answering.</i>
</p>

## Claude Code Agent Demo

<p align="center">
  <img src="./assets/claude_code_cogni_brain_chess_app.png" width="850" alt="Claude Code using Cogni-Brain to generate a chess app">
  <br><i>Claude Code on MacBook using Cogni-Brain as the local DGX Spark backend to generate a chess app.</i>
</p>
