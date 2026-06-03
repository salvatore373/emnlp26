# _Do Tool Failures Help?_ Repository
This is the repository for reproducing the experiments from "Do Tool Failures Help?" on the **MoNaCo** and **ToolQA** benchmark dataset.

## 📦 Prerequisites
Make sure you have Docker installed on your system. If you're going to use GPU-accelerated models, ensure you have the NVIDIA Container Toolkit set up.

## 📥 Data Setup
Before running the experiments, you need to download the benchmark datasets. We've included handy bash scripts to do this!

1. **MoNaCo Benchmark Data**:
   ```bash
   bash tools/monaco_preprocessing/data/download_monaco_data.sh
   ```

2. **ToolQA Benchmark Data**:
   ```bash
   bash tools/toolqa_preprocessing/download_toolqa_data.sh
   ```

## 🛠️ Building the Docker Image

Build the Docker container using the provided `Dockerfile`

```bash
docker build -t do_tool_failures_help:latest .
```

Then, **start the container**:
```bash
docker run --gpus all -it --rm \
 -v $(pwd):/workspace \
 -v /home/salrag/.cache/huggingface:/root/.cache/huggingface \
 do_tool_failures_help:latest \
 /bin/bash
```

```bash
docker run --gpus '"device=1"' -it --rm      -v $(pwd):/workspace      -v /home/salrag/.cache/huggingface:/root/.cache/huggingface      agentinnoise:latest      /bin/bash
```

## 🚀 Serving the Model

Before running the experiments, you must serve the LLM on port `8000`. You can use `llama.cpp` or `vLLM`.

**Example (llama.cpp with gemma-4-E4B-it for RTX 4090):**
```bash
llama-server -hf unsloth/gemma-4-E4B-it-GGUF:Q8_0 \
  --port 8000 --host 0.0.0.0 -ngl 99 \
  --ctx-size 65536 --batch-size 4096 --ubatch-size 512 \
  --parallel 8 --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --cont-batching
```

**Example (vLLM with gemma-4-E4B-it for RTX 4090):**
```bash
/opt/vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-E4B-it \
  --port 8000 \
  --max-model-len 65536 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --quantization fp8 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1
```

## 🧪 Running Experiments

Once your image is built and data is downloaded, you can jump inside the container to execute the tests.

1. **Inside the container, run MoNaCo**:
    ```bash
    python3 main.py \
        --experiment_name test_monaco \
        --model_id "google/gemma-4-E4B-it" \
        --max-entries 10
    ```

2. **Inside the container, run ToolQA**:
    ```bash
    python3 main.py \
        --experiment_name test_toolqa \
        --model_id "google/gemma-4-E4B-it" \
        --max-entries 10
    ```

4. **Inside the container, run the Judge**:
    ```bash
    python3 main.py \
        --experiment_name judge_experiment \
        --model_id "AtlaAI/Selene-1-Mini-Llama-3.1-8B" \
        --judge_parquet_file "/workspace/experiments/monaco_XXX_gemma.parquet" \
        --judge_output_json "/workspace/experiments/monaco_XXX_gemma_stats.json" \
        --judge_output_parquet "/workspace/experiments/monaco_XXX_gemma.parquet" \
        --judge_path_to_user_prompt "/workspace/benchmark_tests/system_prompt_judge.yaml" \
        --judge_compute_simple_stats
    ```

### 🧬 Experiment Configurations (Distractors & Errors)

By default, the benchmark scripts (`test_monaco.py` and `test_toolqa.py`) run the **clean** (no distractors) experiment. No noise is injected in the tool output.

**Example 1: Running the `clean` (baseline) experiment:**
Since this is the default, simply run the benchmark:
```bash
python3 main.py --experiment_name test_monaco --model_id "google/gemma-4-E4B-it" --max-entries 10
```

**Example 2: Running `return_general_error_message` with probability 0.9:**
You can inject errors directly via the CLI using the `--setting` and `--clean_probability` arguments. In these experiments, the `--clean_probability` is not the probability `p` mentioned in the paper, but the inverse probability of the tool returning the correct output, i.e., `--clean_probability == 1-p`.

```bash
python3 main.py \
    --experiment_name test_monaco \
    --model_id "google/gemma-4-E4B-it" \
    --setting "return_general_error_message" \
    --clean_probability 0.9 \
    --max-entries 10
```
> 💡 **Tip:** Check `python3 main.py --help` for additional configuration options, such as using custom prompts!
