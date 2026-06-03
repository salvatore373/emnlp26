import argparse
import logging
import multiprocessing
import os
import random
from datetime import datetime
import json

import numpy as np
import torch

def _parse_bool(s):
    """Parse a boolean value. Returns True, False, or None (for unprovided arg)."""
    if s is None or s == "":
        return None
    if isinstance(s, bool):
        return s
    s_lower = str(s).lower().strip()
    if s_lower in ('true', '1', 'yes', 'on'):
        return True
    if s_lower in ('false', '0', 'no', 'off'):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {s}")

from benchmark_tests.toolqa.test_toolqa import test_toolqa2
from benchmark_tests.responses_judge import judge_experiment, compute_simple_stats
from benchmark_tests.monaco.test_monaco import test_monaco_benchmark2

CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ['CURRENT_TIMESTAMP'] = CURRENT_TIMESTAMP

# MODEL_ID = "Geodd/GLM-4.7-Flash-FP8"
# MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
# MODEL_ID = "Qwen/Qwen3.5-27B"
MODEL_ID = "google/gemma-3-12b-it"


# MODEL_ID = "NousResearch/Hermes-4-14B"
# MODEL_ID = "microsoft/Phi-4-reasoning-plus"
# MODEL_ID = "alibaba-pai/AgenticQwen-8B"
# MODEL_ID = "openai/gpt-oss-20b" # KJML/gpt-oss-20b-FP8-Dynamic EpochEcho/GPT-OSS-20B-Spider-LoRA-FP8
# MODEL_ID = "HuggingFaceTB/SmolLM3-3B"

def check_gpus():
    if torch.cuda.is_available():
        # 1. Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        logging.info(f"🚀 Total GPUs found: {gpu_count}")

        # 2. Iterate and list each one
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            logging.info(f"🖥️ GPU {i}: {gpu_name}")
    else:
        logging.info("😕 No CUDA-capable GPUs detected.")


def set_all_seeds(seed: int = 42) -> None:
    """Set seeds for Python, NumPy and PyTorch to improve reproducibility.

    Notes:
    - Complete bitwise reproducibility is not always possible (some CUDA ops are
      nondeterministic). This function sets common flags to make runs as
      deterministic as reasonably possible.
    - You may also want to set environment variables (e.g., for third-party
      libraries) or use torch.use_deterministic_algorithms(True) if you want
      errors raised when nondeterministic ops are used.
    """
    # Python
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic where possible
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Older torch versions may not have these attributes - ignore in that case
        pass


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run experiments with AgentInNoise.")

    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        choices=["test_monaco", "judge_experiment", "test_toolqa"],
        help="Name of the experiment to run."
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default=MODEL_ID,
        help="Hugging Face model ID to use. Defaults to the model stored in MODEL_ID."
    )

    parser.add_argument(
        "--model_provider",
        type=str,
        default="vllm",
        choices=["vllm", "llama"],
        help="Choose the model provider for the LLM."
    )



    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="The maximum number of steps the agent can perform in each conversation. "
             "If not provided, no limits will be set."
    )


    parser.add_argument(
        "--custom-system-prompt",
        action='store_true',
        help="Use the custom system prompt for the agent"
    )

    parser.add_argument(
        "--setting",
        type=str,
        default="clean",
        choices=["clean", "return_general_error_message"],
        help="The noise setting to apply to tools."
    )

    parser.add_argument(
        "--clean_probability",
        type=float,
        default=1.0,
        help="The probability (0.0 to 1.0) of generating the noisy output when a setting is active."
    )



    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Limit the number of processed benchmark entries. If not provided, all entries are processed."
    )



    parser.add_argument(
        "--enable-thinking",
        type=_parse_bool,
        default=None,
        nargs='?',
        const=True,
        help="Enable thinking mode for the model. Default is None. Use --enable-thinking=false to disable."
    )

    # ONLY FOR judge_experiment
    parser.add_argument("--judge_parquet_file", type=str,
                        help="(only for LLM-as-a-judge) Path to input parquet file")
    parser.add_argument("--judge_output_json", type=str,
                        help="(only for LLM-as-a-judge) Output path for the result JSON stats")
    parser.add_argument("--judge_output_parquet", type=str,
                        help="(only for LLM-as-a-judge) Output path for the result parquet file")

    parser.add_argument("--judge_path_to_user_prompt", type=str,
                        help="(only for LLM-as-a-judge) Path to the user prompt YAML for the judge (user_prompt_judge.yaml)")
    parser.add_argument("--judge_compute_simple_stats", action="store_true",
                        help="Compute statistics about the experiments that do not trigger an LLM judge")


    return parser.parse_args()


if __name__ == "__main__":
    # vLLM spawns child processes for its engine; "fork" copies the parent's
    # CUDA state which cannot be re-initialized → use "spawn" instead.
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_arguments()

    # Set the model ID globally
    MODEL_ID = args.model_id or MODEL_ID
    os.environ['CLI_MODEL_PROVIDER'] = args.model_provider

    # Set deterministic seeds for reproducibility
    set_all_seeds(123)

    if args.experiment_name == "test_toolqa":
        test_toolqa2(
            current_timestamp=CURRENT_TIMESTAMP, model_id=MODEL_ID,
            max_steps=args.max_steps,
            use_custom_system_prompt=args.custom_system_prompt, max_entries=args.max_entries,
            enable_thinking=args.enable_thinking,
            setting=args.setting, clean_probability=args.clean_probability
        )
    elif args.experiment_name == "test_monaco":
        test_monaco_benchmark2(
            current_timestamp=CURRENT_TIMESTAMP, model_id=MODEL_ID,
            max_steps=args.max_steps,
            use_custom_system_prompt=args.custom_system_prompt, max_entries=args.max_entries,
            enable_thinking=args.enable_thinking,
            setting=args.setting, clean_probability=args.clean_probability
        )
    elif args.experiment_name == "judge_experiment":
        assert (args.judge_parquet_file and args.judge_output_json and
                args.judge_output_parquet and args.judge_path_to_user_prompt),\
            ("For judge_experiment experiment, --judge_parquet_file --judge_output_json"
             " --judge_output_parquet --judge_path_to_user_prompt must be provided.")

        judge_experiment(model=MODEL_ID, parquet_file=args.judge_parquet_file, output_json=args.judge_output_json,
                         output_parquet=args.judge_output_parquet, variant="classification",
                         path_to_user_prompt=args.judge_path_to_user_prompt)
        if args.judge_compute_simple_stats:
            logging.info("🚀 Starting compute simple stats phase...")
            compute_simple_stats(
                parquet_file=args.judge_output_parquet,
                output_json=args.judge_output_json,
                output_parquet=args.judge_output_parquet
            )
    else:
        raise NotImplementedError(
            f"Unsupported experiment name '{args.experiment_name}'. Please choose from the available options.")
