import gc
import json
import logging
import os
import re
import time
from typing import List, Dict, Any, Optional

import torch
from smolagents import CodeAgent, MultiStepAgent


def _format_duration(sec: float) -> str:
    sec = int(round(sec))
    if sec <= 0:
        return "0s"
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _benchmark_worker_init(model_id: str, timestamp: str):
    """Worker initializer for ProcessPoolExecutor.
    Must be defined at module level so it can be pickled by multiprocessing.
    - Hides GPU so the worker never acquires a CUDA context (it only talks to
      the vLLM / BGE HTTP servers).
    - Persists model_id as a process-level global so the shard task can use it.
    - Installs an excepthook that logs the full traceback before the process
      dies, making BrokenProcessPool failures debuggable.
    """
    import sys, traceback as _tb
    import logging

    # Configure logging for the child process so we can see progress
    # Try to reuse the parent's log file if provided via environment.
    # Coerce a missing timestamp: prefer the passed value, then env var, then current time.
    ts = timestamp or os.environ.get("CURRENT_TIMESTAMP")

    global CURRENT_TIMESTAMP
    CURRENT_TIMESTAMP = ts
    os.environ["CURRENT_TIMESTAMP"] = CURRENT_TIMESTAMP

    # Make the real error visible before the process is torn down.
    def _worker_excepthook(exc_type, exc_value, exc_tb):
        logging.error(
            "[shard worker] Uncaught exception:\n%s",
            "".join(_tb.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = _worker_excepthook

    # Workers only call the vLLM / BGE HTTP APIs — no direct GPU use.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Persist model_id for the shard task (the global is unset in a fresh
    # spawn process).
    global MODEL_ID
    MODEL_ID = model_id


from benchmark_data.benchmark_data import BenchmarkEntry, BenchmarkData
from tools.base_noisy_tool import BaseNoisyTool

from utils.smolagents_utils import setup_agent_llm, save_conversation_as_json, convert_jsonl_to_parquet


class BenchmarkTester:
    """ A class to evaluate an agent on a benchmark dataset. """

    def tools_initializer(self, requested: Optional[List[str]] = None) -> List[BaseNoisyTool]:
        return []

    def __init__(self,
                 current_timestamp: str = None,
                 experiment_name: str = None,
                 experiment_description: str = None,
                 custom_system_prompt: str = None):
        global CURRENT_TIMESTAMP
        CURRENT_TIMESTAMP = current_timestamp
        self.current_timestamp = current_timestamp
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.custom_system_prompt = custom_system_prompt

        self.pass_at_k: int = 1

    def on_entry_start_processing(self, entry_ind: int, entry: BenchmarkEntry, agent: MultiStepAgent) -> None:
        pass

    def on_entry_end_processing(self, entry_ind: int, entry: BenchmarkEntry, agent: MultiStepAgent) -> None:
        pass

    def on_attempt_start(self, entry_ind: int, entry: BenchmarkEntry, k: int) -> None:
        pass

    def on_attempt_end(self, entry_ind: int, entry: BenchmarkEntry, k: int, response: str, ) -> Optional[
        Dict[str, Any]]:
        pass

    def on_attempt_fail(self, entry_ind: int, entry: BenchmarkEntry, k: int, e: Exception) -> None:
        pass

    @staticmethod
    def _compute_experiment_alias(
            model_id: str,
            enable_thinking: Optional[bool] = None,
    ) -> str:
        alias = model_id.split('/')[1].split("-")[0]

        if enable_thinking or 'reasoning' in model_id.lower() or 'thinking' in model_id.lower():
            alias = f"{alias}-tk"

        return alias.lower()

    def _add_thinking_mode_to_exp_description(self, llm_config: Dict[str, Any]) -> None:
        # Define the string you want to append
        disable_thinking = llm_config.get('disable_thinking', None)
        if disable_thinking == True:
            suffix = ' (thinking mode disabled)'
        elif disable_thinking == False:
            suffix = ' (thinking mode enabled)'
        else:
            suffix = ' (default thinking behavior)'

        # Regex pattern
        # \1 refers to the first capture group, \2 to the second
        pattern = r'("model_id"\s*:\s*"[^"]+)(")'
        replacement = rf'\1{suffix}\2'

        self.experiment_description = re.sub(pattern, replacement, self.experiment_description)

    def test_benchmark(
            self,
            model_id: str = None,
            benchmark: BenchmarkData = None,
            agent_run_additional_fields: Dict[str, Any] = None,
            max_steps: int = None,
            max_entries: int = None,
            enable_thinking: Optional[bool] = None,
    ):
        global MODEL_ID
        MODEL_ID = model_id
        experiment_alias = self._compute_experiment_alias(model_id, enable_thinking)

        given_args = {k: v for k, v in locals().items() if not callable(v) and not isinstance(v, BenchmarkData)}

        def _coerce_value(s):
            if s is None:
                return None
            if isinstance(s, (bool, int, float, list, dict)):
                return s
            s = str(s).strip()
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1]
            low = s.lower()
            if low == 'none':
                return None
            if low in ('true', 'false'):
                return low == 'true'
            try:
                return int(s)
            except Exception:
                try:
                    return float(s)
                except Exception:
                    try:
                        return json.loads(s)
                    except Exception:
                        return s

        logging.info(f"🧪 [{self.current_timestamp}] Making {self.experiment_name} experiment with parameters:"
                     f"\n{json.dumps(given_args, indent=4, default=str)}")

        dest_jsonl = f"./experiments/{benchmark.nickname}_{self.current_timestamp}_{experiment_alias}.jsonl"
        os.makedirs(os.path.dirname(dest_jsonl), exist_ok=True)
        open(dest_jsonl, "w", encoding="utf-8").close()
        logging.info(f"💾 Saving conversations to '{dest_jsonl}' ...")

        # Connect to the local model
        logging.info(f"🔍 Loading model {MODEL_ID}...")
        smol_local_model, llm_config = setup_agent_llm(MODEL_ID, logging=logging, enable_thinking=enable_thinking)
        self._add_thinking_mode_to_exp_description(llm_config)

        # Write the file containing the experiment description
        with open(dest_jsonl.replace(".jsonl", ".txt"), 'w') as f:
            f.write(self.experiment_description)

        # Define the additional arguments for the conversation
        addit_args = {}
        if max_steps is not None:
            addit_args['max_steps'] = max_steps

        # Add the given additional arguments
        if agent_run_additional_fields:
            addit_args.update(agent_run_additional_fields)

        # Instantiate tools
        try:
            available_tools = self.tools_initializer()
        except Exception as e:
            logging.exception("tools_initializer() failed: %s", e)
            raise

        # Initialize the CodeAgent with the local model
        logging.info("🤖 Initializing agent...")
        agent = CodeAgent(
            tools=available_tools,
            model=smol_local_model,
            add_base_tools=False,
            verbosity_level=2,
            additional_authorized_imports=["*"],
            prompt_templates=self.custom_system_prompt,
        )

        all_entries = list(benchmark.retrieve_benchmark_data())

        if max_entries is not None:
            all_entries = all_entries[:max_entries]
        entries_with_ind = list(enumerate(all_entries))

        # Progress timing: log elapsed time and estimated remaining time occasionally.

        total_entries = len(entries_with_ind)
        _processed_entries = 0
        _start_time = time.time()
        _last_log_time = _start_time
        # Log no more frequently than once per minute, and otherwise at ~10% progress steps
        _log_interval_seconds = 60
        _log_every_n = max(1, total_entries // 10)

        for entry_ind, entry in entries_with_ind:
            try:
                logging.info(f"ℹ️ Processing entry {entry_ind}...")

                self.on_entry_start_processing(entry_ind, entry, agent)

                for k in [1]:
                    self.on_attempt_start(entry_ind, entry, k)

                    logging.info(f"❓ [ATTEMPT {k}/1] Question: {entry.question}")

                    try:
                        agent_prompt = entry.question

                        # In Qwen3-14B thinking mode must be enabled/disabled from the input prompt
                        if "qwen3-14b" in model_id.lower():
                            if llm_config.get("disable_thinking") == True:
                                agent_prompt += ' /no_think'
                            elif llm_config.get("disable_thinking") == False:
                                agent_prompt += ' /think'

                        response = agent.run(
                            agent_prompt,
                            **addit_args,
                        )

                        logging.info(f"❓ [ATTEMPT {k}/1] Question: {entry.question}")
                        logging.info(f"🤖 [ATTEMPT {k}/1] Agent answer: {response}")
                        logging.info(
                            f"📌 [ATTEMPT {k}/1] Expected answer:\n{json.dumps(entry.answer, indent=2)}")
                        logging.info("-" * 50)
                        logging.info("✅ Successful run! Exiting...")

                        # Set the additional fields to save in this conversation dump
                        conversation_save_additional_fields = {
                            "id": f"{entry.id}_{k}",
                            "attempt": k,
                            "agent_answer": response,
                        }
                        conversation_save_additional_fields.update(
                            self.on_attempt_end(entry_ind, entry, k, response) or {})

                        # Save this conversation to a JSON
                        # save_conversation_as_json(agent, entry, f"{dest_folder}/{entry.task_id}.json")
                        # Save this conversation to a JSON
                        conv_json = save_conversation_as_json(
                            agent, entry,
                            additional_fields=conversation_save_additional_fields)

                    except Exception as e:
                        conversation_save_additional_fields = self.on_attempt_fail(entry_ind, entry, k, e) or {}
                        logging.exception(f"❌ [ATTEMPT {k}/1] Error at iteration {entry_ind}: {e}")
                        conv_json = json.dumps(
                            {"id": f"{entry.id}_{k}", "entry": entry.to_json(), "result": "Error", "error_msg": str(e),
                             **conversation_save_additional_fields}
                        )

                        if "Connection error" in str(e):
                            logging.error("LLM server connection failed. Please ensure the server is running.")

                    # Save the conversation JSON to a JSONL file
                    with open(dest_jsonl, "a", encoding="utf-8") as f:
                        f.write(conv_json + "\n")
                        f.flush()
                    # convert_jsonl_to_parquet(dest_jsonl, delete_jsonl=False)

                    # Wipe agent memory
                    agent.memory.reset()
                    # Release cached (but unused) CUDA memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                self.on_entry_end_processing(entry_ind, entry, agent)
            except Exception as entry_e:
                logging.exception(f"💥 Fatal error processing entry {entry_ind}: {entry_e}")
            # Periodic progress logging (elapsed time and estimated remaining)
            try:
                _processed_entries += 1
                _now = time.time()
                if (_now - _last_log_time >= _log_interval_seconds) or (_processed_entries % _log_every_n == 0):
                    _elapsed = _now - _start_time
                    _avg = _elapsed / _processed_entries if _processed_entries else 0
                    _remaining = max(0, total_entries - _processed_entries)
                    _est_remaining = _avg * _remaining
                    logging.info(
                        "⏱️ Progress: %d/%d entries processed — elapsed=%s — est_remaining=%s",
                        _processed_entries, total_entries, _format_duration(_elapsed), _format_duration(_est_remaining)
                    )
                    _last_log_time = _now
            except Exception:
                logging.debug("Failed to compute or log progress timing", exc_info=True)

        # Convert the JSONL to a Parquet file for memory and efficiency
        convert_jsonl_to_parquet(dest_jsonl, delete_jsonl=False, process_all=True)
        # Export telemetry logs at the end of the session
        # export_telemetry_logs(CURRENT_TIMESTAMP)
