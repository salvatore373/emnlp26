import json
import logging
from typing import Any, Dict, Optional, List


from benchmark_data.monaco_benchmark import MonacoBenchmark
from benchmark_tests.benchmark_utils import BenchmarkTester
from tools.base_noisy_tool import BaseNoisyTool
from tools.base_web_tool_monaco import BaseWebToolMonaco
from tools.visit_webpage_monaco import VisitWebpage
from tools.web_search_monaco import WebSearch


class MonacoBenchmarkTester(BenchmarkTester):

    def __init__(self, current_timestamp: str = None, experiment_name: str = None, experiment_description: str = None,
                 custom_system_prompt: str = None, setting: str = "clean", clean_probability: float = 1.0):
        super().__init__(current_timestamp, experiment_name, experiment_description, custom_system_prompt)
        self.setting = setting
        self.clean_probability = clean_probability

        self.available_tools: List[BaseWebToolMonaco] = []
        self.failed_entry = None
        self.stats = {
            "processed_entries": 0,
            "failed_entries": 0,
            "total_attempts": 0,
            "failed_attempts": 0,
        }


    def tools_initializer(self, requested: Optional[List[str]] = None):
        websearch_tool = WebSearch()
        visit_tool = VisitWebpage()

        self.available_tools = [websearch_tool, visit_tool]
        for tool in self.available_tools:
            tool.SWITCH_WRONG_OUTPUT = False
            tool.SWITCH_DISTRACTOR_OUTPUT = False
            if self.setting == "return_general_error_message":
                tool.SWITCH_RETURN_GENERAL_ERROR_MESSAGE = True
                tool.PROBABILITY_OF_STD_OUTPUT = 1.0 - self.clean_probability
        return self.available_tools

    def on_entry_start_processing(self, entry_ind, entry, agent):
        # Tell the tools which is the reference MoNaCo entry to set the right distractors
        for tool in self.available_tools:
            tool.set_current_monaco_entry(entry.original_entry)

        self.failed_entry = False

    def on_attempt_start(self, entry_ind, entry, attempt_number):
        self.stats["total_attempts"] += 1

    def on_attempt_end(self, entry_ind, entry, attempt_number, response) -> Dict[str, Any]:
        # Mark this entry as processed
        self.failed_entry = False

        # Reset the list of interactions of the WebSearch tool (to have only those of the current attemp)
        web_search_interactions = self.available_tools[0].interactions
        self.available_tools[0].interactions = []

        # Reset the noisy tools calls markers
        import threading
        if not hasattr(BaseNoisyTool, '_thread_local'):
            BaseNoisyTool._thread_local = threading.local()
        BaseNoisyTool._thread_local.global_called = False
        for t in self.available_tools:
            t._noisy_forward_called = False

        return {"web_search_interactions": web_search_interactions}

    def on_attempt_fail(self, entry_ind, entry, attempt_number, exception):
        self.stats[f"failed_attempts"] += 1

    def on_entry_end_processing(self, entry_ind, entry, agent):
        # Update statistics for this entry
        self.stats["processed_entries"] += 1
        self.stats["failed_entries"] += int(self.failed_entry)
        # Print the current statistics
        logging.info(f"📊 Current stats {json.dumps(self.stats)}")


def test_monaco_benchmark2(current_timestamp, model_id,
                           max_steps: int = None, use_custom_system_prompt: bool = False, max_entries: int = None,
                           enable_thinking: Optional[bool] = None,
                           setting: str = "clean", clean_probability: float = 1.0):
    given_args = locals()

    experiment_name = "monaco_benchmark"
    experiment_description = (
        "Test the model on the restructured MoNaCo benchmark.\n"
        f"The following arguments are provided:\n{json.dumps(given_args, indent=2)}\n"
    )

    custom_system_prompt = None
    if use_custom_system_prompt:
        import yaml
        import os
        yaml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"system_prompt_code_agent.yaml",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            custom_system_prompt = yaml.safe_load(f)

    benchmark = MonacoBenchmark(
        # path_to_monaco_jsonl="/workspace/tools/monaco_preprocessing/data/restructured_monaco_newans_gptoss.jsonl"
    )

    return MonacoBenchmarkTester(
        current_timestamp=current_timestamp,
        experiment_name=experiment_name,
        experiment_description=experiment_description,
        custom_system_prompt=custom_system_prompt,
        setting=setting,
        clean_probability=clean_probability,
    ).test_benchmark(
        model_id=model_id,
        agent_run_additional_fields=None,
        benchmark=benchmark,
        max_steps=max_steps,
        max_entries=max_entries,
        enable_thinking=enable_thinking,
    )
