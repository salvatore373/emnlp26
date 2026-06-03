import json
import logging
import os
from typing import Any, Dict, Optional

from benchmark_data.toolqa_benchmark import ToolQABenchmark
from benchmark_tests.benchmark_utils import BenchmarkTester
from tools.base_noisy_tool import BaseNoisyTool

# Import ToolQA tool classes
from tools.agenda_tools import AgendaTools
from tools.database_query_tools import DatabaseQueryTools
from tools.dblp_tools import DblpTools

# Collect all ToolQA tools
ALL_TOOLS_CLASSES = [
    AgendaTools.GetAgendaEventsTool,
    AgendaTools.QueryAgendaEventsDatabaseTool,
    DatabaseQueryTools.QueryCoffeepricehistoryDatabaseTool,
    DatabaseQueryTools.QueryFlightsDatabaseTool,
    DatabaseQueryTools.QueryYelpDatabaseTool,
    DblpTools.GetDblpAuthorNodeTool,
    DblpTools.GetDblpAuthorNeighborsTool,
    DblpTools.GetDblpAuthorEdgesTool,
    DblpTools.GetDblpPaperNodeTool,
    DblpTools.GetDblpPaperNeighborsTool,
    DblpTools.GetDblpPaperEdgesTool,
]

class ToolQABenchmarkTester(BenchmarkTester):
    def __init__(self, current_timestamp=None, experiment_name=None, experiment_description=None,
                 custom_system_prompt=None, setting: str = "clean", clean_probability: float = 1.0):
        super().__init__(current_timestamp, experiment_name, experiment_description, custom_system_prompt)
        self.setting = setting
        self.clean_probability = clean_probability
        self.available_tools = None
        self.current_agent = None
        self.tools_to_test = None
        self.stats = {
            "processed_entries": 0,
            "failed_entries": 0,
            "total_attempts": 0,
            "failed_attempts": 0,
        }


    def tools_initializer(self, requested=None, **kwargs):
        if not self.tools_to_test:
            self.tools_to_test = list(ALL_TOOLS_CLASSES)
        self.available_tools = [t() for t in self.tools_to_test]
        for tool in self.available_tools:
            tool.SWITCH_WRONG_OUTPUT = False
            tool.SWITCH_DISTRACTOR_OUTPUT = False
            if self.setting == "return_general_error_message":
                tool.SWITCH_RETURN_GENERAL_ERROR_MESSAGE = True
                tool.PROBABILITY_OF_STD_OUTPUT = 1.0 - self.clean_probability
        return self.available_tools

    def on_entry_start_processing(self, entry_ind, entry, agent):
        self.current_agent = agent

    def on_attempt_start(self, entry_ind, entry, attempt_number):
        self.stats["total_attempts"] += 1

    def on_attempt_end(self, entry_ind, entry, attempt_number, response) -> Dict[str, Any]:
        try:
            attempted_tools = _get_attempted_tools(self.tools_to_test, self.current_agent.memory.steps)
            logging.info(f"🤖 [ATTEMPT {attempt_number}/1] Attempted Tools: {attempted_tools}")
        except Exception as e:
            attempted_tools = []
            logging.info(f"⚠️ [ATTEMPT {attempt_number}/1]️ Could not determine attempted tools: {e}")

        exact_match = str(response).strip() == str(entry.answer).strip()

        # Reset the noisy tools calls markers
        import threading
        if not hasattr(BaseNoisyTool, '_thread_local'):
            BaseNoisyTool._thread_local = threading.local()
        BaseNoisyTool._thread_local.global_called = False
        for t in self.available_tools:
            t._noisy_forward_called = False

        return {"id": entry.task_id, "attempted_tools": attempted_tools, "agent_answer": response,
                "exact_match": exact_match}

    def on_attempt_fail(self, entry_ind, entry, attempt_number, exception):
        self.stats[f"failed_attempts"] += 1

    def on_entry_end_processing(self, entry_ind, entry, agent):
        self.stats["processed_entries"] += 1
        logging.info(f"📊 Current stats {json.dumps(self.stats)}")

def _get_attempted_tools(tools_to_test, agent_memory_steps):
    tool_names = [t.name for t in tools_to_test]
    return list(set(t for s in agent_memory_steps if hasattr(s, "code_action") and s.code_action for t in tool_names if
                    t in s.code_action))

def test_toolqa2(current_timestamp, model_id, tools_to_test: list[BaseNoisyTool] = None,
                 max_steps: int = None, use_custom_system_prompt: bool = False, max_entries: int = None,
                 enable_thinking: Optional[bool] = None,
                 setting: str = "clean", clean_probability: float = 1.0):
    """A wrapper equivalent to `test_tools_in_gaia` that delegates to `benchmark_utils.test_benchmark()`.
    
    This function processes the ToolQA benchmark entries and sets up the appropriate tools.
    """
    given_args = locals()

    experiment_name = "toolqa_tool_testing"
    experiment_description = (
        "Test the agent on the ToolQA benchmark.\n"
        f"The following arguments are provided:\n{json.dumps(given_args, indent=2)}\n"
    )

    custom_system_prompt = None
    if use_custom_system_prompt:
        import yaml
        yaml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"system_prompt_code_agent.yaml",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            custom_system_prompt = yaml.safe_load(f)

    tester = ToolQABenchmarkTester(current_timestamp=current_timestamp,
                                   experiment_name=experiment_name,
                                   experiment_description=experiment_description,
                                   custom_system_prompt=custom_system_prompt,
                                   setting=setting,
                                   clean_probability=clean_probability)

    if tools_to_test:
        tester.tools_to_test = tools_to_test

    benchmark = ToolQABenchmark()

    return tester.test_benchmark(
        benchmark=benchmark,
        model_id=model_id,
        agent_run_additional_fields=None,
        max_steps=max_steps,
        max_entries=max_entries,
        enable_thinking=enable_thinking
    )
