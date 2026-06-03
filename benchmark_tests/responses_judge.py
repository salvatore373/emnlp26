"""
In this script I want to implement a LLM as a judge setting. The argments to this script can be passed also from the command line.
It takes in input:
 - a path to a parquet file
 - the model to use
 - the output path for the result JSON
 - the output path for the result parquet

The parquet file stores the result of the execution of the agent over a benchmark dataset. It has the following columns: memory, entry, id, agent_answer, exact_match, error_msg, result.
IDs have the format '{entry_id}_{num_attemp}': each entry is processed with a pass@k approach, so 'num_attemp' is in range [1, k].
The entry column contains a stringified JSON from which you have to extract the 'question' and 'valid_answers' field: this will later be called 'the question' and 'the ground truth answer'. Whereas, the 'agent_answer' field will be called 'the agent answer'.
Take into account that 'agent_answer' can be None. In this case, increment the 'failed_attempt' counter and skip this entry. If all the attemps for a given entry are failed, increment the 'failed_entries' counter.
In all the other cases, you have to use the LLM as a judge to evaluate the 'agent_answer' against the 'ground truth answer'.

The should be loaded leveraging the utils.llm_utils.load_llm function. The system prompt to be used is stored in the 'system_prompt_judge.yaml' file. The user prompt is stored in the 'user_prompt_judge.yaml' file. It has to be formatted properly to include the entry's question, agent answer and ground truth answer.
Write optimal system and user prompts, asking the LLM to evaluate whether the agent answer is correct, partially correct, or wrong. The LLM judge answer should be in structured format to avoid parsing errors. Based on the given answer, increment the right counter of correct, partially correct, or wrong attempts.
If at least one of the attempts for a given entry is correct, increment the 'correct_entries' counter. If none of the attempts for a given entry are correct but at least one is partially correct, increment the 'partially_correct_entries' counter. If none of the attempts for a given entry are correct or partially correct, increment the 'wrong_entries' counter.

Finally, save a new parquet file from the given one, where you add a new column: judge_answer, which can be 'correct', 'partially_correct' or 'wrong'.
Moreover, save a JSON file with the statistics.

You have to be as concise and as efficient as possible.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Literal, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

import yaml
from pydantic import BaseModel


class ClassificationAssessment(BaseModel):
    assessment: Literal["correct", "partially_correct", "wrong"]


class ScoreAssessment(BaseModel):
    score: int


import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import uuid

# Ensure we can import from `utils` if run from anywhere inside the project
# adding the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.llm_utils import load_llm


# Shared parquet compression setting so all writers use the same format
PARQUET_COMPRESSION = "zstd"


def judge_experiment(model, variant, parquet_file, output_json, output_parquet, path_to_user_prompt, judge_col_name=None, update_stats=True):
    # Support comma-separated variant lists (e.g., "classification,score")
    if isinstance(variant, str) and "," in variant:
        subvariants = [v.strip() for v in variant.split(",") if v.strip()]
        current_parquet = parquet_file
        for subv in subvariants:
            logging.info(f"🔁 Multi-variant detected, running sub-variant '{subv}'")
            # Run each sub-variant sequentially, feeding the previous output parquet
            judge_experiment(model, subv, current_parquet, output_json, output_parquet, path_to_user_prompt, judge_col_name, update_stats)
            current_parquet = output_parquet
        return

    # Load LLM
    logging.info(f"🧠 Loading LLM {model}")
    client = load_llm(model, scope='judge')

    # Load Prompts
    with open(os.path.join(script_dir, "system_prompt_judge.yaml"), "r") as f:
        sys_prompt_data = yaml.safe_load(f)
        system_prompt = sys_prompt_data.get(f"instruction_{variant}",
                                            sys_prompt_data.get("instruction_classification", ""))

    with open(path_to_user_prompt, "r") as f:
        user_prompt_data = yaml.safe_load(f)
        user_prompt_tmpl = user_prompt_data.get("template", "")
        few_shots = user_prompt_data.get("few_shots", [])

    # Stats
    if variant == "classification":
        stats = {
            "correct_attempts": 0,
            "partially_correct_attempts": 0,
            "wrong_attempts": 0,
            "failed_attempts": 0,
            "correct_entries": 0,
            "partially_correct_entries": 0,
            "wrong_entries": 0,
            "failed_entries": 0
        }
    else:
        stats = {
            "score_5_attempts": 0,
            "score_4_attempts": 0,
            "score_3_attempts": 0,
            "score_2_attempts": 0,
            "score_1_attempts": 0,
            "failed_attempts": 0,
            "score_5_entries": 0,
            "score_4_entries": 0,
            "score_3_entries": 0,
            "score_2_entries": 0,
            "score_1_entries": 0,
            "failed_entries": 0
        }

    # Batch Processing setup: try pyarrow.dataset scanner, fall back to ParquetFile.iter_batches
    logging.info(f"📦 Streaming parquet records out-of-core from {parquet_file}...")

    clean_model_id = model.split("/")[-1]
    if judge_col_name is None:
        judge_col_name = f"judge_answer_{clean_model_id}_{variant}"

    # Prepare iterator and original schema using DuckDB when possible
    batch_iter = None
    try:
        import duckdb
        con = duckdb.connect()
        reader = con.query(f"SELECT * FROM '{parquet_file}'").to_arrow_reader(batch_size=20)
        original_schema = reader.schema
        batch_iter = reader
        logging.info("Using DuckDB for parquet streaming")
        
        # just to try
        reader = con.query(f"SELECT * FROM '{parquet_file}'").to_arrow_reader(batch_size=20)
        next(reader)
        # reset the iter
        batch_iter = con.query(f"SELECT * FROM '{parquet_file}'").to_arrow_reader(batch_size=20)
    except Exception as e:
        logging.warning(f"DuckDB failed: {e}. Falling back to ParquetFile.iter_batches")
        pf = pq.ParquetFile(parquet_file)
        original_schema = pf.schema_arrow
        batch_iter = pf.iter_batches(batch_size=20)

    judge_answer_type = pa.string()

    # Avoid duplicating the judge column if it's already present in the input
    existing_field_names = [f.name for f in original_schema]
    if judge_col_name in existing_field_names:
        new_schema = original_schema
    else:
        new_schema = original_schema.append(pa.field(judge_col_name, judge_answer_type, nullable=True))

    # Determine safe output path to avoid truncating the input file when input==output
    abs_in = os.path.abspath(parquet_file)
    abs_out = os.path.abspath(output_parquet)
    if abs_in == abs_out:
        tmp_suffix = f".tmp.{uuid.uuid4().hex}"
        output_parquet_tmp = output_parquet + tmp_suffix
        will_replace_original = True
        logging.info("🔁 Input and output parquet are the same — writing to temporary file first")
    else:
        output_parquet_tmp = output_parquet
        will_replace_original = False

    os.makedirs(os.path.dirname(output_parquet_tmp) if os.path.dirname(output_parquet_tmp) else ".", exist_ok=True)

    # Tracking struct for pass@K calculation
    entry_judgements = defaultdict(list)

    # Peek the batch iterator to detect empty inputs (and to avoid silent no-op)
    import itertools
    batch_iter = iter(batch_iter)
    try:
        first_batch = next(batch_iter)
    except StopIteration:
        first_batch = None

    if first_batch is None:
        logging.warning(f"⚠️  No batches found in parquet '{parquet_file}'. The file may be empty or unreadable.")
        # Still create an empty output parquet with same schema
        with pq.ParquetWriter(output_parquet_tmp, new_schema, compression=PARQUET_COMPRESSION) as writer:
            pass
        if will_replace_original:
            os.replace(output_parquet_tmp, output_parquet)
            logging.info(f"🔁 Replaced original parquet with temporary: {output_parquet}")
        return

    # Re-chain the first batch back to the iterator
    batch_iter = itertools.chain([first_batch], batch_iter)

    def process_row(row):
        entry_str = row.get("entry")
        agent_answer = row.get("agent_answer")
        row_id = row.get("id")

        split_val = str(row_id).split("_")
        base_entry_id = "_".join(split_val[:-1]) if len(split_val) > 1 else str(row_id)

        if row.get('memory', 'None') == 'None':
            return "None", base_entry_id, "failed_attempt"

        try:
            entry_data = json.loads(entry_str)
            question = entry_data.get("question", "")
            valid_answers = entry_data.get("valid_answers", entry_data.get("answers", entry_data.get("answer", [])))
        except (json.JSONDecodeError, TypeError):
            question = ""
            valid_answers = []

        messages = [{"role": "system", "content": system_prompt}]

        for shot in few_shots:
            shot_user = user_prompt_tmpl.format(
                question=shot.get("question", "").strip(),
                ground_truth_answer=shot.get("gt_answer", "").strip(),
                agent_answer=shot.get("agent_answer", "").strip()
            )
            messages.append({"role": "user", "content": shot_user})

            if variant == "classification":
                shot_ans = json.dumps({"assessment": shot.get("classification", "wrong")})
            else:
                shot_ans = json.dumps({"score": shot.get("score", 1)})

            messages.append({"role": "assistant", "content": shot_ans})

        user_prompt = user_prompt_tmpl.format(
            question=question,
            ground_truth_answer=json.dumps(valid_answers, indent=2),
            agent_answer=str(agent_answer)
        )
        messages.append({"role": "user", "content": user_prompt})

        logging.info(f"🔎 Evaluating entry {base_entry_id} with:"
                     f"\n\t- Question: {question}"
                     f"\n\t- GT answer: {json.dumps(valid_answers, indent=2)}"
                     f"\n\t- Agent answer: {agent_answer}")

        try:
            fmt = ClassificationAssessment if variant == "classification" else ScoreAssessment
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format=fmt
            )
            ans_obj = response.choices[0].message.parsed

            if variant == "classification":
                ans = ans_obj.assessment
            else:
                ans = ans_obj.score
                if ans < 1 or ans > 5:
                    ans = 1
                ans = str(ans)

            logging.info(
                f"⚖️ Judged " + (f"as '{ans}'." if variant == "classification" else f"with score {ans}."))
            return ans, base_entry_id, ans
        except Exception as e:
            logging.exception(f"❌ Error calling LLM for prompt: {e}")
            return "None", base_entry_id, "failed_attempt"

    with pq.ParquetWriter(output_parquet_tmp, new_schema, compression=PARQUET_COMPRESSION) as writer:
        for batch_index, batch in enumerate(batch_iter):
            df_chunk = pl.from_arrow(batch)
            judge_answers = []

            with ThreadPoolExecutor(max_workers=20) as executor:
                results = list(executor.map(process_row, df_chunk.iter_rows(named=True)))

            for ans, base_entry_id, stat_increment in results:
                judge_answers.append(ans)
                if ans != "None":
                    entry_judgements[base_entry_id].append(ans)

                if stat_increment == "failed_attempt":
                    stats["failed_attempts"] += 1
                elif stat_increment != "None" and stat_increment is not None:
                    if variant == "classification":
                        if stat_increment == "correct":
                            stats["correct_attempts"] += 1
                        elif stat_increment == "partially_correct":
                            stats["partially_correct_attempts"] += 1
                        elif stat_increment == "wrong":
                            stats["wrong_attempts"] += 1
                    else:
                        stats[f"score_{stat_increment}_attempts"] += 1

            # Convert python structure into proper pa.array corresponding back into chunk logic
            ans_array = pa.array(judge_answers, type=judge_answer_type)
            new_batch = batch.append_column(judge_col_name, ans_array)
            writer.write_batch(new_batch)
            logging.info(f"🧩 Processed batch chunk {batch_index + 1}")

    # Calculate Entry-level evaluations completely natively via lightweight default dict
    for entry_id, judgements in entry_judgements.items():
        valid_judgements = [j for j in judgements if j is not None]

        if not valid_judgements:
            stats["failed_entries"] += 1
            continue

        if variant == "classification":
            if any(j == "correct" for j in valid_judgements):
                stats["correct_entries"] += 1
            elif any(j == "partially_correct" for j in valid_judgements):
                stats["partially_correct_entries"] += 1
            else:
                stats["wrong_entries"] += 1
        else:
            best_score = max(valid_judgements)
            stats[f"score_{best_score}_entries"] += 1

    if update_stats:
        # Save statistics directly
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)

        existing_stats = {}
        if os.path.exists(output_json):
            try:
                with open(output_json, "r") as f:
                    existing_stats = json.load(f)
                    if not isinstance(existing_stats, dict):
                        existing_stats = {}
            except json.JSONDecodeError:
                pass
        # Ensure model key exists before assigning variant stats
        existing_stats.setdefault(model, {})[variant] = stats

        with open(output_json, "w") as f:
            json.dump(existing_stats, f, indent=4)

        logging.info(f"💾 Saved completed stats metadata to {output_json}")


def compute_simple_stats(parquet_file, output_json, output_parquet):
    """
    Compute lightweight statistics for a parquet of agent attempts and augment the
    output JSON with summary metrics.

    Stats keys produced (concise definitions):
    - average_num_steps_per_attempt: mean of the final step number per attempt
        (attempts without step info are excluded).
    - average_num_steps_per_entry: mean across entries of each entry's average
        final step over its attempts.
    - average_first_syntax_error_step_per_attempt: mean step index of the first
        syntax error, computed only over attempts where a syntax error occurred.
    - average_first_syntax_error_step_per_entry: mean across entries of the
        earliest syntax-error step among that entry's attempts (entries without
        syntax errors are excluded).
    - average_first_error_step_per_attempt: mean step index of the first error
        of any type, computed only over attempts where any error occurred.
    - average_first_error_step_per_entry: mean across entries of the earliest
        first-error step among that entry's attempts (entries without errors are
        excluded).
    - average_total_num_syntax_errors_per_attempt: mean count of syntax errors
        per attempt (attempts without syntax errors contribute zero to this
        average calculation).
    - average_total_num_syntax_errors_per_entry: mean across entries of each
        entry's average number of syntax errors over its attempts.
    - average_total_num_errors_per_attempt: mean count of generic (any-type)
        errors per attempt (includes syntax and other error messages).
    - average_total_num_errors_per_entry: mean across entries of each entry's
        average number of generic errors across its attempts.
    - num_terminations_due_to_max_steps_per_attempt: total attempts that
        terminated with the "Reached max steps." condition.
    - num_terminations_due_to_max_steps_per_entry: fraction of entries that had
        at least one attempt terminated due to max steps (value in [0,1]).
    - num_attempts_with_error: total number of attempts that reported any error.
    - num_entries_with_error: total number of entries with at least one attempt
        that reported an error.
    - num_attempts_with_1_error / num_attempts_with_2_or_more_errors:
        counts of attempts with exactly 1 or with 2+ errors respectively.
    - num_entries_with_1_error / num_entries_with_2_or_more_errors:
        counts of entries whose attempts sum to exactly 1 or to 2+ errors.

    All averages are floats; counts are integers. The function also produces
    per-model and per-subset aggregates written into the output JSON.
    """
    logging.info(f"📊 Computing simple stats for {parquet_file}...")

    existing_stats = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, "r") as f:
                existing_stats = json.load(f)
                if not isinstance(existing_stats, dict):
                    existing_stats = {}
        except json.JSONDecodeError:
            pass

    # Batch Processing setup: try DuckDB, fall back to ParquetFile.iter_batches
    try:
        import duckdb
        con = duckdb.connect()
        reader = con.query(f"SELECT * FROM '{parquet_file}'").to_arrow_reader(batch_size=20)
        original_schema = reader.schema
        batch_iter = reader
        logging.info("Using DuckDB for parquet streaming")

        # just to try
        reader = con.query(f"SELECT * FROM '{parquet_file}'").to_arrow_reader(batch_size=20)
        next(reader)
        # reset the iter
        batch_iter = con.query(f"SELECT * FROM '{parquet_file}'").to_arrow_reader(batch_size=20)
    except Exception as e:
        logging.warning(f"DuckDB failed: {e}. Falling back to ParquetFile.iter_batches")
        pf = pq.ParquetFile(parquet_file)
        original_schema = pf.schema_arrow
        batch_iter = pf.iter_batches(batch_size=20)

    # Define any new columns for the schema here if needed
    new_schema = original_schema.append(
        pa.field("tot_num_steps", pa.string(), nullable=True)
    ).append(
        pa.field("step_first_syntax_error", pa.string(), nullable=True)
    ).append(
        pa.field("total_num_syntax_errors", pa.string(), nullable=True)
    ).append(
        pa.field("answer_type", pa.string(), nullable=True)
    )
    # Alternative: Preserve original schema
    # new_schema = original_schema

    abs_in = os.path.abspath(parquet_file)
    abs_out = os.path.abspath(output_parquet)
    if abs_in == abs_out:
        tmp_suffix = f".tmp.{uuid.uuid4().hex}"
        output_parquet_tmp = output_parquet + tmp_suffix
        will_replace_original = True
    else:
        output_parquet_tmp = output_parquet
        will_replace_original = False

    os.makedirs(os.path.dirname(output_parquet_tmp) if os.path.dirname(output_parquet_tmp) else ".", exist_ok=True)

    # Initialize the new stats
    stats = {
        "average_num_steps_per_attempt": 0,
        "average_num_steps_per_entry": 0,  # avg over attempts in same entry, then average over all entries
        "average_first_syntax_error_step_per_attempt": 0,
        # avg over attempts in same entry, then average over all entries
        "average_first_syntax_error_step_per_entry": 0,
        # Generic error (first step where any error occurred)
        "average_first_error_step_per_attempt": 0,
        "average_first_error_step_per_entry": 0,
        "average_total_num_syntax_errors_per_attempt": 0,
        "average_total_num_syntax_errors_per_entry": 0,
        # Average number of all generic errors (syntax + others)
        "average_total_num_errors_per_attempt": 0,
        "average_total_num_errors_per_entry": 0,
        # avg over attempts in same entry, then average over all entries
        "num_terminations_due_to_max_steps_per_attempt": 0,
        "num_terminations_due_to_max_steps_per_entry": 0,
        "num_attempts_with_error": 0,  # including syntax, max steps, ...
        "num_entries_with_error": 0,  # including syntax, max steps, ...
        "num_attempts_with_1_error": 0,  # count of attempts where exactly 1 error occurred
        "num_attempts_with_2_or_more_errors": 0,  # count of attempts where 2 or more errors occurred
        "num_entries_with_1_error": 0,  # count of entries where exactly 1 error occurred across all its attempts
        "num_entries_with_2_or_more_errors": 0,
        # count of entries where 2 or more errors occurred across all its attempts
    }

    default_classification = {
        # Original metrics
        "correct_attempts": 0, "partially_correct_attempts": 0, "wrong_attempts": 0, "failed_attempts": 0,
        "correct_entries": 0, "partially_correct_entries": 0, "wrong_entries": 0, "failed_entries": 0,

        # New: number of entries/attempts marked with a specific judgement where one or more errors occurred
        "correct_entries_with_error": 0, "partially_correct_entries_with_error": 0, "wrong_entries_with_error": 0,
        "correct_attempts_with_error": 0, "partially_correct_attempts_with_error": 0, "wrong_attempts_with_error": 0,

        # New: intermediate variables to compute the average number of tokens for a given judgement
        "correct_attempts_tot_tokens": 0, "partially_correct_attempts_tot_tokens": 0, "wrong_attempts_tot_tokens": 0,
        "correct_attempts_with_tokens_count": 0, "partially_correct_attempts_with_tokens_count": 0,
        "wrong_attempts_with_tokens_count": 0,

        "correct_entries_tot_tokens": 0, "partially_correct_entries_tot_tokens": 0, "wrong_entries_tot_tokens": 0,
        "correct_entries_count_for_tokens": 0, "partially_correct_entries_count_for_tokens": 0,
        "wrong_entries_count_for_tokens": 0,
    }
    default_score = {
        # Original metrics
        "score_5_attempts": 0, "score_4_attempts": 0, "score_3_attempts": 0, "score_2_attempts": 0,
        "score_1_attempts": 0, "failed_attempts": 0,
        "score_5_entries": 0, "score_4_entries": 0, "score_3_entries": 0, "score_2_entries": 0, "score_1_entries": 0,
        "failed_entries": 0,

        # New: number of entries/attempts marked with a specific judgement where one or more errors occurred
        "score_5_entries_with_error": 0, "score_4_entries_with_error": 0, "score_3_entries_with_error": 0,
        "score_2_entries_with_error": 0, "score_1_entries_with_error": 0,
        "score_5_attempts_with_error": 0, "score_4_attempts_with_error": 0, "score_3_attempts_with_error": 0,
        "score_2_attempts_with_error": 0, "score_1_attempts_with_error": 0,

        # New: intermediate variables to compute the average number of tokens for a given judgement
        "score_5_attempts_tot_tokens": 0, "score_4_attempts_tot_tokens": 0, "score_3_attempts_tot_tokens": 0,
        "score_2_attempts_tot_tokens": 0, "score_1_attempts_tot_tokens": 0,
        "score_5_attempts_with_tokens_count": 0, "score_4_attempts_with_tokens_count": 0,
        "score_3_attempts_with_tokens_count": 0, "score_2_attempts_with_tokens_count": 0,
        "score_1_attempts_with_tokens_count": 0,

        "score_5_entries_tot_tokens": 0, "score_4_entries_tot_tokens": 0, "score_3_entries_tot_tokens": 0,
        "score_2_entries_tot_tokens": 0, "score_1_entries_tot_tokens": 0,
        "score_5_entries_count_for_tokens": 0, "score_4_entries_count_for_tokens": 0,
        "score_3_entries_count_for_tokens": 0, "score_2_entries_count_for_tokens": 0,
        "score_1_entries_count_for_tokens": 0,
    }

    # Initialize the statistics per judge model and judgement type
    judge_answers_col_names = [col_name for col_name in original_schema.names if col_name.startswith("judge_answer_")]
    judge_models_and_type_stats: Dict[str, Dict[str, Dict]] = {}  # model_id -> list of available judge_types
    for col_name in judge_answers_col_names:
        model_name = col_name.replace("judge_answer_", '')
        model_name = model_name[:model_name.rfind('_')]
        judgement_type = col_name.split("_")[-1]

        if judgement_type == "classification":
            keys_template = default_classification.copy()
        else:
            keys_template = default_score.copy()

        # Dynamically add keys from existing_stats if any
        if model_name in existing_stats and judgement_type in existing_stats[model_name]:
            for k, v in existing_stats[model_name][judgement_type].items():
                if (isinstance(v, int) or isinstance(v, float)) and k not in keys_template:
                    keys_template[k] = 0

        judge_models_and_type_stats.setdefault(model_name, {})[judgement_type] = {
            "boolean": keys_template.copy(), "list": keys_template.copy(), "single": keys_template.copy(),
        }

    total_attempts = 0
    sum_num_steps_attempt = 0
    sum_first_syntax_attempt = 0
    sum_total_syntax_attempt = 0
    # Sum of total generic errors (all error messages) seen across attempts
    sum_total_errors_attempt = 0
    # Counters for attempts with syntax/any errors (used to compute averages only over occurrences)
    num_attempts_with_syntax_error = 0
    num_attempts_with_any_error = 0
    sum_first_error_attempt = 0

    entry_stats = defaultdict(
        lambda: {"num_steps": [], "first_syntax": [], "first_error": [], "total_syntax": [], "max_steps": [],
                 "answer_type": "", "has_error": []})
    entry_judgements_by_subset = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    import itertools
    batch_iter = iter(batch_iter)
    try:
        first_batch = next(batch_iter)
    except StopIteration:
        first_batch = None

    if first_batch is None:
        logging.warning(f"⚠️  No batches found in parquet '{parquet_file}'.")
        with pq.ParquetWriter(output_parquet_tmp, new_schema, compression=PARQUET_COMPRESSION) as writer:
            pass
        if will_replace_original:
            os.replace(output_parquet_tmp, output_parquet)
        return

    batch_iter = itertools.chain([first_batch], batch_iter)

    with pq.ParquetWriter(output_parquet_tmp, new_schema, compression=PARQUET_COMPRESSION) as writer:
        for batch_index, batch in enumerate(batch_iter):
            df_chunk = pl.from_arrow(batch)

            tot_num_steps_data = []
            step_first_syntax_error_data = []
            total_num_syntax_errors_data = []
            answer_type_data = []

            for row in df_chunk.iter_rows(named=True):
                # Compute statistics for this dataset entry
                first_syntax = -1
                total_syntax = 0
                last_step_local = -1
                max_steps_reached = False
                attempt_has_error = False
                first_error = -1

                memory = [] if row.get("memory") == 'None' else json.loads(row.get("memory", "[]"))
                num_errors_in_attempt = 0
                for message in memory:
                    # Find the first step when a syntax error occurred
                    error_msg = message.get("error", "") or ""
                    if error_msg.startswith("Error in code parsing") or error_msg.startswith("Code parsing failed"):
                        total_syntax += 1
                        if first_syntax == -1:
                            first_syntax = message.get("step_number", -1)
                    elif error_msg == "Reached max steps.":
                        # This attempt reached the maximum number of allowed steps
                        max_steps_reached = True
                    # Track first occurrence of any error
                    if error_msg and first_error == -1:
                        first_error = message.get("step_number", -1)

                    if error_msg:
                        # A general error occurred
                        attempt_has_error = True
                        num_errors_in_attempt += 1

                    # Save the last step number
                    last_step_local = max(last_step_local, (message.get("step_number", -1) or -1))

                # Take the total number of tokens in the conversation from the last message object
                tot_num_tokens = (memory[-1].get("token_usage", {}) or {}).get("total_tokens", -1) if memory else -1
                # Get the number of golden passages in this entry
                entry_data = json.loads(row.get('entry', '{}'))
                num_golden_passages =\
                    len(set(itertools.chain.from_iterable([gp['url'] for gp in entry_data.get("golden_passages", [])])))

                base_entry_id = entry_data.get("id")

                if last_step_local != -1:
                    total_attempts += 1
                    sum_num_steps_attempt += last_step_local

                # Syntax-error attempt aggregates
                if first_syntax != -1:
                    sum_first_syntax_attempt += first_syntax
                    num_attempts_with_syntax_error += 1

                if total_syntax != -1:
                    sum_total_syntax_attempt += total_syntax

                if max_steps_reached:
                    stats["num_terminations_due_to_max_steps_per_attempt"] += 1

                # Generic error aggregates (first error step)
                if attempt_has_error:
                    stats["num_attempts_with_error"] += 1
                    num_attempts_with_any_error += 1
                    if first_error != -1:
                        sum_first_error_attempt += first_error

                entry_stats[base_entry_id]["num_steps"].append(last_step_local if last_step_local != -1 else None)
                entry_stats[base_entry_id]["first_syntax"].append(first_syntax if first_syntax != -1 else None)
                entry_stats[base_entry_id]["first_error"].append(first_error if first_error != -1 else None)
                entry_stats[base_entry_id]["total_syntax"].append(total_syntax if total_syntax != -1 else None)
                entry_stats[base_entry_id]["max_steps"].append(max_steps_reached)
                entry_stats[base_entry_id]["has_error"].append(attempt_has_error)
                if "num_errors" not in entry_stats[base_entry_id]:
                    entry_stats[base_entry_id]["num_errors"] = []
                entry_stats[base_entry_id]["num_errors"].append(num_errors_in_attempt)
                # Accumulate total generic error count for attempt-level average
                sum_total_errors_attempt += num_errors_in_attempt
                # Record number of golden passages for this entry (same across attempts)
                entry_stats[base_entry_id]["num_golden_passages"] = num_golden_passages

                # Infer the type of answer
                try:
                    valid_answers = entry_data.get("valid_answers", entry_data.get("answers", []))
                    gt_answer = valid_answers[0] if valid_answers else None
                except (json.JSONDecodeError, TypeError, IndexError):
                    gt_answer = None

                if isinstance(gt_answer, str) and gt_answer.lower() in ['yes', 'no', 'true', 'false']:
                    answer_type = "boolean"
                elif isinstance(gt_answer, list):
                    answer_type = "list"
                else:
                    answer_type = "single"

                entry_stats[base_entry_id]["answer_type"] = answer_type

                tot_num_steps_data.append(str(last_step_local) if last_step_local != -1 else "None")
                step_first_syntax_error_data.append(str(first_syntax) if first_syntax != -1 else "None")
                total_num_syntax_errors_data.append(str(total_syntax) if total_syntax != -1 else "None")
                answer_type_data.append(answer_type)

                # Update the statistic of the subset this entry belongs to
                for col_name in row.keys():
                    if not col_name.startswith("judge_answer_"):
                        continue
                    model_name = col_name.replace("judge_answer_", '')
                    model_name = model_name[:model_name.rfind('_')]
                    judgement_type = col_name.split("_")[-1]
                    ans = row.get(col_name)

                    target_stats = judge_models_and_type_stats[model_name][judgement_type][answer_type]
                    if row.get('memory', 'None') == 'None' or ans == "None" or ans is None:
                        target_stats["failed_attempts"] += 1
                    else:
                        if judgement_type == "classification":
                            target_stats[f"{ans}_attempts"] += 1
                            if attempt_has_error:
                                target_stats[f"{ans}_attempts_with_error"] += 1
                            if tot_num_tokens > 0:
                                target_stats[f"{ans}_attempts_tot_tokens"] += tot_num_tokens
                                target_stats[f"{ans}_attempts_with_tokens_count"] += 1
                        else:
                            target_stats[f"score_{ans}_attempts"] += 1
                            if attempt_has_error:
                                target_stats[f"score_{ans}_attempts_with_error"] += 1
                            if tot_num_tokens > 0:
                                target_stats[f"score_{ans}_attempts_tot_tokens"] += tot_num_tokens
                                target_stats[f"score_{ans}_attempts_with_tokens_count"] += 1

                    entry_judgements_by_subset[model_name][judgement_type][answer_type][base_entry_id].append({
                        "ans": ans,
                        "has_error": attempt_has_error,
                        "tot_tokens": tot_num_tokens
                    })

            tot_num_steps_arr = pa.array(tot_num_steps_data, type=pa.string())
            step_first_syntax_error_arr = pa.array(step_first_syntax_error_data, type=pa.string())
            total_num_syntax_errors_arr = pa.array(total_num_syntax_errors_data, type=pa.string())
            answer_type_arr = pa.array(answer_type_data, type=pa.string())

            batch = batch.append_column("tot_num_steps", tot_num_steps_arr)
            batch = batch.append_column("step_first_syntax_error", step_first_syntax_error_arr)
            batch = batch.append_column("total_num_syntax_errors", total_num_syntax_errors_arr)
            batch = batch.append_column("answer_type", answer_type_arr)

            writer.write_batch(batch)

    total_entries = len(entry_stats)
    if total_attempts > 0:
        stats["average_num_steps_per_attempt"] = sum_num_steps_attempt / total_attempts
        stats["average_first_syntax_error_step_per_attempt"] = (
                    sum_first_syntax_attempt / num_attempts_with_syntax_error) if num_attempts_with_syntax_error > 0 else 0
        stats["average_first_error_step_per_attempt"] = (
                    sum_first_error_attempt / num_attempts_with_any_error) if num_attempts_with_any_error > 0 else 0
        stats["average_total_num_syntax_errors_per_attempt"] = sum_total_syntax_attempt / total_attempts
        # Average total number of generic errors per attempt (syntax + other errors)
        stats["average_total_num_errors_per_attempt"] = sum_total_errors_attempt / total_attempts

    if total_entries > 0:
        sum_avg_num_steps_entry = 0
        sum_avg_total_syntax_entry = 0
        sum_avg_total_errors_entry = 0
        sum_avg_max_steps_entry = 0
        sum_has_error_entry = 0

        # Entry-level accumulators for first-error/syntax-first-step averages
        sum_first_syntax_entry = 0
        num_entries_with_syntax_error = 0
        sum_first_error_entry = 0
        num_entries_with_any_error = 0

        for entry_id, estats in entry_stats.items():
            num_steps = [e for e in estats["num_steps"] if e is not None]
            first_syntax_list = estats.get("first_syntax", [])
            first_error_list = estats.get("first_error", [])
            total_syntax = estats.get("total_syntax", [])
            max_steps = estats.get("max_steps", [])
            has_error = estats.get("has_error", [])
            num_errors = estats.get("num_errors", [])

            sum_avg_num_steps_entry += sum(num_steps) / len(num_steps) if num_steps else 0

            # Entry-level first syntax error: take earliest step among attempts for this entry
            valid_syntax_vals = [f for f in first_syntax_list if f is not None and f > -1]
            if valid_syntax_vals:
                sum_first_syntax_entry += min(valid_syntax_vals)
                num_entries_with_syntax_error += 1

            # Entry-level first generic error: earliest step among attempts
            valid_error_vals = [f for f in first_error_list if f is not None and f > -1]
            if valid_error_vals:
                sum_first_error_entry += min(valid_error_vals)
                num_entries_with_any_error += 1

            sum_avg_total_syntax_entry += sum(total_syntax) / len(total_syntax) if total_syntax else 0
            # Entry-level average of generic errors (average over this entry's attempts)
            sum_avg_total_errors_entry += sum(num_errors) / len(num_errors) if num_errors else 0
            sum_avg_max_steps_entry += any(max_steps)
            sum_has_error_entry += any(has_error)

            total_errors_in_entry = sum(num_errors)
            if total_errors_in_entry == 1:
                stats["num_entries_with_1_error"] += 1
            elif total_errors_in_entry >= 2:
                stats["num_entries_with_2_or_more_errors"] += 1

        stats["average_num_steps_per_entry"] = sum_avg_num_steps_entry / total_entries
        stats["average_first_syntax_error_step_per_entry"] = (
                    sum_first_syntax_entry / num_entries_with_syntax_error) if num_entries_with_syntax_error > 0 else 0
        stats["average_first_error_step_per_entry"] = (
                    sum_first_error_entry / num_entries_with_any_error) if num_entries_with_any_error > 0 else 0
        stats["average_total_num_syntax_errors_per_entry"] = sum_avg_total_syntax_entry / total_entries
        # Average total number of generic errors per entry (averaged across its attempts)
        stats["average_total_num_errors_per_entry"] = sum_avg_total_errors_entry / total_entries
        stats["num_terminations_due_to_max_steps_per_entry"] = sum_avg_max_steps_entry / total_entries
        stats["num_entries_with_error"] = sum_has_error_entry

    # Aggregate counts per number of golden passages for each model/judgement type
    per_golden_passages_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    for model_name, type_dict in entry_judgements_by_subset.items():
        for judgement_type, ans_type_dict in type_dict.items():
            for answer_type, entries in ans_type_dict.items():
                target_stats = judge_models_and_type_stats[model_name][judgement_type][answer_type]
                for entry_id, judgements in entries.items():
                    valid_judgements = [j for j in judgements if j["ans"] is not None and j["ans"] != "None"]
                    # Determine the golden-passage bucket for this entry
                    gp_val = entry_stats.get(entry_id, {}).get("num_golden_passages", None)
                    gp_key = str(gp_val) if gp_val is not None else "None"

                    if not valid_judgements:
                        target_stats["failed_entries"] += 1
                        per_golden_passages_counts[model_name][judgement_type][gp_key]["failed_entries"] += 1
                        continue

                    entry_has_error = any(j["has_error"] for j in judgements)
                    entry_tot_tokens = sum(j["tot_tokens"] for j in valid_judgements if j["tot_tokens"] > 0)

                    if judgement_type == "classification":
                        if any(j["ans"] == "correct" for j in valid_judgements):
                            target_stats["correct_entries"] += 1
                            per_golden_passages_counts[model_name][judgement_type][gp_key]["correct_entries"] += 1
                            if entry_has_error:
                                target_stats["correct_entries_with_error"] += 1
                                per_golden_passages_counts[model_name][judgement_type][gp_key][
                                    "correct_entries_with_error"] += 1
                            if entry_tot_tokens > 0:
                                target_stats["correct_entries_tot_tokens"] += entry_tot_tokens
                                target_stats["correct_entries_count_for_tokens"] += 1
                        elif any(j["ans"] == "partially_correct" for j in valid_judgements):
                            target_stats["partially_correct_entries"] += 1
                            per_golden_passages_counts[model_name][judgement_type][gp_key][
                                "partially_correct_entries"] += 1
                            if entry_has_error:
                                target_stats["partially_correct_entries_with_error"] += 1
                                per_golden_passages_counts[model_name][judgement_type][gp_key][
                                    "partially_correct_entries_with_error"] += 1
                            if entry_tot_tokens > 0:
                                target_stats["partially_correct_entries_tot_tokens"] += entry_tot_tokens
                                target_stats["partially_correct_entries_count_for_tokens"] += 1
                        else:
                            target_stats["wrong_entries"] += 1
                            per_golden_passages_counts[model_name][judgement_type][gp_key]["wrong_entries"] += 1
                            if entry_has_error:
                                target_stats["wrong_entries_with_error"] += 1
                                per_golden_passages_counts[model_name][judgement_type][gp_key][
                                    "wrong_entries_with_error"] += 1
                            if entry_tot_tokens > 0:
                                target_stats["wrong_entries_tot_tokens"] += entry_tot_tokens
                                target_stats["wrong_entries_count_for_tokens"] += 1
                    else:
                        best_score = max(int(j["ans"]) for j in valid_judgements)
                        target_stats[f"score_{best_score}_entries"] += 1
                        per_golden_passages_counts[model_name][judgement_type][gp_key][
                            f"score_{best_score}_entries"] += 1
                        if entry_has_error:
                            target_stats[f"score_{best_score}_entries_with_error"] += 1
                            per_golden_passages_counts[model_name][judgement_type][gp_key][
                                f"score_{best_score}_entries_with_error"] += 1
                        if entry_tot_tokens > 0:
                            target_stats[f"score_{best_score}_entries_tot_tokens"] += entry_tot_tokens
                            target_stats[f"score_{best_score}_entries_count_for_tokens"] += 1
                            per_golden_passages_counts[model_name][judgement_type][gp_key][
                                f"score_{best_score}_entries_tot_tokens"] += entry_tot_tokens

    # Compute averages and clean up intermediate token count variables
    # Also accumulate token totals so we can compute aggregated avg tokens across subsets
    aggregate_token_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"tot": 0, "count": 0})))
    for model_name, type_dict in judge_models_and_type_stats.items():
        for judgement_type, ans_type_dict in type_dict.items():
            for answer_type, stats_dict in ans_type_dict.items():
                keys_to_delete = []
                for k in list(stats_dict.keys()):
                    if k.endswith("_tot_tokens"):
                        base_key = k[:-len("_tot_tokens")]
                        count_key = f"{base_key}_with_tokens_count" if "attempts" in base_key else f"{base_key}_count_for_tokens"

                        tot_tokens = stats_dict[k]
                        count = stats_dict.get(count_key, 0)

                        # Accumulate totals for aggregated-level computation later
                        aggregate_token_totals[model_name][judgement_type][base_key]["tot"] += tot_tokens
                        aggregate_token_totals[model_name][judgement_type][base_key]["count"] += count

                        # Set the average tokens stat at subset level
                        stats_dict[f"{base_key}_avg_tokens"] = tot_tokens / count if count > 0 else 0

                        keys_to_delete.append(k)
                        if count_key in stats_dict:
                            keys_to_delete.append(count_key)

                for k in set(keys_to_delete):
                    if k in stats_dict:
                        del stats_dict[k]

    # Build aggregated stats across answer_type subsets (main level) using accumulated totals
    aggregated_stats_by_model = defaultdict(dict)
    for model_name, type_dict in judge_models_and_type_stats.items():
        for judgement_type, ans_type_dict in type_dict.items():
            agg = {}
            # Sum all numeric counters across subsets, skipping subset-level avg token keys
            for answer_type, stats_dict in ans_type_dict.items():
                for k, v in stats_dict.items():
                    if isinstance(v, (int, float)):
                        if k.endswith("_avg_tokens"):
                            continue
                        agg[k] = agg.get(k, 0) + v

            # Compute aggregated avg tokens from totals collected earlier
            totals_for_type = aggregate_token_totals.get(model_name, {}).get(judgement_type, {})
            for base_key, td in totals_for_type.items():
                tot = td.get("tot", 0)
                count = td.get("count", 0)
                agg[f"{base_key}_avg_tokens"] = tot / count if count > 0 else 0

                # Save aggregated stats to attach later under the correct model key
            aggregated_stats_by_model[model_name][judgement_type] = agg

    if will_replace_original:
        os.replace(output_parquet_tmp, output_parquet)

    # Save statistics directly
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)

    existing_stats["extra_stats"] = stats

    # Attach per-model judgement stats, mapping short model names to existing_stats keys
    def _is_main_level_key(k: str) -> bool:
        # Keys that should be promoted to model_id/variant level.
        # NOTE: we intentionally exclude keys that end with "_with_error" so
        # they are saved under "extra_stats" instead of the main namespace.
        return any(k.endswith(suf) for suf in ("_attempts", "_entries", "_attempts_tot_tokens", "_entries_tot_tokens",
                                               "_attempts_with_tokens_count",
                                               "_entries_count_for_tokens")) or k.startswith("num_")

    for model_name, type_dict in judge_models_and_type_stats.items():
        model_key = next((k for k in existing_stats.keys() if k.split('/')[-1] == model_name), model_name)
        existing_stats.setdefault(model_key, {})
        for judgement_type, ans_type_dict in type_dict.items():
            # Build a fresh judgement dict to avoid carrying over stale counters
            new_jdict: Dict[str, Any] = {}
            new_jdict["per_subset_stats"] = ans_type_dict

            # attach aggregated main-level stats computed earlier if present
            agg = aggregated_stats_by_model.get(model_name, {}).get(judgement_type, None)
            extra = {}
            if agg is not None:
                for k, v in agg.items():
                    if _is_main_level_key(k):
                        new_jdict[k] = v
                    else:
                        extra[k] = v

            if extra:
                new_jdict["extra_stats"] = extra

            # Replace the judgement_type dictionary so repeated runs are idempotent
            existing_stats.setdefault(model_key, {})
            existing_stats[model_key][judgement_type] = new_jdict

    # Persist per-golden-passage aggregated counts, remapping model keys similarly
    try:
        per_gp_mapped = {}
        for m, jt_dict in per_golden_passages_counts.items():
            model_key = next((k for k in existing_stats.keys() if k.split('/')[-1] == m), m)
            per_gp_mapped[model_key] = {jt: {gp: dict(vals) for gp, vals in gp_dict.items()} for jt, gp_dict in
                                        jt_dict.items()}
        existing_stats["per_golden_passages_counts"] = per_gp_mapped
    except Exception:
        existing_stats["per_golden_passages_counts"] = {}

    # Move any existing *_with_error keys found at model/variant level into extra_stats
    for model_key, mt in list(existing_stats.items()):
        if not isinstance(mt, dict):
            continue
        for judgement_type, jdict in list(mt.items()):
            if not isinstance(jdict, dict):
                continue
            # Skip top-level helper keys
            if judgement_type in ("per_golden_passages_counts", "extra_stats"):
                continue
            to_move = [k for k in list(jdict.keys()) if
                       k.endswith("_with_error") and k not in ("extra_stats", "per_subset_stats")]
            if not to_move:
                continue
            extras = jdict.setdefault("extra_stats", {})
            for k in to_move:
                extras[k] = jdict.pop(k)

    with open(output_json, "w") as f:
        json.dump(existing_stats, f, indent=4)

    logging.info(f"💾 Saved simple stats to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM as a judge")
    parser.add_argument("--parquet_file", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--model", type=str, required=False, help="Model to use")
    parser.add_argument("--output_json", type=str, required=False, default="", help="Output path for the result JSON stats (default: parquet_file with .json)")
    parser.add_argument("--output_parquet", type=str, required=False, default="", help="Output path for the result parquet file (default: same as --parquet_file)")
    parser.add_argument("--variant", type=str, required=False, default="",
                        help="Which evaluation variant to use (e.g., 'classification', 'score', or 'classification,score').")
    parser.add_argument("--compute_simple_stats", action="store_true",
                        help="Compute statistics about the experiments that do not trigger an LLM judge")

    args = parser.parse_args()

    # Resolve default output paths if not provided
    if not args.output_parquet:
        effective_output_parquet = args.parquet_file
    else:
        effective_output_parquet = args.output_parquet

    if not args.output_json:
        if args.parquet_file.lower().endswith(".parquet"):
            effective_output_json = args.parquet_file[:-len(".parquet")] + ".json"
        else:
            effective_output_json = args.parquet_file + ".json"
    else:
        effective_output_json = args.output_json

    variants = [v.strip() for v in args.variant.split(",")] if args.variant else []
    current_parquet = args.parquet_file

    if not variants and not args.compute_simple_stats:
        logging.warning("No evaluation variants or stats specified. Nothing to do.")

    for variant in variants:
        if variant not in ["classification", "score"]:
            raise ValueError(f"Invalid variant: {variant}. Supported: 'classification', 'score'")

        logging.info(f"🚀 Starting evaluation phase for variant: {variant}")
        judge_experiment(
            model=args.model,
            variant=variant,
            parquet_file=current_parquet,
            output_json=effective_output_json,
            output_parquet=effective_output_parquet
        )
        # For sequential phases, use the output of the previous phase as input
        current_parquet = effective_output_parquet

    if args.compute_simple_stats:
        logging.info("🚀 Starting compute simple stats phase...")
        compute_simple_stats(
            parquet_file=current_parquet,
            output_json=effective_output_json,
            output_parquet=effective_output_parquet
        )
        current_parquet = effective_output_parquet
