import dataclasses
import json
import logging
import os
from enum import Enum
from smolagents import OpenAIModel
from typing import Optional


def setup_agent_llm(model_id: str, logging=None, enable_thinking: Optional[bool] = None):
    """
    Returns an instance of the `model_id` LLM to as agent.
    """
    api_base = "http://localhost:8000/v1"
    model_provider = os.environ.get('CLI_MODEL_PROVIDER', 'vllm')

    # Return an instance of the LLM that can be used by smolagents
    client_args = {
        "model_id": model_id,
        "api_base": api_base,
        "api_key": "token-abc123",  # vLLM doesn't validate this, but the param is required
        "timeout": 300,
        "max_tokens": 2048,
    }

    if "mistral" not in model_id.lower():
        client_args["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking if enable_thinking is not None else True}}
        if "qwen3.6" in model_id.lower():
            client_args["extra_body"]["chat_template_kwargs"]["preserve_thinking"] = True

    if "gpt-oss" in model_id.lower():
        client_args['reasoning_effort'] = 'high' if (
            enable_thinking if enable_thinking is not None else True) else "low"

    model_client = OpenAIModel(
        **client_args
    )
    os.environ['AGENT_IN_NOISE_MODEL_PROVIDER'] = model_provider
    return model_client, {}


def agent_memory_serializer(obj):
    # 1. Handle Primitives (The base case)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # 2. Handle Enums
    if isinstance(obj, Enum):
        return obj.value

    # 3. Handle Dataclasses (Manual iteration to avoid asdict() crashes)
    if dataclasses.is_dataclass(obj):
        result = {}
        for field in dataclasses.fields(obj):
            # NEW: Skip the "tokenizer" key specifically
            if field.name == "tokenizer":
                continue

            val = getattr(obj, field.name)
            result[field.name] = agent_memory_serializer(val)
        return result

    # 4. Handle standard Dictionaries
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # NEW: Skip the "tokenizer" key specifically
            if str(k) == "tokenizer":
                continue
            result[str(k)] = agent_memory_serializer(v)
        return result

    # 5. Handle Lists/Tuples
    if isinstance(obj, (list, tuple)):
        return [agent_memory_serializer(i) for i in obj]

    # 6. The Safety Net (Prevents AgentError.__init__ crashes)
    # If it's a complex object we don't recognize, just stringify it.
    return str(obj)


def save_conversation_as_json(agent, entry, output_json_path=None, additional_fields=None):
    memory_obj = agent.memory.steps
    json_to_save = {"memory": memory_obj, "entry": json.dumps(entry.to_json())}
    if additional_fields is not None:
        json_to_save.update(additional_fields)

    if output_json_path is None:
        return json.dumps(json_to_save, default=agent_memory_serializer)
    else:
        with open(output_json_path, "w") as f:
            json.dump(json_to_save, f, default=agent_memory_serializer, indent=4)


def convert_jsonl_to_parquet(dest_jsonl, delete_jsonl=True, process_all=False):
    if os.path.exists(dest_jsonl) and os.path.getsize(dest_jsonl) > 0:
        import polars as pl
        import json
        logging.info("📦 Converting collected data to Parquet...")
        try:
            dest_parquet = dest_jsonl.replace(".jsonl", ".parquet")

            if process_all:
                data = []
                all_keys = set()
                with open(dest_jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            # Ensure all top-level values are strictly strings natively
                            for k, v in list(obj.items()):
                                if v is None:
                                    obj[k] = "None"
                                elif k in ("memory", "entry") or isinstance(v, (dict, list)):
                                    if not isinstance(v, str):
                                        obj[k] = json.dumps(v)
                                elif not isinstance(v, str):
                                    obj[k] = str(v)
                            data.append(obj)
                            all_keys.update(obj.keys())
                        except json.JSONDecodeError as e:
                            logging.warning(f"Skipping malformed JSON line: {e}")
                            continue

                if not data:
                    logging.info("⚠️ Warning: No valid JSON data found in file.")
                    return

                # Schema reconciliation: missing values -> "None"
                for obj in data:
                    for k in all_keys:
                        if k not in obj:
                            obj[k] = "None"

                df = pl.DataFrame(data)
                df.write_parquet(dest_parquet)

            else:
                # Efficient O(1) last-line read via backward seeking
                last_line = ""
                with open(dest_jsonl, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    filesize = f.tell()
                    if filesize > 0:
                        chunk_size = 1048576  # 1MB chunk
                        position = filesize
                        first_chunk = True
                        found = False

                        while position > 0 and not found:
                            read_size = min(chunk_size, position)
                            position -= read_size
                            f.seek(position)
                            chunk = f.read(read_size)

                            if first_chunk:
                                if chunk and chunk[-1] == ord('\n'):
                                    chunk = chunk[:-1]
                                first_chunk = False

                            newline_idx = chunk.rfind(b'\n')
                            if newline_idx != -1:
                                f.seek(position + newline_idx + 1)
                                last_line = f.read().decode('utf-8')
                                found = True

                        if not found:
                            # File is a single line without trailing newlines found
                            f.seek(0)
                            last_line = f.read().decode('utf-8')

                if not last_line.strip():
                    logging.info("⚠️ Warning: No valid JSON data found in file.")
                    return

                try:
                    obj = json.loads(last_line)
                    # Ensure all top-level values are strictly strings natively
                    for k, v in list(obj.items()):
                        if v is None:
                            obj[k] = "None"
                        elif k in ("memory", "entry") or isinstance(v, (dict, list)):
                            if not isinstance(v, str):
                                obj[k] = json.dumps(v)
                        elif not isinstance(v, str):
                            obj[k] = str(v)
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping malformed JSON line: {e}")
                    return

                new_df = pl.DataFrame([obj])

                if os.path.exists(dest_parquet):
                    existing_df = pl.read_parquet(dest_parquet)

                    # existing_cols = set(existing_df.columns)
                    # new_cols = set(new_df.columns)
                    #
                    # # Schema reconciliation: union of all columns, missing values -> "None"
                    # # Add "None" for columns the new row lacks
                    # for col in existing_cols - new_cols:
                    #     new_df = new_df.with_columns(pl.lit("None").alias(col))
                    #
                    # # Add "None" for columns the existing parquet lacks
                    # for col in new_cols - existing_cols:
                    #     existing_df = existing_df.with_columns(pl.lit("None").alias(col))
                    #
                    # # Align types to String if they clash, ensuring literal "None" works across
                    # for col in existing_df.columns:
                    #     if existing_df.schema[col] != new_df.schema[col]:
                    #         existing_df = existing_df.with_columns(pl.col(col).cast(pl.String))
                    #         new_df = new_df.with_columns(pl.col(col).cast(pl.String))
                    #
                    # # Concat the exact matched schema
                    # # Align new_df's columns to match existing_df's order
                    # new_df = new_df.select(existing_df.columns)
                    # df = pl.concat([existing_df, new_df], how="vertical")

                    df = pl.concat([existing_df, new_df], how="diagonal").fill_null('None')
                else:
                    df = new_df

                df.write_parquet(dest_parquet)

            # Validation: Check if the file exists and has data
            if os.path.exists(dest_parquet):
                # We use scan().collect() to verify we can actually read the data back
                # Checking the row count ensures it's not an empty file
                parquet_count = pl.scan_parquet(dest_parquet).select(pl.len()).collect().item()

                if parquet_count > 0:
                    if delete_jsonl:
                        logging.info(f"✅ Successfully converted {parquet_count} rows. Deleting temp file.")
                        os.remove(dest_jsonl)
                    else:
                        logging.info(f"✅ Successfully converted {parquet_count} rows. Keeping JSONL as requested.")
                else:
                    logging.info("⚠️ Warning: Parquet file is empty. Keeping JSONL for safety.")

        except Exception as e:
            logging.exception(f"❌ Conversion failed: {e}")
            logging.info("💾 JSONL file preserved for manual recovery.")
