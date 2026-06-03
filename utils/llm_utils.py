import os
from typing import Literal, Optional

_LLM_PORT = "8000"
_LLM_BASE_URL = f"http://localhost:{_LLM_PORT}/v1"

def load_llm(model_id: str, scope: Literal['agent', 'judge'] = 'agent', enable_thinking: Optional[bool] = None):
    """
    Returns an OpenAI client configured to interact with a running LLM server.
    """
    import openai
    api_base = _LLM_BASE_URL
    model_provider = os.environ.get('CLI_MODEL_PROVIDER', 'vllm')
    client = openai.OpenAI(
        base_url=api_base,
        api_key="token-abc123"  # vLLM/llama.cpp doesn't validate this but requires it
    )
    os.environ['AGENT_IN_NOISE_MODEL_PROVIDER'] = model_provider
    return client
