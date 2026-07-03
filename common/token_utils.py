#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import contextvars
import hashlib
import logging
import os
import shutil
import threading
import tiktoken
from functools import lru_cache

from common.file_utils import get_project_base_directory


def _ensure_tiktoken_cache() -> str:
    cache_dir = get_project_base_directory()
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir

    bundled_encoding_path = get_project_base_directory("ragflow_deps", "cl100k_base.tiktoken")
    encoding_url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    cached_encoding_path = os.path.join(cache_dir, hashlib.sha1(encoding_url.encode()).hexdigest())

    if os.path.exists(bundled_encoding_path) and not os.path.exists(cached_encoding_path):
        shutil.copyfile(bundled_encoding_path, cached_encoding_path)

    return cache_dir


tiktoken_cache_dir = _ensure_tiktoken_cache()
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
# encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
encoder = tiktoken.get_encoding("cl100k_base")


# --- Embedding-model-aware tokenization -------------------------------------
# RAGFlow counts/truncates with cl100k (a GPT tokenizer) everywhere, which is
# badly wrong for the XLM-Roberta-based SEA-LION-E5 embedding model: it
# over-counts Thai ~4.7x (chunks over-split) and under-counts English ~0.82x
# (chunks exceed the model's 512-token limit -> silent tail loss). These helpers
# count/truncate in the embedding model's OWN vocabulary so chunk sizing and
# embedding truncation match what the model consumes. Uses the lightweight
# `tokenizers` lib to load the model's tokenizer.json directly (no transformers).
# Falls back to cl100k if the file is missing, so the system keeps working.
_EMB_TOKENIZER_FILE = os.environ.get("EMBEDDING_TOKENIZER_FILE", "/ragflow/conf/sealion_e5_tokenizer.json")
_emb_lock = threading.Lock()


@lru_cache(maxsize=1)
def _embedding_tokenizer():
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(_EMB_TOKENIZER_FILE)
    except Exception:
        return None


def num_tokens_from_string_for_embedding(string: str) -> int:
    """Token count in the EMBEDDING model's vocabulary (not cl100k). Use for chunk
    sizing so pieces match what the embedding model actually consumes."""
    t = _embedding_tokenizer()
    if t is None:
        return len(encoder.encode(string))  # cl100k fallback
    try:
        with _emb_lock:
            return len(t.encode(string).ids)
    except Exception:
        return len(encoder.encode(string))


def truncate_for_embedding(string: str, max_len: int) -> str:
    """Truncate to max_len EMBEDDING-model tokens (not cl100k) — prevents silent
    tail-loss when chunks are embedded."""
    t = _embedding_tokenizer()
    if t is None:
        return encoder.decode(encoder.encode(string)[:max_len])
    try:
        with _emb_lock:
            ids = t.encode(string).ids[:max_len]
            return t.decode(ids)
    except Exception:
        return encoder.decode(encoder.encode(string)[:max_len])


# Per-run token usage sink. An agent run (Canvas.run) installs a mutable dict here
# at the start of each turn; every LLMBundle chat call adds its provider-reported
# usage to it. This is the single chokepoint that aggregates token usage across all
# LLM calls in a run (query rewriting, cross-language translation, tool reasoning,
# and the final streamed answer) regardless of which component or helper issued the
# call. Default None means "not inside a tracked run" and callers must no-op.
token_usage_sink: contextvars.ContextVar = contextvars.ContextVar("ragflow_token_usage_sink", default=None)

# Per-run Langfuse correlating attributes (e.g. {"session_id": ..., "user_id": ...}).
# Installed by Canvas.run so RAGFlow's own Langfuse generations can be grouped by
# session and user even though the agent's LLMBundles are created without them.
langfuse_run_attrs: contextvars.ContextVar = contextvars.ContextVar("ragflow_langfuse_run_attrs", default=None)


# Guards sink mutations: concurrent tool calls (asyncio.gather + thread_pool_exec,
# which copies the context so worker threads share the same sink dict) can otherwise
# race on the read-modify-write of the counters.
_sink_lock = threading.Lock()


def record_run_token_usage(prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0) -> None:
    """Add a single LLM call's token usage to the active run sink, if any.

    Safe to call from anywhere: when no run sink is installed it does nothing.
    """
    sink = token_usage_sink.get()
    if sink is None:
        return
    try:
        with _sink_lock:
            sink["prompt_tokens"] += int(prompt_tokens or 0)
            sink["completion_tokens"] += int(completion_tokens or 0)
            sink["total_tokens"] += int(total_tokens or 0)
            sink["calls"] += 1
    except Exception:
        # Never let usage bookkeeping break a request; log at debug so a malformed
        # sink or token value is still traceable without adding noise.
        logging.debug("Failed to record run token usage", exc_info=True)


def usage_from_response(resp) -> dict:
    """Extract a {prompt_tokens, completion_tokens, total_tokens} split from an LLM response.

    Handles OpenAI/OpenRouter-style ``resp.usage`` objects and dict variants. Missing
    fields default to 0; ``total_tokens`` falls back to prompt+completion when absent.
    """
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if resp is None:
        return out

    usage = None
    try:
        usage = getattr(resp, "usage", None)
        if usage is None and isinstance(resp, dict):
            usage = resp.get("usage")
    except Exception:
        usage = None
    if usage is None:
        return out

    def _get(obj, *names):
        for n in names:
            try:
                v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
            except Exception:
                v = None
            if v:
                return int(v)
        return 0

    out["prompt_tokens"] = _get(usage, "prompt_tokens", "input_tokens")
    out["completion_tokens"] = _get(usage, "completion_tokens", "output_tokens")
    out["total_tokens"] = _get(usage, "total_tokens")
    if not out["total_tokens"]:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        code_list = encoder.encode(string)
        return len(code_list)
    except Exception:
        return 0


def total_token_count_from_response(resp):
    """
    Extract token count from LLM response in various formats.

    Handles None responses and different response structures from various LLM providers.
    Returns 0 if token count cannot be determined.
    """
    if resp is None:
        return 0

    try:
        if hasattr(resp, "usage") and hasattr(resp.usage, "total_tokens"):
            return resp.usage.total_tokens
    except Exception:
        pass

    try:
        if hasattr(resp, "usage_metadata") and hasattr(resp.usage_metadata, "total_tokens"):
            return resp.usage_metadata.total_tokens
    except Exception:
        pass

    try:
        if hasattr(resp, "meta") and hasattr(resp.meta, "billed_units") and hasattr(resp.meta.billed_units, "input_tokens"):
            return resp.meta.billed_units.input_tokens
    except Exception:
        pass

    if isinstance(resp, dict) and "usage" in resp and "total_tokens" in resp["usage"]:
        try:
            return resp["usage"]["total_tokens"]
        except Exception:
            pass

    if isinstance(resp, dict) and "usage" in resp and "input_tokens" in resp["usage"] and "output_tokens" in resp["usage"]:
        try:
            return resp["usage"]["input_tokens"] + resp["usage"]["output_tokens"]
        except Exception:
            pass

    if isinstance(resp, dict) and "meta" in resp and "tokens" in resp["meta"] and "input_tokens" in resp["meta"]["tokens"] and "output_tokens" in resp["meta"]["tokens"]:
        try:
            return resp["meta"]["tokens"]["input_tokens"] + resp["meta"]["tokens"]["output_tokens"]
        except Exception:
            pass
    return 0


def truncate(string: str, max_len: int) -> str:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])
