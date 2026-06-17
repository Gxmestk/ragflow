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


import os
import tiktoken

from common.file_utils import get_project_base_directory

tiktoken_cache_dir = get_project_base_directory()
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
from functools import lru_cache
import threading

_EMB_TOKENIZER_FILE = os.environ.get("EMBEDDING_TOKENIZER_FILE", "/ragflow/conf/sealion_e5_tokenizer.json")
_emb_lock = threading.Lock()


@lru_cache(maxsize=1)
def _embedding_tokenizer():
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(_EMB_TOKENIZER_FILE)
    except Exception:
        return None


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        code_list = encoder.encode(string)
        return len(code_list)
    except Exception:
        return 0


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

    if isinstance(resp, dict) and 'usage' in resp and 'total_tokens' in resp['usage']:
        try:
            return resp["usage"]["total_tokens"]
        except Exception:
            pass

    if isinstance(resp, dict) and 'usage' in resp and 'input_tokens' in resp['usage'] and 'output_tokens' in resp['usage']:
        try:
            return resp["usage"]["input_tokens"] + resp["usage"]["output_tokens"]
        except Exception:
            pass

    if isinstance(resp, dict) and 'meta' in resp and 'tokens' in resp['meta'] and 'input_tokens' in resp['meta']['tokens'] and 'output_tokens' in resp['meta']['tokens']:
        try:
            return resp["meta"]["tokens"]["input_tokens"] + resp["meta"]["tokens"]["output_tokens"]
        except Exception:
            pass
    return 0


def truncate(string: str, max_len: int) -> str:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])


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
