#!/usr/bin/env python3
"""
Standalone TokenChunker debug harness.

Runs the chunker in isolation against synthetic parser output to pinpoint
where chunks are lost. No Docker services needed (skips PDF preview restore).

Usage (inside the ragflow container or with PYTHONPATH set to ragflow/):
    cd /ragflow
    PYTHONPATH=/ragflow python3 rag/flow/tests/debug_chunker.py
    PYTHONPATH=/ragflow python3 rag/flow/tests/debug_chunker.py --mode delimiter --delimiters "\\n\\n"
    PYTHONPATH=/ragflow python3 rag/flow/tests/debug_chunker.py --json-input /path/to/parser_output.json
"""
import argparse
import asyncio
import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

logging.basicConfig(level=logging.WARNING)


class FakeCanvas:
    """Minimal stand-in for Graph so ProcessBase can initialise."""
    def __init__(self, tenant_id="test-tenant", doc_id=None):
        self._tenant_id = tenant_id
        self._doc_id = doc_id

    def callback(self, component_id, progress=None, message=""):
        if progress is not None and progress < 0:
            print(f"  [{component_id}] ERROR (progress={progress}): {message}")
        elif message:
            print(f"  [{component_id}] {message}")

    def is_canceled(self):
        return False

    def get_component(self, cid):
        return {"obj": SimpleNamespace(component_name="test"), "downstream": [], "upstream": []}

    def get_component_name(self, cid):
        return "test"


def make_param(**overrides):
    """Create a TokenChunkerParam with overrides."""
    from rag.flow.chunker.token_chunker import TokenChunkerParam
    p = TokenChunkerParam()
    for k, v in overrides.items():
        setattr(p, k, v)
    p.check()
    return p


def make_chunker(canvas, param, cid="TokenChunker:0"):
    """Instantiate a TokenChunker with a fake canvas."""
    from rag.flow.chunker.token_chunker import TokenChunker
    return TokenChunker(canvas, cid, param)


def sample_json_result(text_blocks=3, words_per_block=80):
    """Generate realistic parser-like JSON output (text blocks with positions)."""
    blocks = []
    lorem = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
    )
    for i in range(text_blocks):
        text = " ".join([lorem] * (words_per_block // 15 + 1))[:words_per_block * 6]
        blocks.append({
            "text": f"Section {i + 1}.\n{text}",
            "doc_type_kwd": "text",
            "page_number": i + 1,
            "x0": 72.0, "x1": 540.0, "top": 100.0 + i * 50, "bottom": 140.0 + i * 50,
        })
    return blocks


def build_upstream_kwargs(json_result, name="test.pdf", extra_keys=None):
    """Build the kwargs dict that Parser would pass to the chunker."""
    kwargs = {
        "_created_time": 0.0,
        "_elapsed_time": 0.5,
        "name": name,
        "file": {"id": "file-1", "created_by": "user-1"},
        "output_format": "json",
        "json": json_result,
    }
    if extra_keys:
        kwargs.update(extra_keys)
    return kwargs


async def run_chunker(chunker, kwargs):
    """Run the chunker and return (output, error)."""
    await chunker.invoke(**kwargs)
    out = chunker.output()
    err = chunker.error()
    return out, err


def report(label, param, kwargs, output, error):
    """Print a detailed report of the chunker run."""
    from rag.flow.chunker.token_chunker import _compile_delimiter_pattern

    print(f"\n{'='*70}")
    print(f"SCENARIO: {label}")
    print(f"{'='*70}")
    print(f"  delimiter_mode:  {param.delimiter_mode}")
    print(f"  chunk_token_size: {param.chunk_token_size}")
    print(f"  delimiters:       {param.delimiters}")
    print(f"  children_delimiters: {param.children_delimiters}")
    print(f"  overlapped_pct:   {param.overlapped_percent}")

    compiled = _compile_delimiter_pattern(param.delimiters)
    custom_pattern = "|".join(
        __import__("re").escape(t)
        for t in sorted(set(param.children_delimiters), key=len, reverse=True)
    ) if param.children_delimiters else ""
    print(f"  compiled delimiter_pattern: {compiled!r}")
    print(f"  compiled children_pattern:  {custom_pattern!r}")

    print(f"\n  INPUT: {len(kwargs.get('json', []))} json blocks, "
          f"output_format={kwargs.get('output_format')}")
    extra = {k: v for k, v in kwargs.items()
             if k not in ("_created_time", "_elapsed_time", "name", "file", "output_format", "json")}
    if extra:
        print(f"  EXTRA KEYS (may fail extra='forbid'): {list(extra.keys())}")

    if error:
        print(f"\n  *** ERROR: {error}")
    else:
        chunks = output.get("chunks", [])
        print(f"\n  OUTPUT: {len(chunks)} chunks")
        for i, c in enumerate(chunks[:5]):
            text = c.get("text", "")
            preview = text[:100].replace("\n", "\\n") + ("..." if len(text) > 100 else "")
            print(f"    [{i}] type={c.get('doc_type_kwd','text')} len={len(text)} tokens={len(text)//4} :: {preview}")
        if len(chunks) > 5:
            print(f"    ... and {len(chunks) - 5} more")

    # Diagnosis
    print(f"\n  DIAGNOSIS:", end=" ")
    if error:
        if "Input error" in str(error) or "validation" in str(error).lower():
            print("SCHEMA VALIDATION FAILED — an extra key was passed that the schema rejects (extra='forbid').")
        else:
            print("CHUNKER RAISED AN EXCEPTION.")
    elif not output.get("chunks"):
        print("NO CHUNKS — chunker completed but produced zero output. Check if all text was filtered/empty.")
    else:
        print(f"OK — {len(output['chunks'])} chunks produced.")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Debug the TokenChunker in isolation")
    parser.add_argument("--mode", default="token_size",
                        choices=["token_size", "delimiter", "one"],
                        help="delimiter_mode for the chunker")
    parser.add_argument("--delimiters", default=None,
                        help="Comma-separated delimiter strings (for 'delimiter' mode). "
                             "Wrap in backticks for regex: `\\n\\n`. Example: --delimiters '`\\n\\n`,`?`")
    parser.add_argument("--children", default=None,
                        help="Com-separated children delimiter strings")
    parser.add_argument("--token-size", type=int, default=512, help="chunk_token_size")
    parser.add_argument("--blocks", type=int, default=5, help="Number of sample text blocks")
    parser.add_argument("--extra-key", default=None,
                        help="Inject an extra key to test schema validation failure (e.g. --extra-key markdown_result)")
    parser.add_argument("--json-input", default=None,
                        help="Path to a JSON file with real parser output to use as input")
    args = parser.parse_args()

    canvas = FakeCanvas()

    # Load or generate input
    if args.json_input:
        with open(args.json_input) as f:
            json_result = json.load(f)
        print(f"Loaded {len(json_result)} blocks from {args.json_input}")
    else:
        json_result = sample_json_result(text_blocks=args.blocks)

    # Parse delimiter args
    delimiters = []
    if args.delimiters:
        raw = args.delimiters.strip()
        if raw.startswith("`") or "," in raw:
            parts = [p.strip() for p in raw.split(",")]
        else:
            parts = [raw]
        delimiters = parts

    children = []
    if args.children:
        children = [c.strip() for c in args.children.split(",")]

    extra_keys = None
    if args.extra_key:
        extra_keys = {args.extra_key: "surprise value that should not be here"}

    scenarios = []

    # Scenario 1: the requested config
    scenarios.append((
        f"Requested config (mode={args.mode}, delimiters={delimiters})",
        make_param(
            delimiter_mode=args.mode,
            delimiters=delimiters,
            children_delimiters=children,
            chunk_token_size=args.token_size,
        ),
        build_upstream_kwargs(json_result, extra_keys=extra_keys),
    ))

    if not args.json_input:
        # Scenario 2: default token_size (known-good baseline)
        scenarios.append((
            "Baseline: default token_size (known-good)",
            make_param(delimiter_mode="token_size", chunk_token_size=512),
            build_upstream_kwargs(json_result),
        ))

        # Scenario 3: delimiter mode with backtick-wrapped newlines
        scenarios.append((
            "Delimiter mode with backtick-wrapped \\n\\n",
            make_param(delimiter_mode="delimiter", delimiters=["`\\n\\n`"], chunk_token_size=512),
            build_upstream_kwargs(json_result),
        ))

        # Scenario 4: delimiter mode with RAW newline (no backticks — the frontend default)
        scenarios.append((
            "Delimiter mode with RAW \\n (no backticks — frontend sends this)",
            make_param(delimiter_mode="delimiter", delimiters=["\n"], chunk_token_size=512),
            build_upstream_kwargs(json_result),
        ))

        # Scenario 5: inject extra key to demonstrate schema failure
        scenarios.append((
            "Schema failure: extra 'markdown_result' key leaked from upstream",
            make_param(delimiter_mode="token_size", chunk_token_size=512),
            build_upstream_kwargs(json_result, extra_keys={"markdown_result": "leaked"}),
        ))

    print(f"\nTokenChunker Debug Harness")
    print(f"Input: {len(json_result)} blocks, "
          f"total chars: {sum(len(b.get('text','')) for b in json_result)}")

    for label, param, kwargs in scenarios:
        chunker = make_chunker(canvas, param)
        output, error = await run_chunker(chunker, kwargs)
        report(label, param, kwargs, output, error)


if __name__ == "__main__":
    asyncio.run(main())
