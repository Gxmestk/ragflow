#!/usr/bin/env python3
"""
Run a real PDF through Parser → TokenChunker and report every step.
Bypasses MinIO storage by reading the file from disk.

Usage (inside container):
    cd /ragflow
    PYTHONPATH=/ragflow python3 rag/flow/tests/debug_pdf.py /tmp/test.pdf
    PYTHONPATH=/ragflow python3 rag/flow/tests/debug_pdf.py /tmp/test.pdf --mode delimiter --delimiters '`\\n\\n`'
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.WARNING)

# Patch the isinstance assertion so FakeCanvas works without full Graph init
_orig_component_init = None
def _patch_component_base():
    import agent.component.base as base_mod
    global _orig_component_init
    _orig_component_init = base_mod.ComponentBase.__init__
    def _patched_init(self, canvas, id, param):
        from agent.component.base import ComponentParamBase
        self._canvas = canvas
        self._id = id
        self._param = param
        if hasattr(param, 'check'):
            param.check()
    base_mod.ComponentBase.__init__ = _patched_init
_patch_component_base()


class FakeCanvas:
    """Minimal Graph stand-in that passes the isinstance check."""
    def __new__(cls, *args, **kwargs):
        from agent.canvas import Graph
        obj = object.__new__(cls)
        # Set _is_graph marker that isinstance checks may use
        return obj

    def __init__(self, tenant_id="test-tenant"):
        from agent.canvas import Graph
        self._tenant_id = tenant_id
        self._doc_id = None
        self.task_id = "debug-task"
        self.path = []
        self.components = {}
        self._global_variables = {}

    def callback(self, component_id, progress=None, message=""):
        if message:
            tag = "ERROR" if (progress is not None and progress < 0) else "INFO"
            print(f"  [{component_id}][{tag}] {message}")

    def is_canceled(self):
        return False

    def get_component(self, cid):
        return {"obj": SimpleNamespace(component_name="test"), "downstream": [], "upstream": []}

    def get_component_name(self, cid):
        return cid.split(":")[0] if ":" in cid else cid


async def run_parser(pdf_path, canvas):
    """Run the Parser component on a local PDF file."""
    from common import settings
    settings.init_settings()
    from rag.flow.parser.parser import Parser, ParserParam

    with open(pdf_path, "rb") as f:
        blob = f.read()

    print(f"\n{'='*70}")
    print(f"STEP 1: PARSER")
    print(f"{'='*70}")
    print(f"  File: {pdf_path} ({len(blob):,} bytes)")

    param = ParserParam()
    # Provide required VLM configs so check() passes
    if "audio" in param.setups and not param.setups["audio"].get("vlm", {}).get("llm_id"):
        param.setups["audio"]["vlm"] = {"llm_id": "SenseVoiceSmall"}
    if "video" in param.setups and not param.setups["video"].get("vlm", {}).get("llm_id"):
        param.setups["video"]["vlm"] = {"llm_id": "SenseVoiceSmall"}
    param.check()

    parser = Parser(canvas, "Parser:0", param)

    # Monkey-patch FileService.get_blob to return our local bytes
    with patch("api.db.services.file_service.FileService.get_blob", return_value=blob):
        kwargs = {
            "name": os.path.basename(pdf_path),
            "file": {"id": "test-file", "created_by": "test-user"},
        }
        await parser.invoke(**kwargs)

    out = parser.output()
    err = parser.error()

    if err:
        print(f"  PARSER ERROR: {err}")
        return None, err

    print(f"\n  Parser output keys: {[k for k in out.keys() if not k.startswith('_')]}")
    print(f"  output_format: {out.get('output_format')}")

    json_result = out.get("json", [])
    print(f"  json blocks: {len(json_result)}")

    if json_result:
        total_chars = sum(len(b.get("text", "")) for b in json_result)
        types = {}
        for b in json_result:
            t = b.get("doc_type_kwd", "text")
            types[t] = types.get(t, 0) + 1
        print(f"  total text chars: {total_chars:,}")
        print(f"  block types: {types}")
        print(f"  first block preview:")
        first = json_result[0]
        text = first.get("text", "")
        print(f"    text ({len(text)} chars): {text[:200]!r}")
        print(f"    keys: {list(first.keys())}")

    return out, None


async def run_chunker(parser_output, canvas, chunker_params):
    """Run the TokenChunker with the parser's output."""
    from rag.flow.chunker.token_chunker import TokenChunker, TokenChunkerParam

    print(f"\n{'='*70}")
    print(f"STEP 2: TOKEN CHUNKER")
    print(f"{'='*70}")

    param = TokenChunkerParam()
    for k, v in chunker_params.items():
        setattr(param, k, v)
    param.check()

    print(f"  delimiter_mode: {param.delimiter_mode}")
    print(f"  chunk_token_size: {param.chunk_token_size}")
    print(f"  delimiters: {param.delimiters}")
    print(f"  children_delimiters: {param.children_delimiters}")
    print(f"  overlapped_percent: {param.overlapped_percent}")

    chunker = TokenChunker(canvas, "TokenChunker:0", param)

    # The kwargs the chunker receives = parser's full output dict
    await chunker.invoke(**parser_output)

    out = chunker.output()
    err = chunker.error()

    if err:
        print(f"\n  *** CHUNKER ERROR: {err}")
        return None, err

    chunks = out.get("chunks", [])
    print(f"\n  Chunks produced: {len(chunks)}")

    for i, c in enumerate(chunks[:5]):
        text = c.get("text", "")
        preview = text[:150].replace("\n", "\\n")
        print(f"    [{i}] type={c.get('doc_type_kwd','?')} len={len(text)} :: {preview}{'...' if len(text)>150 else ''}")
    if len(chunks) > 5:
        print(f"    ... and {len(chunks)-5} more")

    return out, None


async def main():
    cli = argparse.ArgumentParser(description="Debug PDF through Parser→Chunker")
    cli.add_argument("pdf_path", help="Path to PDF file")
    cli.add_argument("--mode", default="token_size",
                     choices=["token_size", "delimiter", "one"])
    cli.add_argument("--delimiters", default=None,
                     help="Delimiters for 'delimiter' mode, backtick-wrapped. E.g: '`\\n\\n`'")
    cli.add_argument("--token-size", type=int, default=512)
    cli.add_argument("--all-modes", action="store_true",
                     help="Test all three delimiter modes")
    args = cli.parse_args()

    canvas = FakeCanvas()

    # Step 1: Parse the PDF
    parser_output, err = await run_parser(args.pdf_path, canvas)
    if err:
        print(f"\nParser failed, cannot continue.")
        sys.exit(1)

    # Step 2: Run chunker with different configs
    if args.all_modes:
        configs = [
            ("token_size (default)", {"delimiter_mode": "token_size", "chunk_token_size": 512}),
            ("delimiter with raw \\n", {"delimiter_mode": "delimiter", "delimiters": ["\n"]}),
            ("delimiter with backtick \\n", {"delimiter_mode": "delimiter", "delimiters": ["`\n`"]}),
            ("one chunk", {"delimiter_mode": "one"}),
        ]
    else:
        params = {"delimiter_mode": args.mode, "chunk_token_size": args.token_size}
        if args.delimiters:
            params["delimiters"] = [d.strip() for d in args.delimiters.split(",")]
        configs = [(f"requested ({args.mode})", params)]

    for label, params in configs:
        print(f"\n{'─'*70}")
        print(f"CONFIG: {label}")
        out, err = await run_chunker(parser_output, canvas, params)
        if err:
            print(f"  → FAILED")
        elif out and out.get("chunks"):
            print(f"  → OK: {len(out['chunks'])} chunks")
        else:
            print(f"  → NO CHUNKS (zero output)")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
