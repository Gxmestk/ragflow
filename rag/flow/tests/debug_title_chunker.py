#!/usr/bin/env python3
"""
Reproduce the user's pipeline: Parser (output_format=json) → TitleChunker (hierarchy).

Uses DeepDOC parser since MinerU is not always running; both produce the same
output_format="json" structure that TitleChunker consumes, so this exercises the
exact code path that was broken.

Usage (inside container):
    cd /ragflow
    PYTHONPATH=/ragflow python3 rag/flow/tests/debug_title_chunker.py /tmp/test.pdf
    PYTHONPATH=/ragflow python3 rag/flow/tests/debug_title_chunker.py /tmp/test.pdf --canvas-id 98531ebe5af311f193b491b597f15fca
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

logging.basicConfig(level=logging.WARNING)


def patch_component_base():
    import agent.component.base as base_mod
    orig = base_mod.ComponentBase.__init__
    def patched(self, canvas, id, param):
        self._canvas = canvas
        self._id = id
        self._param = param
        if hasattr(param, "check"):
            param.check()
    base_mod.ComponentBase.__init__ = patched


class FakeCanvas:
    def __init__(self, tenant_id="debug-tenant"):
        self._tenant_id = tenant_id
        self._doc_id = None
        self.task_id = "debug-title-task"
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
        return {"obj": SimpleNamespace(component_name="x"), "downstream": [], "upstream": []}

    def get_component_name(self, cid):
        return cid.split(":")[0] if ":" in cid else cid

    def get_tenant_id(self):
        return self._tenant_id


def load_title_chunker_params():
    """Load real TitleChunker params from canvas DSL, or fall back to a Thai-aware default."""
    cli_args = sys.argv
    canvas_id = None
    for i, a in enumerate(cli_args):
        if a == "--canvas-id" and i + 1 < len(cli_args):
            canvas_id = cli_args[i + 1]
    if canvas_id:
        try:
            from api.db.services.canvas_service import UserCanvasService
            e, cvs = UserCanvasService.get_by_id(canvas_id)
            if e:
                dsl = json.loads(cvs.dsl) if isinstance(cvs.dsl, str) else cvs.dsl
                for cid, info in dsl.get("components", {}).items():
                    cn = info.get("obj", {}).get("component_name", "")
                    if cn == "TitleChunker":
                        params = info.get("obj", {}).get("params", {})
                        print(f"  [DSL] Loaded TitleChunker config from canvas {canvas_id}: method={params.get('method')}, hierarchy={params.get('hierarchy')}, levels={len(params.get('levels', []))} groups")
                        return params
        except Exception as e:
            print(f"  [DSL] Could not load canvas config: {e}; using default")

    return {
        "method": "hierarchy",
        "hierarchy": 3,
        "include_heading_content": False,
        "root_chunk_as_heading": False,
        "levels": [
            ["^#[^#]", "^##[^#]", "^###[^#]", "^####[^#]"],
            ["PART (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
             "Chapter (I+V?|VI*|XI|IX|X)", "Section [0-9]+", "Article [0-9]+"],
            ["^(บทที่|CHAPTER|PART|ภาคที่|MODULE|UNIT)\\s+[0-9A-Z๐-๙]+",
             "^(หัวข้อ|SECTION|TOPIC|บทเรียน|LESSON)\\s+[0-9A-Z๐-๙]+",
             "^(ข้อ|ITEM|POINT|CLAUSE)\\s+[0-9๐-๙]+"],
        ],
    }


async def run_parser(pdf_path, canvas):
    from common import settings
    settings.init_settings()
    from rag.flow.parser.parser import Parser, ParserParam

    with open(pdf_path, "rb") as f:
        blob = f.read()

    print(f"\n{'='*70}\nSTEP 1: PARSER (DeepDOC, output_format=json)\n{'='*70}")
    print(f"  File: {pdf_path} ({len(blob):,} bytes)")

    param = ParserParam()
    if "audio" in param.setups and not param.setups["audio"].get("vlm", {}).get("llm_id"):
        param.setups["audio"]["vlm"] = {"llm_id": "SenseVoiceSmall"}
    if "video" in param.setups and not param.setups["video"].get("vlm", {}).get("llm_id"):
        param.setups["video"]["vlm"] = {"llm_id": "SenseVoiceSmall"}
    param.check()

    parser = Parser(canvas, "Parser:0", param)
    with patch("api.db.services.file_service.FileService.get_blob", return_value=blob):
        await parser.invoke(
            name=os.path.basename(pdf_path),
            file={"id": "test-file", "created_by": "test-user"},
        )

    out = parser.output()
    err = parser.error()
    if err:
        print(f"  PARSER ERROR: {err}")
        return None

    print(f"  output_format: {out.get('output_format')}")
    print(f"  json blocks: {len(out.get('json', []))}")
    types = {}
    for b in out.get("json", []):
        t = b.get("doc_type_kwd", "text")
        types[t] = types.get(t, 0) + 1
    print(f"  block types: {types}")
    return out


async def run_title_chunker(parser_output, canvas, params_dict):
    from rag.flow.chunker.title_chunker.title_chunker import TitleChunker
    from rag.flow.chunker.title_chunker.common import TitleChunkerParam, BaseTitleChunker

    print(f"\n{'='*70}\nSTEP 2: TITLE CHUNKER\n{'='*70}")
    print(f"  method: {params_dict.get('method')}")
    print(f"  hierarchy: {params_dict.get('hierarchy')}")
    print(f"  levels groups: {len(params_dict.get('levels', []))}")

    param = TitleChunkerParam()
    param.method = params_dict.get("method", "hierarchy")
    param.hierarchy = params_dict.get("hierarchy", 3)
    param.include_heading_content = params_dict.get("include_heading_content", False)
    param.root_chunk_as_heading = params_dict.get("root_chunk_as_heading", False)
    param.levels = params_dict.get("levels", [])
    param.check()

    chunker = TitleChunker(canvas, "TitleChunker:0", param)
    await chunker.invoke(**parser_output)

    out = chunker.output()
    err = chunker.error()
    if err:
        print(f"\n  *** CHUNKER ERROR: {err}")
        return None

    chunks = out.get("chunks", [])
    print(f"\n  Chunks produced: {len(chunks)}")
    for i, c in enumerate(chunks[:5]):
        text = c.get("text", "")
        preview = text[:120].replace("\n", "\\n")
        print(f"    [{i}] len={len(text)} :: {preview}{'...' if len(text)>120 else ''}")
    if len(chunks) > 5:
        print(f"    ... and {len(chunks)-5} more")

    # Also report which level group was selected (for diagnostics)
    try:
        from rag.flow.chunker.title_chunker.schema import TitleChunkerFromUpstream
        from_upstream = TitleChunkerFromUpstream.model_validate(parser_output)
        instance = BaseTitleChunker.__new__(BaseTitleChunker)
        instance.param = param
        instance.from_upstream = from_upstream
        line_records = instance.extract_line_records()
        print(f"\n  [diag] extract_line_records returned {len(line_records)} records")
        if line_records:
            selected = instance.select_level_group(
                [r["text"] for r in line_records], param.levels
            )
            print(f"  [diag] selected level group has {len(selected)} patterns (of {len(param.levels)} groups)")
    except Exception as e:
        print(f"  [diag] could not inspect level selection: {e}")

    return out


async def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("pdf_path")
    cli.add_argument("--canvas-id", default=None)
    args = cli.parse_args()

    patch_component_base()
    canvas = FakeCanvas()

    parser_output = await run_parser(args.pdf_path, canvas)
    if parser_output is None:
        sys.exit(1)

    params_dict = load_title_chunker_params()
    out = await run_title_chunker(parser_output, canvas, params_dict)

    print(f"\n{'='*70}")
    if out and out.get("chunks"):
        print(f"RESULT: {len(out['chunks'])} chunks — FIX VERIFIED")
    else:
        print(f"RESULT: 0 chunks — BUG STILL PRESENT")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
