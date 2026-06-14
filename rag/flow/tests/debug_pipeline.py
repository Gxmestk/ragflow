#!/usr/bin/env python3
"""
Full pipeline debug harness — runs a dataflow Pipeline against a real document
and reports per-component output, errors, and chunk counts.

Usage (inside the ragflow container):
    cd /ragflow
    PYTHONPATH=/ragflow python3 rag/flow/tests/debug_pipeline.py \\
        --dsl /path/to/canvas_dsl.json --doc-id <document-id> --tenant-id <tenant-id>

Or extract the DSL from a canvas ID:
    PYTHONPATH=/ragflow python3 rag/flow/tests/debug_pipeline.py \\
        --canvas-id 98531ebe5af311f193b491b597f15fca --doc-id <doc-id>

To get the DSL without running, add --dump-dsl (writes to stdout).
"""
import argparse
import asyncio
import json
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_canvas_dsl(canvas_id):
    """Fetch DSL from the user_canvas table by canvas ID."""
    from api.db.services.canvas_service import UserCanvasService
    e, cvs = UserCanvasService.get_by_id(canvas_id)
    if not e:
        raise RuntimeError(f"Canvas {canvas_id} not found")
    return cvs.dsl


def patch_pipeline_for_debug():
    """Monkey-patch Pipeline.run to log each component's output after it runs."""
    from rag.flow.pipeline import Pipeline
    orig_run = Pipeline.run

    async def debug_run(self, **kwargs):
        from agent.canvas import Graph
        log_key = f"{self._flow_id}-{self.task_id}-logs"

        if not self.path:
            self.path.append("File")

        # Run File
        cpn = self.get_component_obj(self.path[0])
        await cpn.invoke(**kwargs)
        _report_component("File", cpn)

        if cpn.error():
            print(f"\n*** File component failed: {cpn.error()}")
            return {}

        idx = len(self.path) - 1
        cpn = self.get_component_obj(self.path[idx])
        idx += 1
        self.path.extend(cpn.get_downstream())

        while idx < len(self.path):
            last = self.get_component_obj(self.path[idx - 1])
            cur = self.get_component_obj(self.path[idx])
            await cur.invoke(**last.output())
            _report_component(self.path[idx], cur)
            if cur.error():
                print(f"\n*** {self.path[idx]} failed: {cur.error()}")
                return {}
            idx += 1
            self.path.extend(cur.get_downstream())

        final = self.get_component_obj(self.path[-1])
        out = final.output()
        print(f"\n{'='*70}")
        print(f"FINAL OUTPUT (from {self.path[-1]})")
        print(f"{'='*70}")
        for k, v in out.items():
            if isinstance(v, list):
                print(f"  {k}: list[{len(v)}]")
                for i, item in enumerate(v[:3]):
                    s = json.dumps(item, ensure_ascii=False, default=str)[:200]
                    print(f"    [{i}] {s}")
                if len(v) > 3:
                    print(f"    ... {len(v)-3} more")
            elif isinstance(v, str):
                print(f"  {k}: {v[:200]!r}")
            elif isinstance(v, dict):
                print(f"  {k}: dict keys={list(v.keys())[:10]}")
            else:
                print(f"  {k}: {v!r}")
        return out

    Pipeline.run = debug_run


def _report_component(cid, cpn):
    """Print a summary of one component's output."""
    out = cpn.output()
    err = cpn.error()
    name = cpn.component_name

    print(f"\n{'─'*70}")
    status = "ERROR" if err else "OK"
    print(f"[{cid}] ({name}) — {status}")

    if err:
        print(f"  ERROR: {err}")
        return

    for k, v in out.items():
        if k.startswith("_"):
            continue
        if isinstance(v, list):
            total_chars = sum(len(str(item.get("text", ""))) for item in v if isinstance(item, dict))
            print(f"  {k}: list[{len(v)}] ({total_chars} chars total)")
        elif isinstance(v, str) and len(v) > 100:
            print(f"  {k}: str[{len(v)}] = {v[:100]!r}...")
        elif isinstance(v, dict):
            print(f"  {k}: dict = {list(v.keys())}")
        elif v is not None:
            print(f"  {k}: {v!r}")


async def run(dsl, doc_id, tenant_id):
    """Run the pipeline with debug instrumentation."""
    from common import settings
    from rag.flow.pipeline import Pipeline

    settings.init_settings()

    patch_pipeline_for_debug()

    pipeline = Pipeline(
        dsl,
        tenant_id=tenant_id,
        doc_id=doc_id,
        task_id="debug-" + (doc_id or "notask")[:8],
        flow_id="debug-flow",
    )
    pipeline.reset()

    print(f"\nPipeline path: {pipeline.path}")
    print(f"Components: {list(pipeline.components.keys())}")

    # Print each component's params
    print(f"\n{'='*70}")
    print("COMPONENT PARAMETERS")
    print(f"{'='*70}")
    for cid, cpn_info in pipeline.components.items():
        obj = cpn_info["obj"]
        param = obj._param
        params_dict = {k: v for k, v in param.__dict__.items()
                       if not k.startswith("_") and k not in ("inputs", "outputs")}
        print(f"\n[{cid}] ({obj.component_name})")
        print(f"  {json.dumps(params_dict, indent=2, default=str, ensure_ascii=False)}")

    result = await pipeline.run()

    print(f"\n{'='*70}")
    if not result:
        print("PIPELINE PRODUCED NO OUTPUT — check component errors above.")
    elif "chunks" in result:
        chunks = result["chunks"]
        print(f"SUCCESS: {len(chunks)} chunks produced.")
    else:
        print(f"Pipeline completed but no 'chunks' key in final output.")
        print(f"Available keys: {list(result.keys())}")

    return result


def main():
    cli = argparse.ArgumentParser(description="Debug a dataflow pipeline")
    cli.add_argument("--dsl", help="Path to DSL JSON file")
    cli.add_argument("--canvas-id", help="Canvas ID to fetch DSL from DB")
    cli.add_argument("--doc-id", required=True, help="Document ID to process")
    cli.add_argument("--tenant-id", help="Tenant ID (auto-detected from doc if omitted)")
    cli.add_argument("--dump-dsl", action="store_true",
                     help="Just print the DSL JSON and exit (don't run)")
    args = cli.parse_args()

    if args.dsl:
        with open(args.dsl) as f:
            dsl = f.read()
    elif args.canvas_id:
        dsl = get_canvas_dsl(args.canvas_id)
    else:
        cli.error("Either --dsl or --canvas-id is required")

    if args.dump_dsl:
        # Pretty-print the DSL, focusing on component params
        dsl_obj = json.loads(dsl) if isinstance(dsl, str) else dsl
        comps = dsl_obj.get("components", {})
        for cid, info in comps.items():
            obj = info.get("obj", {})
            cn = obj.get("component_name", "?")
            params = obj.get("params", {})
            print(f"\n[{cid}] ({cn})")
            print(json.dumps(params, indent=2, ensure_ascii=False))
        return

    tenant_id = args.tenant_id
    if not tenant_id and args.doc_id:
        from api.db.services.document_service import DocumentService
        e, doc = DocumentService.get_by_id(args.doc_id)
        if e:
            tenant_id = doc.tenant_id
            print(f"Auto-detected tenant_id={tenant_id} from document.")

    if not tenant_id:
        cli.error("--tenant-id is required (or provide a valid --doc-id)")

    asyncio.run(run(dsl, args.doc_id, tenant_id))


if __name__ == "__main__":
    main()
