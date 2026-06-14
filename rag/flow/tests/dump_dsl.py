"""Dump the actual Parser + TitleChunker params from a canvas DSL."""
import json
import sys
from api.db.services.canvas_service import UserCanvasService


def main(canvas_id):
    e, cvs = UserCanvasService.get_by_id(canvas_id)
    if not e:
        print(f"Canvas {canvas_id} not found", file=sys.stderr)
        sys.exit(1)
    dsl = json.loads(cvs.dsl) if isinstance(cvs.dsl, str) else cvs.dsl
    out = {"components": {}}
    for cid, info in dsl.get("components", {}).items():
        cn = info.get("obj", {}).get("component_name", "")
        if cn in ("TitleChunker", "Parser"):
            out["components"][cid] = {
                "component_name": cn,
                "params": info.get("obj", {}).get("params", {}),
            }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
