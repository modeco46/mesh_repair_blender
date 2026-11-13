"""Standalone worker launched by Blender's bundled Python."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


PACKAGE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACKAGE_DIR / "vendor"
if VENDOR_DIR.is_dir():
    sys.path.insert(0, str(VENDOR_DIR))

_DLL_HANDLES = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for directory in (VENDOR_DIR, VENDOR_DIR / "meshlib"):
        if directory.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))

# Executing this file directly makes the embedded core a top-level package.
sys.path.insert(0, str(PACKAGE_DIR))


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class Channel:
    def __init__(self, path: Path):
        self.stream = path.open("a", encoding="utf-8", buffering=1)

    def send(self, kind: str, value) -> None:
        self.stream.write(json.dumps({"type": kind, "value": value}, ensure_ascii=False, allow_nan=False) + "\n")

    def close(self) -> None:
        self.stream.close()


def _export_arrays(stl_path: Path, vertices_path: Path, faces_path: Path) -> None:
    import numpy as np
    from meshlib import mrmeshpy as mm, mrmeshnumpy as mn

    mesh = mm.loadMesh(stl_path)
    mesh.pack()
    np.save(vertices_path, np.asarray(mn.getNumpyVerts(mesh), dtype=np.float32), allow_pickle=False)
    np.save(faces_path, np.asarray(mn.getNumpyFaces(mesh.topology), dtype=np.int32), allow_pickle=False)


def main(argv=None) -> int:
    if argv is None and "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("check", "repair"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--progress-file", required=True, type=Path)
    parser.add_argument("--result-file", required=True, type=Path)
    parser.add_argument("--vertices", type=Path)
    parser.add_argument("--faces", type=Path)
    args = parser.parse_args(argv)
    args.job_dir.mkdir(parents=True, exist_ok=True)
    channel = Channel(args.progress_file)
    try:
        from core.engine import Settings, SharpeningRetryRequired, clean_json, run

        values = json.loads(args.settings.read_text(encoding="utf-8"))
        settings = Settings(**values)
        if args.operation == "check":
            from core.inspection import inspect_mesh

            report = inspect_mesh(args.source, lambda text: channel.send("log", text),
                                  lambda event: channel.send("progress", event))
        else:
            report = run(args.source, settings=settings, emit=lambda text: channel.send("log", text),
                         progress=lambda event: channel.send("progress", event), work_dir=args.job_dir)
            if args.vertices is None or args.faces is None:
                raise ValueError("Result array paths are required for repair.")
            _export_arrays(Path(report["artifacts"]["stl"]), args.vertices, args.faces)
        report = clean_json(report)
        report_path = args.job_dir.parent / "report.json"
        _atomic_json(report_path, report)
        _atomic_json(args.result_file, {"status": "ok", "operation": args.operation,
                                       "report": str(report_path)})
        return 0
    except Exception as exc:
        try:
            from core.engine import SharpeningRetryRequired, clean_json
        except Exception:
            SharpeningRetryRequired = ()

            def clean_json(value):
                return value
        if SharpeningRetryRequired and isinstance(exc, SharpeningRetryRequired):
            retry_report = clean_json(exc.report)
            report_path = args.job_dir.parent / "report.json"
            _atomic_json(report_path, retry_report)
            _atomic_json(args.result_file, {"status": "retry_without_sharpening", "message": str(exc),
                                           "report": str(report_path)})
            return 2
        trace_path = args.job_dir.parent / "worker-error.txt"
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")
        _atomic_json(args.result_file, {"status": "error", "message": str(exc), "trace": str(trace_path)})
        return 1
    finally:
        channel.close()


if __name__ == "__main__":
    raise SystemExit(main())
