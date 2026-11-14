"""Run with Blender, not pytest: end-to-end bridge and worker smoke test."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import bpy


ROOT = Path(os.environ.get("MESH_REPAIR_ADDON_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

import mesh_repair_blender  # noqa: E402
from mesh_repair_blender.bridge import create_mesh_object, write_selected_stl  # noqa: E402


def main():
    mesh_repair_blender.register()
    bpy.ops.mesh.primitive_cube_add(size=2)
    with tempfile.TemporaryDirectory(prefix="mesh-repair-blender-smoke-") as folder:
        folder = Path(folder)
        exported = write_selected_stl(bpy.context, folder / "source.stl", "MILLIMETERS")
        settings = folder / "settings.json"
        settings.write_text(json.dumps({"quality_mm": .05, "disable_cuda": True}), encoding="utf-8")
        python = os.environ.get("MESH_REPAIR_PYTHON")
        if not python:
            raise RuntimeError("Set MESH_REPAIR_PYTHON to a Python with MeshLib/SciPy installed.")
        result = folder / "result.json"
        command = [
            python, str(ROOT / "mesh_repair_blender" / "worker.py"), "repair",
            "--source", str(folder / "source.stl"), "--settings", str(settings),
            "--job-dir", str(folder / "job"), "--progress-file", str(folder / "progress.jsonl"),
            "--result-file", str(result), "--vertices", str(folder / "vertices.npy"),
            "--faces", str(folder / "faces.npy"),
        ]
        subprocess.run(command, check=True)
        assert json.loads(result.read_text(encoding="utf-8"))["status"] == "ok"
        repaired = create_mesh_object(bpy.context, folder / "vertices.npy", folder / "faces.npy",
                                      "Cube_repaired", exported["mm_per_blender_unit"])
        assert len(repaired.data.polygons) == 12
    mesh_repair_blender.unregister()
    print("MESH_REPAIR_BLENDER_SMOKE_OK")


if __name__ == "__main__":
    main()
