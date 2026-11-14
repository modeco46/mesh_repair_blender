"""Run with Blender, not pytest: end-to-end bridge and worker smoke test."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import bpy
import numpy as np


ROOT = Path(os.environ.get("MESH_REPAIR_ADDON_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

import mesh_repair_blender  # noqa: E402
from mesh_repair_blender.bridge import create_mesh_object, write_selected_stl  # noqa: E402
from mesh_repair_blender import addon, heatmap  # noqa: E402


def main():
    mesh_repair_blender.register()
    settings = bpy.context.scene.mesh_repair_settings
    assert abs(settings.quality_mm - .02) < 1e-6
    assert all(getattr(settings, name) for name in (
        "collapse_coplanar", "simplify_curved", "weld_close_vertices", "remove_small_shells", "hide_sources"))
    assert not settings.restore_sharp_edges and not settings.disable_cuda and not settings.strict_tolerance
    assert not settings.show_advanced and not hasattr(settings, "unit_mode")
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.length_unit = "MILLIMETERS"
    bpy.context.scene.unit_settings.scale_length = .001
    bpy.ops.mesh.primitive_cube_add(size=2)
    with tempfile.TemporaryDirectory(prefix="mesh-repair-blender-smoke-") as folder:
        folder = Path(folder)
        exported = write_selected_stl(bpy.context, folder / "source.stl", "SCENE")
        assert abs(exported["mm_per_blender_unit"] - 1) < 1e-6
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
        assert not any(face.use_smooth for face in repaired.data.polygons)
        heatmap.attach(repaired, np.linspace(0, .04, len(repaired.data.vertices)), .02)
        assert heatmap.available(repaired)
        spaces = [a.spaces.active for a in bpy.context.screen.areas if a.type == "VIEW_3D"]
        previous = [(s.shading.type, s.shading.color_type, s.shading.light) for s in spaces]
        heatmap.show(bpy.context, repaired)
        assert heatmap.active(repaired)
        assert all(s.shading.color_type == "VERTEX" and s.shading.light == "FLAT" for s in spaces)
        heatmap.hide()
        assert [(s.shading.type, s.shading.color_type, s.shading.light) for s in spaces] == previous
        # TIMER progress must explicitly invalidate the 3D area's sidebar without a mouse event.
        redraws = []
        area = SimpleNamespace(type="VIEW_3D", tag_redraw=lambda: redraws.append(True))
        runtime = bpy.context.window_manager.mesh_repair_runtime
        fake_context = SimpleNamespace(window_manager=SimpleNamespace(
            mesh_repair_runtime=runtime, windows=[SimpleNamespace(screen=SimpleNamespace(areas=[area]))]))
        progress_file = folder / "timer-progress.jsonl"
        progress_file.write_text('{"type":"progress","value":{"percent":35,"stage":"test"}}\n')
        addon._ACTIVE = {"id": "test", "timer": "test-timer", "progress_file": progress_file,
                         "progress_offset": 0, "progress_pending": b"", "process": SimpleNamespace(poll=lambda: None)}
        modal = addon._MeshRepairModal()
        modal._mesh_repair_job_id = "test"
        # Match Blender's Event API: there is no event.timer attribute.
        assert "timer" not in bpy.types.Event.bl_rna.properties
        assert modal.modal(fake_context, SimpleNamespace(type="TIMER")) == {"PASS_THROUGH"}
        assert runtime.progress == 35 and redraws
        addon._ACTIVE = None
    mesh_repair_blender.unregister()
    print("MESH_REPAIR_BLENDER_SMOKE_OK")


if __name__ == "__main__":
    main()
