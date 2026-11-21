"""Run in Blender with the release ZIP installed/enabled through the extension manager."""
import importlib
import json
import os
from types import SimpleNamespace
import time

import bpy
import numpy as np


def finish(addon, job):
    job["process"].wait(timeout=180)
    owner = SimpleNamespace(_mesh_repair_job_id=job["id"])
    assert addon._MeshRepairModal.modal(owner, bpy.context, SimpleNamespace(type="TIMER")) == {"FINISHED"}
    runtime = bpy.context.window_manager.mesh_repair_runtime
    assert not runtime.busy
    return runtime, json.loads(bpy.data.texts[runtime.report_text].as_string())


def main():
    package = "bl_ext.user_default.mesh_repair_blender"
    assert package in bpy.context.preferences.addons, "ZIP must be installed and enabled"
    addon = importlib.import_module(package + ".addon")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = .001
    bpy.ops.wm.stl_import(filepath=os.environ["MESH_REPAIR_FIXTURE"])
    settings = scene.mesh_repair_settings
    assert abs(settings.quality_mm - .02) < 1e-6 and not settings.disable_cuda
    source = bpy.context.active_object
    assert len(source.data.polygons) > 190000

    # Invoke the installed operators. Background Blender has no interactive
    # event loop, so deliver the final timer explicitly after each worker exits.
    assert bpy.ops.mesh_repair.check_selected("INVOKE_DEFAULT") == {"RUNNING_MODAL"}
    job = addon._ACTIVE
    runtime = bpy.context.window_manager.mesh_repair_runtime
    until = time.monotonic()+45
    while not job.get("last_worker_progress") and time.monotonic() < until:
        addon._read_progress(job, runtime)
        if job["process"].poll() is not None:
            break
        time.sleep(.05)
    assert bpy.ops.mesh_repair.stop() == {"FINISHED"}
    runtime, cancelled = finish(addon, job)
    assert cancelled["status"] == "cancelled" and cancelled["last_progress"]["stage"]
    assert cancelled["elapsed_seconds"] > 0 and not source.hide_get()

    assert bpy.ops.mesh_repair.repair_selected("INVOKE_DEFAULT") == {"RUNNING_MODAL"}
    job = addon._ACTIVE
    runtime, report = finish(addon, job)
    assert runtime.stage == "Готово", report.get("error", runtime.message)
    assert report["blender_addon_version"] == "1.4.0" and report["engine_revision"] == "blender-1.4.0"
    info = report["after"]
    assert info["closed"] and info["topology_valid"] and info["components"] == 1
    assert info["boundary_loops"] == info["self_intersecting_faces"] == 0
    assert abs(info["signed_volume_mm3"] - 366.6) < 4
    mesh = runtime.result_object.data
    assert len(mesh.polygons) == info["triangles"] < report["before"]["triangles"]
    assert source.hide_get() and not mesh.polygons[0].use_smooth
    faces = np.load(job["faces"], allow_pickle=False)
    actual = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", actual)
    assert np.array_equal(actual.reshape(-1, 3), faces)
    print("INSTALLED_REPAIR_OK", info["triangles"], info["signed_volume_mm3"], report["elapsed_seconds"])


if __name__ == "__main__":
    main()
