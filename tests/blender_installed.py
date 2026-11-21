"""Run in Blender with the release ZIP installed/enabled through the extension manager."""
import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import time

import bpy
import numpy as np


def finish(addon, job, timeout=180):
    job["process"].wait(timeout=timeout)
    owner = SimpleNamespace(_mesh_repair_job_id=job["id"])
    assert addon._MeshRepairModal.modal(owner, bpy.context, SimpleNamespace(type="TIMER")) == {"FINISHED"}
    runtime = bpy.context.window_manager.mesh_repair_runtime
    assert not runtime.busy
    return runtime, json.loads(bpy.data.texts[runtime.report_text].as_string())


def write_folded_ring(path):
    # Deliberate self-contact, two missing facets, and a reversed patch. The
    # body has one through-hole. Write strips to avoid a 2M-element Python list.
    rows, columns = 1500, 720
    u = np.arange(rows)[:, None] * 2*np.pi/rows
    v = (np.arange(columns)[None, :] + .3) * 2*np.pi/columns
    radius = 2.4 + .7*np.cos(v) + .03*np.cos(7*u)
    z = .74*np.sin(v)*(.48+np.cos(v)) + .037*np.sin(5*u)
    points = np.stack((radius*np.cos(u), radius*np.sin(u), z), axis=-1).astype('<f4')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    with path.open('wb') as stream:
        stream.write(b'Folded ring'.ljust(80, b'\0'))
        stream.write((2*rows*columns-2).to_bytes(4, 'little'))
        for i in range(rows):
            a, b = points[i], points[(i+1) % rows]
            c, d = np.roll(b, -1, axis=0), np.roll(a, -1, axis=0)
            strip = np.zeros(2*columns, dtype=dtype)
            strip['vertices'][::2] = np.stack((a, b, c), axis=1)
            strip['vertices'][1::2] = np.stack((a, c, d), axis=1)
            if 90 <= i < 96:
                strip['vertices'][80:90] = strip['vertices'][80:90, ::-1]
            if i in (100, 1000):
                strip = np.delete(strip, 321)
            strip.tofile(stream)


def check_large_repair(addon):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'folded-ring.stl'
        write_folded_ring(path)
        bpy.ops.wm.stl_import(filepath=str(path))
    source = bpy.context.active_object
    bpy.ops.object.select_all(action='DESELECT')
    source.select_set(True)
    bpy.context.scene.mesh_repair_settings.quality_mm = .01
    assert bpy.ops.mesh_repair.repair_selected("INVOKE_DEFAULT") == {"RUNNING_MODAL"}
    runtime, report = finish(addon, addon._ACTIVE, timeout=420)
    assert runtime.stage == 'Готово', report.get('error', runtime.message)
    assert report['before']['triangles'] == 2_159_998
    assert report['algorithm'] == 'closed_openvdb_level_set', report['voxel_passes']
    assert report['dense_routing']['needs_voxels'] and not report['local_repair']['attempted']
    seed, *rest = report['simplification']['attempts']
    assert seed['phase'] == 'dense_seed' and all(p['input_triangles'] < seed['input_triangles'] for p in rest)
    info = report['after']
    assert info['topology_valid'] and info['closed'] and info['components'] == 1
    assert info['boundary_loops'] == info['self_intersecting_faces'] == 0
    assert 2*info['vertices'] == info['triangles']  # Exactly one handle.
    assert info['triangles'] < 200_000 and 15.0 < info['signed_volume_mm3'] < 15.5
    deviation = report['source_deviation']['max_deviation_all_mm']
    assert deviation <= .010001 and len(runtime.result_object.data.polygons) == info['triangles']
    print('DENSE_INSTALLED_REPAIR_OK', info['triangles'], info['signed_volume_mm3'], deviation, report['elapsed_seconds'])


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
    runtime = bpy.context.window_manager.mesh_repair_runtime
    status = bpy.ops.mesh_repair.check_selected("INVOKE_DEFAULT")
    assert status == {"RUNNING_MODAL"}, (status, runtime.stage, runtime.message)
    job = addon._ACTIVE
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
    assert cancelled["events"] and all("elapsed_seconds" in event for event in cancelled["events"])

    status = bpy.ops.mesh_repair.repair_selected("INVOKE_DEFAULT")
    assert status == {"RUNNING_MODAL"}, (status, runtime.stage, runtime.message)
    job = addon._ACTIVE
    runtime, report = finish(addon, job)
    assert runtime.stage == "Готово", report.get("error", runtime.message)
    assert report["blender_addon_version"] == "1.4.2" and report["engine_revision"] == "blender-1.4.2"
    assert job["diagnostics"]["after_hole_filling"] == report["after_hole_filling"]
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
    check_large_repair(addon)


if __name__ == "__main__":
    main()
