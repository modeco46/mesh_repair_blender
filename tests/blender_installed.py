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


def write_open_relief(path):
    # Two folds cross the base. Removing base facets at the crossings leaves
    # small openings for which filling the rim would create an intersection.
    width, height = 384, 368
    x, y = np.meshgrid(np.linspace(-8, 8, width), np.linspace(-10, 10, height))
    z = (1.65+.35*np.cos(.65*x)*np.sin(.47*y)+.19*np.cos(3*np.hypot(x-1, y+.8))
         +.1*np.cos(3*y+.5*np.sin(2*x))+.28*np.cos(1.3*x+.1*y)*np.cos(.6*y))
    z -= 2.9*np.exp(-((x+.14)**2/.93**2+(y-.33)**2/1.27**2))
    z -= 2.8*np.exp(-((x-3.1)**2/1.1**2+(y+1.92)**2/1.1**2))
    top = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    base = top.copy()
    base[:, 2] = 0
    vertices = np.vstack((top, base)).astype('<f4')
    ids = np.arange(width*height, dtype=np.int32).reshape(height, width)
    a = ids[:-1, :-1].ravel()
    b, c, d = a+1, a+1+width, a+width
    upper = np.stack((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)), axis=1).reshape(-1, 3)
    heights = z.ravel()[upper]
    crossing = np.flatnonzero((heights.min(axis=1) < 0) & (heights.max(axis=1) > 0))
    centers = top[upper[crossing], :2].mean(axis=1)
    holes = [crossing[np.argmax(centers @ [np.cos(t), np.sin(t)])] for t in np.arange(12)*2*np.pi/12]
    bottom = np.delete(upper[:, ::-1]+ids.size, np.unique(holes), axis=0)
    rim = np.r_[ids[0], ids[1:, -1], ids[-1, -2::-1], ids[-2:0:-1, 0]]
    nxt = np.roll(rim, -1)
    faces = np.vstack((upper, bottom, np.stack((nxt, rim, rim+ids.size), axis=1),
                       np.stack((nxt, rim+ids.size, nxt+ids.size), axis=1)))
    triangles = np.zeros(len(faces), dtype=[('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    triangles['v'] = vertices[faces]
    with path.open('wb') as stream:
        stream.write(b'Open relief'.ljust(80, b'\0'))
        stream.write(len(faces).to_bytes(4, 'little'))
        triangles.tofile(stream)


def check_open_relief(addon):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'open-relief.stl'
        write_open_relief(path)
        bpy.ops.wm.stl_import(filepath=str(path))
    source = bpy.context.active_object
    bpy.ops.object.select_all(action='DESELECT')
    source.select_set(True)
    bpy.context.scene.mesh_repair_settings.quality_mm = .02
    assert bpy.ops.mesh_repair.repair_selected('INVOKE_DEFAULT') == {'RUNNING_MODAL'}
    runtime, report = finish(addon, addon._ACTIVE, timeout=420)
    assert runtime.stage == 'Готово', report.get('error', runtime.message)
    strategy = report['boundary_band']
    assert strategy['accepted'] and strategy['method'] == 'oriented_scan', strategy
    blocks = report['voxel_passes'][-1]['slabs']
    assert blocks['completed_layers'] == blocks['total_layers'] and blocks['projected_samples'] < blocks['unique_grid_samples']
    info = report['after']
    assert info['closed'] and info['topology_valid'] and info['components'] == 1
    assert info['boundary_loops'] == info['self_intersecting_faces'] == 0
    assert 505 < info['signed_volume_mm3'] < 510 and 1000 < info['triangles'] < 100_000
    assert len(runtime.result_object.data.polygons) == info['triangles'] and not runtime.result_object.data.polygons[0].use_smooth
    print('OPEN_RELIEF_INSTALLED_OK', info['triangles'], info['signed_volume_mm3'], report['elapsed_seconds'])


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
    assert report["blender_addon_version"] == "1.5.0" and report["engine_revision"] == "blender-1.5.0"
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
    check_open_relief(addon)


if __name__ == "__main__":
    main()
