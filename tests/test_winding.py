import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn

from mesh_repair_blender.core import engine, holes
from mesh_repair_blender.core.deviation import recover_exterior_details
from mesh_repair_blender.core.engine import Progress, Settings, diagnostics, valid
from mesh_repair_blender.core.winding import winding_slabs


@pytest.mark.parametrize("sharpen", [False, True])
def test_partitioned_grid_matches_native_surface_and_sharpening(sharpen):
    mesh = mm.makeCube(mm.Vector3f(.7, .6, .9), mm.Vector3f(.11, -.3, .23))
    params = mm.GeneralOffsetParameters() if sharpen else mm.OffsetParameters()
    params.voxelSize = .052
    params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
    params.closeHolesInHoleWindingNumber = True
    events = []
    # Both devices implement the same SDK interface. Exercise the grid assembly
    # and its boundary planes even on build machines without a CUDA device.
    result = winding_slabs(mesh, mm, mm.FastWindingNumber(mesh), params, events.append,
                           sharpen=sharpen, budget=710)
    if sharpen:
        params.mode = mm.GeneralOffsetParametersMode.Sharpening
        native = mm.generalOffsetMesh(mm.MeshPart(mesh), 0, params)
    else:
        native = mm.mcOffsetMesh(mm.MeshPart(mesh), 0, params)
    assert valid(diagnostics(result, mm))
    assert events[0]['completed_layers'] == 0 and events[-1]['completed_layers'] == events[-1]['total_layers']
    assert events[-1]['completed_blocks'] == len(events)-1 > 2
    assert result.topology.numValidFaces() == native.topology.numValidFaces()
    assert mm.findMaxDistanceSq(mm.MeshPart(result), mm.MeshPart(native)) < 1e-10


def test_closed_fixed_rim_uses_cpu_band_even_on_cuda_machine(monkeypatch):
    mesh = mm.makeCube(mm.Vector3f(.4, .5, .6))
    class Device:
        def FastWindingNumber(self, mesh):
            pytest.fail('Closed narrow band should not allocate a full-field CUDA winding calculator')
    monkeypatch.setattr(engine, 'cuda_module', lambda: Device())
    monkeypatch.setattr(engine, 'DENSE_WINDING_VOXEL_LIMIT', 100)
    report = {'hole_filling': {'details': [{'method': 'fixed_rim_triangulation'}]}, 'warnings': []}
    events = []
    result = engine.sparse_repair(mesh, mm, Settings(.025, restore_sharp_edges=False),
                                  Progress(lambda _: None, checkpoint=events.append), report, lambda _: None)
    assert valid(diagnostics(result, mm))
    assert len(report['voxel_passes']) == 1
    assert events[0]['voxel_pass']['field_evaluation'] == 'closed_cpu_surface_band'


def test_colliding_smooth_lid_retries_untouched_rim(monkeypatch):
    mesh = mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(0, 0, 0))
    verts = mn.getNumpyVerts(mesh)
    faces = mn.getNumpyFaces(mesh.topology)
    mesh.deleteFaces(mn.faceBitSetFromBools((verts[faces, 2] == 1).all(axis=1)))
    monkeypatch.setattr(holes, 'planar_boundary', lambda *args: False)
    native = holes.patch_collisions
    def collisions(before, after, patch, sdk):
        if after.topology.numValidVerts() > before.topology.numValidVerts():
            return patch
        return native(before, after, patch, sdk)
    monkeypatch.setattr(holes, 'patch_collisions', collisions)
    result, report = holes.fill_simple_holes(mesh, mm, .02, defer_intersections=True)
    assert report['details'][0]['method'] == 'fixed_rim_triangulation'
    assert report['pending_local_fix'] == 0 and valid(diagnostics(result, mm))
    np.testing.assert_array_equal(mn.getNumpyVerts(result), verts)


def test_small_exterior_error_recovered_without_changing_connectivity():
    mesh = mm.makeCube(mm.Vector3f(3, 3, 3), mm.Vector3f(0, 0, 0))
    source = mm.Mesh(mesh)
    vertex = mm.VertId(0)
    source.points[vertex] = source.points[vertex] + mm.Vector3f(-.017, 0, 0)
    source.invalidateCaches()
    report = {}
    result, checked = recover_exterior_details(source, mesh, mesh, mm, .015, {'cuda_disabled': True}, report)
    assert report['accepted'] and report['exterior_over_before'] == 1 and report['exterior_over_after'] == 0
    assert valid(checked)
    np.testing.assert_array_equal(mn.getNumpyFaces(result.topology), mn.getNumpyFaces(mesh.topology))


def test_failed_cuda_batch_restarts_full_field_on_cpu(monkeypatch):
    from types import SimpleNamespace
    from mesh_repair_blender.core import winding
    mesh = mm.makeCube(mm.Vector3f(.4, .5, .6))
    mesh.topology.deleteFace(mm.FaceId(0))
    monkeypatch.setattr(engine, 'cuda_module', lambda: SimpleNamespace(FastWindingNumber=mm.FastWindingNumber))
    monkeypatch.setattr(engine, 'DENSE_WINDING_VOXEL_LIMIT', 100)
    def failed_block(mesh, mm, kernel, params, completed, **kwargs):
        completed(dict(total_layers=10, completed_layers=2, completed_blocks=1))
        raise RuntimeError('test device error')
    monkeypatch.setattr(winding, 'winding_slabs', failed_block)
    report = {'warnings': []}
    result = engine.sparse_repair(mesh, mm, Settings(.03, restore_sharp_edges=False),
                                  Progress(lambda _: None), report, lambda _: None)
    attempted = report['voxel_passes'][0]
    assert attempted['slabs']['completed_blocks'] == 1
    assert attempted['fallback_device'] == 'cpu_native_parallel' and attempted['cuda_error'] == 'test device error'
    assert valid(diagnostics(result, mm))


def test_original_torus_handle_is_preserved():
    from mesh_repair_blender.core.seams import close_voxel_handle
    mesh = mm.makeTorus(1, .2, 16, 16)
    info = diagnostics(mesh, mm)
    report = {}
    out, checked = close_voxel_handle(mesh, info, info, mesh, mm, .015, report)
    assert out is mesh and checked == info and not report
