import numpy as np
import pytest
from meshlib import mrmeshpy as mm

from mesh_repair_blender.core import engine, scan_field, winding


def punctured_sphere():
    mesh = mm.makeUVSphere(.65, 33, 24)
    mesh.topology.deleteFace(mm.FaceId(10))
    return mesh


@pytest.mark.parametrize('shape', ['puncture', 'handle', 'cavity', 'overlap', 'axis_aligned'])
def test_scan_and_native_field_on_the_same_grid(shape):
    mesh = punctured_sphere()
    step = .043
    if shape == 'handle':
        mesh = mm.makeTorus(.8, .26, 35, 15)
        mesh.topology.deleteFace(mm.FaceId(10))
    elif shape in ('cavity', 'overlap'):
        origin = mm.Vector3f.diagonal(-.22) if shape == 'cavity' else mm.Vector3f(.23, -.23, -.22)
        inner = mm.makeCube(mm.Vector3f.diagonal(.45), origin)
        if shape == 'cavity':
            inner.topology.flipOrientation()
        mesh.addMesh(inner)
    elif shape == 'axis_aligned':
        mesh = mm.makeCube(mm.Vector3f(.625, .75, .875), mm.Vector3f(-.25, .125, -.5))
        step = .0625

    plan = winding.hole_field_boxes(mesh, mm, step)
    boxes = plan[0] if plan else np.empty((0, 2, 3))
    params = mm.OffsetParameters()
    params.voxelSize = step
    params.closeHolesInHoleWindingNumber = True
    params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
    kernel = mm.FastWindingNumber(mesh)
    events = []
    grid = mm.calcOriginAndDimensions(mesh.computeBoundingBox(), step)
    signs = np.empty(tuple(reversed(list(grid.dimensions))), bool)
    reference_signs = np.empty_like(signs)
    candidate = scan_field.scan_surface(mesh, mm, kernel, params, boxes, events.append, {'cuda_disabled': True},
                                        budget=3200, probe_signs=signs)
    full = winding.winding_slabs(mesh, mm, kernel, params, lambda _: None, budget=3200, probe_signs=reference_signs)
    checked = winding.compare_probe_surfaces(full, candidate, mm, step, step)
    assert checked['probe_matched'] and checked['probe_surface_distance_mm'] < 2e-6
    np.testing.assert_array_equal(signs, reference_signs)
    end = events[-1]
    assert end['completed_blocks'] > 1 and end['completed_layers'] == grid.dimensions.z
    assert end['projected_samples'] < end['unique_grid_samples'] == signs.size
    assert engine.valid(engine.diagnostics(candidate, mm))
    assert candidate.volume() == pytest.approx(full.volume(), rel=1e-6)


def test_scan_uses_only_small_rims_and_bounds_ambiguous_probe_regions():
    cube = mm.makeCube()
    cube.topology.deleteFace(mm.FaceId(0))
    assert winding.hole_field_boxes(cube, mm, .02) is None
    boxes, _ = winding.hole_field_boxes(punctured_sphere(), mm, .02)
    grid = mm.calcOriginAndDimensions(cube.computeBoundingBox(), .1)
    signs = np.zeros((5, 5, 5), bool)
    changed = signs.copy()
    changed[1, 2, 3] = True
    guards = scan_field.guard_probe_disagreements(signs, changed, grid, .1, boxes)
    np.testing.assert_array_equal(guards[:-1], boxes)
    p = np.array(list(grid.origin)) + .1*np.array([3.5, 2.5, 1.5])
    assert (p > guards[-1, 0]).all() and (p < guards[-1, 1]).all()
    assert scan_field.guard_probe_disagreements(signs, ~signs, grid, .1, boxes) is None


@pytest.mark.parametrize('failure', ['none', 'fine_scan', 'invalid_surface', 'reference_probe'])
def test_optimized_route_and_fallback_keep_the_requested_step(monkeypatch, failure):
    mesh = punctured_sphere()
    q = .035
    monkeypatch.setattr(engine, 'DENSE_WINDING_VOXEL_LIMIT', 1)
    monkeypatch.setattr(engine, 'TIMING_PROBE_MIN_CELLS', 1)
    native_scan = scan_field.scan_surface
    native_slabs = winding.winding_slabs

    def scan(mesh, sdk, kernel, params, *args, **kw):
        if params.voxelSize == pytest.approx(q):
            if failure == 'fine_scan':
                raise scan_field.ScanUnavailable('test ray did not close')
            if failure == 'invalid_surface':
                return mm.makeCube(mm.Vector3f.diagonal(.1))
        return native_scan(mesh, sdk, kernel, params, *args, **kw)

    def slabs(*args, **kw):
        if failure == 'reference_probe':
            raise RuntimeError('test native probe unavailable')
        return native_slabs(*args, **kw)

    monkeypatch.setattr(scan_field, 'scan_surface', scan)
    monkeypatch.setattr(winding, 'winding_slabs', slabs)
    report = {'warnings': [], 'acceleration': {'cuda_disabled': True}, 'after_hole_filling': engine.topology_info(mesh, mm)}
    out = engine.sparse_repair(mesh, mm, engine.Settings(q, restore_sharp_edges=False, disable_cuda=True),
                               engine.Progress(lambda _: None), report, lambda _: None)
    assert engine.valid(engine.diagnostics(out, mm))
    assert report['voxel_passes'][-1]['voxel_size_mm'] == q and out.volume() > 1.
    plan = report['boundary_band']
    if failure == 'reference_probe':
        assert not plan['probe_matched'] and plan['fallback_reason']
    else:
        assert plan['probe_matched'] and plan['method'] == 'oriented_scan'
        assert plan['accepted'] == (failure == 'none')
    if failure != 'none':
        assert report['voxel_passes'][-1].get('field_evaluation') != 'oriented_scan_with_full_hole_boxes'
