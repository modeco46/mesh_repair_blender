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


def test_multibody_band_keeps_both_bodies(monkeypatch):
    mesh = punctured_sphere()
    second = punctured_sphere()
    second.transform(mm.AffineXf3f.translation(mm.Vector3f(1.8, 0, 0)))
    mesh.addMesh(second)
    monkeypatch.setattr(engine, 'DENSE_WINDING_VOXEL_LIMIT', 1)
    monkeypatch.setattr(engine, 'TIMING_PROBE_MIN_CELLS', 1)
    report = {'warnings': [], 'acceleration': {'cuda_disabled': True}, 'after_hole_filling': engine.topology_info(mesh, mm)}
    q = .04
    out = engine.sparse_repair(mesh, mm, engine.Settings(q, restore_sharp_edges=False, disable_cuda=True),
                               engine.Progress(lambda _: None), report, lambda _: None)
    assert report['boundary_band']['accepted']
    after = engine.diagnostics(out, mm)
    assert engine.valid(after) and after['components'] == 2
    params = mm.OffsetParameters()
    params.voxelSize = q
    params.closeHolesInHoleWindingNumber = True
    direct = winding.winding_slabs(mesh, mm, mm.FastWindingNumber(mesh), params, lambda _: None)
    assert after['triangles'] == direct.topology.numValidFaces()
    assert mm.findMaxDistanceSq(mm.MeshPart(out), mm.MeshPart(direct)) < 1e-10


@pytest.mark.parametrize('retry_ok', [True, False])
def test_band_microcrack_retries_once_in_input_coordinates(monkeypatch, retry_ok):
    from mesh_repair_blender.core import seams
    mesh = punctured_sphere()
    q = .035
    translation = mm.Vector3f(13, -41, 7)
    monkeypatch.setattr(engine, 'DENSE_WINDING_VOXEL_LIMIT', 1)
    monkeypatch.setattr(engine, 'TIMING_PROBE_MIN_CELLS', 1)
    monkeypatch.setattr(seams, 'fill_voxel_seams', lambda surface, *args: (surface, {'accepted': False}))

    def no_scan(*args, **kwargs):
        raise scan_field.ScanUnavailable('test native band')

    monkeypatch.setattr(scan_field, 'scan_surface', no_scan)
    native = winding.winding_slabs
    fine_boxes = []
    seam_centers = []

    def slabs(mesh, sdk, kernel, params, completed, **kw):
        surface = native(mesh, sdk, kernel, params, completed, **kw)
        boxes = kw.get('boundary_boxes')
        if params.voxelSize == pytest.approx(q) and boxes is not None:
            fine_boxes.append(boxes.copy())
            if len(fine_boxes) == 1 or not retry_ok:
                face = mm.FaceId(surface.topology.faceSize()//2)
                seam_centers.append(list(surface.triCenter(face)))
                surface.topology.deleteFace(face)
        return surface

    monkeypatch.setattr(winding, 'winding_slabs', slabs)
    report = {'warnings': [], 'acceleration': {'cuda_disabled': True}, 'after_hole_filling': engine.topology_info(mesh, mm)}
    out = engine.sparse_repair(mesh, mm, engine.Settings(q, restore_sharp_edges=False, disable_cuda=True),
                               engine.Progress(lambda _: None), report, lambda _: None, translation)
    plan = report['boundary_band']
    assert len(fine_boxes) == 2 and plan['seam_guard_boxes'] == 1
    lo, hi = fine_boxes[-1][-1]
    assert np.all(lo < seam_centers[0]) and np.all(hi > seam_centers[0])
    assert plan['accepted'] is retry_ok
    assert len(plan['rejected_passes']) == (1 if retry_ok else 2)
    assert abs(out.computeBoundingBox().center().x-translation.x) < .05
    assert engine.valid(engine.diagnostics(out, mm))
    if not retry_ok:
        assert report['voxel_passes'][-1].get('field_evaluation') != 'surface_band_with_full_hole_boxes'


def test_separated_hole_guards_bound_total_solid_angle():
    # Many holes on separate pieces. The common area bound would cover more
    # than the allowed volume; each fan is small relative to its neighbours.
    mesh = mm.Mesh()
    for i in range(5):
        for j in range(4):
            for k in range(4):
                part = mm.makeCube(mm.Vector3f.diagonal(.1), mm.Vector3f(i, j, k))
                part.topology.deleteFace(mm.FaceId(0))
                mesh.addMesh(part)
    q = .01
    boxes, info = winding.hole_field_boxes(mesh, mm, q)
    assert len(boxes) == 80 and info['guard_bound'] == 'nearest_fan_separation'
    assert info['box_volume_fraction_upper_bound'] < .15
    raw = []
    areas = []
    for edge in mesh.topology.findHoleRepresentiveEdges():
        loop = mm.trackRightBoundaryLoop(mesh.topology, edge)
        pts = np.array([list(mesh.points[mesh.topology.org(e)]) for e in loop])
        raw.append((pts.min(axis=0), pts.max(axis=0)))
        areas.append(.5*np.linalg.norm(np.cross(pts[1]-pts[0], pts[2]-pts[0])))
    raw = np.asarray(raw)
    for p in np.random.default_rng(5).uniform(-.2, 4.5, (250, 3)):
        if np.any(((p >= boxes[:, 0]) & (p <= boxes[:, 1])).all(axis=1)):
            continue
        gap = np.maximum(0, np.maximum(raw[:, 0]-p, p-raw[:, 1]))
        assert np.sum(areas/np.einsum('ij,ij->i', gap, gap)) <= 4*np.pi*.125
