"""Fine-tolerance regressions with native MeshLib geometry and bounded corrections."""
import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn

from mesh_repair_blender.core import engine, reduction
from mesh_repair_blender.core.engine import Settings, Progress, diagnostics, valid
from mesh_repair_blender.core.seams import fill_voxel_seams


def creased_sphere(scale=.1):
    mesh = mm.makeUVSphere(1, 24, 24)
    vertices = mn.getNumpyVerts(mesh).copy()
    vertices[np.argmax(vertices[:, 0])] += [0, .6, 0]
    return mn.meshFromFacesVerts(mn.getNumpyFaces(mesh.topology), vertices*scale)


def test_large_streamed_grid_is_not_rejected_by_dense_cell_count(monkeypatch):
    # Dimensions of the user's report; ~6.4 billion cells at 0.01 mm.
    source = mm.makeCube(mm.Vector3f(21.5, 12.2079, 24.1098))
    calls = []

    def mesher(part, offset, params):
        calls.append((params.voxelSize, params.memoryEfficient, params.fwn))
        return mm.makeCube()

    monkeypatch.setattr(mm, 'mcOffsetMesh', mesher)
    monkeypatch.setattr(mm, 'offsetMesh', lambda *a: pytest.fail('must use streamed backend'))
    report = {'warnings': []}
    result = engine.sparse_repair(source, mm, Settings(.01, restore_sharp_edges=False, disable_cuda=True),
                                 Progress(lambda _: None), report, lambda _: None)
    estimate = report['voxel_estimate']
    assert estimate['cells'] > 6e9 and estimate['storage'] == 'streamed_layers'
    assert estimate['probe_cells'] < 4_100_000
    assert len(calls) == 2 and all(efficient and fwn is None for _, efficient, fwn in calls)
    assert calls[-1][0] == pytest.approx(.01)  # Never silently increase the requested step.
    assert valid(diagnostics(result, mm))


def test_native_grid_index_overflow_is_rejected(monkeypatch):
    source = mm.makeCube(mm.Vector3f(1e8, 1e8, 1e8))
    monkeypatch.setattr(mm, 'mcOffsetMesh', lambda *a: pytest.fail('must reject before allocation'))
    with pytest.raises(RuntimeError, match='диапазон индексов'):
        engine.sparse_repair(source, mm, Settings(.001, restore_sharp_edges=False),
                             Progress(lambda _: None), {'warnings': []}, lambda _: None)


@pytest.mark.parametrize('quality', [.01, .011, .015, .02])
def test_native_streamed_reconstruction_and_export_at_fine_steps(tmp_path, monkeypatch, quality):
    source = mm.makeCube(mm.Vector3f(.3, .3, .3))
    source.topology.deleteFace(mm.FaceId(0))
    source.addMesh(mm.makeCube(mm.Vector3f(.3, .3, .3), mm.Vector3f(.1, .1, .1)))
    monkeypatch.setattr(engine, 'DENSE_WINDING_VOXEL_LIMIT', 1)
    report = {'warnings': []}
    result = engine.sparse_repair(source, mm, Settings(quality, restore_sharp_edges=False, disable_cuda=True),
                                 Progress(lambda _: None), report, lambda _: None)
    assert valid(report['reconstruction_validation'])
    assert report['voxel_passes'][-1]['backend'] == 'marching_cubes_by_parts'
    assert report['voxel_passes'][-1]['voxel_size_mm'] == quality
    path = tmp_path/'result.stl'
    mm.saveMesh(result, path)
    assert valid(diagnostics(mm.loadMesh(path), mm))


@pytest.mark.parametrize('quality', [.01, .011, .015, .02])
def test_fine_tolerance_intersections_fixed_without_changing_connectivity(quality):
    mesh = creased_sphere()
    before = diagnostics(mesh, mm)
    vertices = mn.getNumpyVerts(mesh).copy()
    assert before['self_intersecting_faces'] == 4
    report = {}
    result, after = engine.repair_reconstruction_intersections(mesh, before, mm, Settings(quality),
        Progress(lambda _: None), report, lambda _: None)
    item = report['reconstruction_local_repair']
    assert item['accepted'] and item['method'] == 'bounded_relax'
    assert valid(after) and after['components'] == before['components']
    assert result.topology == mesh.topology
    assert item['verification']['upper_bound_mm'] <= quality*.25
    assert np.array_equal(vertices, mn.getNumpyVerts(mesh))  # Input must survive rejected attempts too.
    assert diagnostics(mesh, mm)['self_intersecting_faces'] == 4


def test_hundreds_of_residual_faces_and_unrelated_small_shell_are_preserved():
    mesh = creased_sphere()
    for index in range(1, 49):
        part = creased_sphere()
        part.transform(mm.AffineXf3f.translation(mm.Vector3f(index*.3, 0, 0)))
        mesh.addMesh(part)
    mesh.addMesh(mm.makeCube(mm.Vector3f(.001, .001, .001), mm.Vector3f(-1, 0, 0)))
    before = diagnostics(mesh, mm)
    assert before['self_intersecting_faces'] == 196 and before['components'] == 50
    report = {}
    result, after = engine.repair_reconstruction_intersections(mesh, before, mm, Settings(.011),
        Progress(lambda _: None), report, lambda _: None)
    assert valid(after) and after['components'] == 50
    assert result.topology == mesh.topology
    assert report['reconstruction_local_repair']['verification']['verified']


def test_correspondence_proof_rejects_changed_topology_and_excessive_movement():
    source = mm.makeCube()
    candidate = mm.Mesh(source)
    candidate.transform(mm.AffineXf3f.translation(mm.Vector3f(.01, 0, 0)))
    assert not reduction.vertex_displacement_bound(source, candidate, .0025)['verified']
    candidate = mm.Mesh(source)
    candidate.topology.deleteFace(mm.FaceId(0))
    proof = reduction.vertex_displacement_bound(source, candidate, .0025)
    assert not proof['verified'] and proof['reason'] == 'topology_changed'


def test_failed_correction_retains_invalid_mesh_and_diagnostics(monkeypatch):
    mesh = creased_sphere()
    before = diagnostics(mesh, mm)
    # Simulate a native fix that leaves all intersections in place.
    monkeypatch.setattr(mm, 'relax', lambda *a: True)
    monkeypatch.setattr(mm.SelfIntersections, 'fix', lambda *a: None)
    report = {}
    result, after = engine.repair_reconstruction_intersections(mesh, before, mm, Settings(.011),
        Progress(lambda _: None), report, lambda _: None)
    assert result is mesh and after is before
    assert not report['reconstruction_local_repair']['accepted']
    assert len(report['reconstruction_local_repair']['attempts']) == 4


def test_repair_exception_carries_partial_report(tmp_path, monkeypatch):
    source = tmp_path/'damaged.stl'
    mm.saveMesh(creased_sphere(), source)

    def fail_reconstruction(mesh, mm, settings, tracker, report, emit, **kwargs):
        report['algorithm'] = 'sparse_openvdb_winding'
        report['reconstruction_validation'] = diagnostics(mesh, mm)
        report['reconstruction_local_repair'] = {'accepted': False, 'reason': 'test_rejection'}
        return mesh

    monkeypatch.setattr(engine, 'sparse_repair', fail_reconstruction)
    with pytest.raises(RuntimeError, match='Результат не прошёл проверку') as failure:
        engine.run(source, settings=Settings(.011, local_fix_max_faces=0, local_fix_fraction=0),
                   work_dir=tmp_path/'job')
    report = failure.value.report
    assert report['status'] == 'error' and report['settings']['quality_mm'] == .011
    assert report['reconstruction_validation']['self_intersecting_faces'] > 0
    assert report['reconstruction_local_repair']['reason'] == 'test_rejection'
    assert 'artifacts' not in report


def test_tiny_voxel_crack_closes_without_moving_old_vertices():
    mesh = mm.makeCube(mm.Vector3f(.01, .000001, .01), mm.Vector3f(0, 0, 0))
    mesh.topology.deleteFace(mm.FaceId(0))
    before = mn.getNumpyVerts(mesh).copy()
    result, report = fill_voxel_seams(mesh, mm, .0025)
    assert report['accepted'] and report['verification']['upper_bound_mm'] <= .0025
    assert result.topology.isClosed() and not mesh.topology.isClosed()
    assert np.array_equal(before, mn.getNumpyVerts(result)[:len(before)])
    assert valid(diagnostics(result, mm))


def test_large_hole_is_not_mistaken_for_a_voxel_crack():
    mesh = mm.makeCube()
    mesh.topology.deleteFace(mm.FaceId(0))
    result, report = fill_voxel_seams(mesh, mm, .0025)
    assert result is mesh and not report['accepted']
    assert report['reason'] == 'seam_exceeds_tolerance'


@pytest.mark.parametrize('retry_succeeds', [True, False])
def test_accuracy_retry_is_bounded_and_keeps_requested_voxel_size(monkeypatch, retry_succeeds):
    damaged = creased_sphere()
    calls = []

    def reconstruct(part, offset, params):
        calls.append((params.voxelSize, params.windingNumberBeta))
        return mm.makeCube() if len(calls) > 1 and retry_succeeds else mm.Mesh(damaged)

    monkeypatch.setattr(mm, 'offsetMesh', reconstruct)
    monkeypatch.setattr(engine, 'repair_reconstruction_intersections', lambda mesh, info, *a: (mesh, info))
    report = {'warnings': []}
    engine.sparse_repair(mm.makeCube(mm.Vector3f(.1, .1, .1)), mm,
        Settings(.011, restore_sharp_edges=False, disable_cuda=True), Progress(lambda _: None), report, lambda _: None)
    np.testing.assert_allclose(calls, [(.011, 2.), (.011, 4.)])
    assert report['accurate_winding_retry']['accepted'] == retry_succeeds
    assert valid(report['reconstruction_validation']) == retry_succeeds
