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
    monkeypatch.setattr(mm, 'fixMeshDegeneracies', lambda *a: None)
    monkeypatch.setattr(mm.SelfIntersections, 'fix', lambda *a: None)
    report = {}
    result, after = engine.repair_reconstruction_intersections(mesh, before, mm, Settings(.011),
        Progress(lambda _: None), report, lambda _: None)
    assert result is mesh and after is before
    assert not report['reconstruction_local_repair']['accepted']
    assert len(report['reconstruction_local_repair']['attempts']) == 8


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
        Settings(.011, restore_sharp_edges=False, disable_cuda=True, accurate_winding_max_seconds=180),
        Progress(lambda _: None), report, lambda _: None)
    np.testing.assert_allclose(calls, [(.011, 2.), (.011, 4.)])
    assert report['accurate_winding_retry']['accepted'] == retry_succeeds
    assert valid(report['reconstruction_validation']) == retry_succeeds


def test_accuracy_retry_cancels_native_work_at_deadline(monkeypatch):
    clock = [0.]
    calls = []
    damaged = creased_sphere()
    monkeypatch.setattr(engine.time, 'monotonic', lambda: clock[0])

    def reconstruct(part, offset, params):
        calls.append(params.windingNumberBeta)
        if params.windingNumberBeta > 2:
            clock[0] = 181.
            assert not params.callBack(.5)
            raise RuntimeError('Operation canceled')
        return mm.Mesh(damaged)

    monkeypatch.setattr(mm, 'offsetMesh', reconstruct)
    monkeypatch.setattr(engine, 'repair_reconstruction_intersections', lambda mesh, info, *a: (mesh, info))
    report = {'warnings': []}
    with pytest.raises(TimeoutError, match='лимиту времени'):
        engine.sparse_repair(mm.makeCube(mm.Vector3f(.1, .1, .1)), mm,
            Settings(.011, restore_sharp_edges=False, disable_cuda=True, accurate_winding_max_seconds=180),
            Progress(lambda _: None),
            report, lambda _: None)
    assert calls == [2., 4.]
    assert report['accurate_winding_retry']['reason'] == 'winding_retry_time_limit'
    assert not report['accurate_winding_retry']['accepted']
    assert report['reconstruction_validation']['self_intersecting_faces'] == 4


@pytest.mark.parametrize('limit', [-1, float('nan'), float('inf')])
def test_retry_time_limit_must_be_finite_and_nonnegative(limit):
    with pytest.raises(ValueError):
        Settings(accurate_winding_max_seconds=limit).validate()


def test_failed_local_repair_does_not_restart_the_full_grid_by_default(monkeypatch):
    damaged = creased_sphere()
    calls = []
    monkeypatch.setattr(mm, 'offsetMesh', lambda *a: calls.append(True) or mm.Mesh(damaged))
    monkeypatch.setattr(engine, 'repair_reconstruction_intersections', lambda mesh, info, *a: (mesh, info))
    report = {'warnings': []}
    engine.sparse_repair(mm.makeCube(mm.Vector3f(.1, .1, .1)), mm,
        Settings(.015, restore_sharp_edges=False, disable_cuda=True), Progress(lambda _: None), report, lambda _: None)
    assert len(calls) == 1
    assert not valid(report['reconstruction_validation'])


def test_partial_improvements_are_retained_until_the_last_collision_is_fixed():
    base = mm.makeUVSphere(.08, 24, 24)
    faces, original = mn.getNumpyFaces(base.topology), mn.getNumpyVerts(base)
    rng = np.random.default_rng(7)
    for _ in range(3):
        vertices = original.copy()
        indices = rng.choice(len(vertices), 3, replace=False)
        vertices[indices] += rng.normal(0, .007, size=(3, 3))
    mesh = mn.meshFromFacesVerts(faces, vertices)
    report = {'acceleration': {'cuda_disabled': True}}
    before = diagnostics(mesh, mm)
    result, after = engine.repair_reconstruction_intersections(mesh, before, mm, Settings(.011),
        Progress(lambda _: None), report, lambda _: None)
    item = report['reconstruction_local_repair']
    retained = [a for a in item['attempts'] if a.get('retained')]
    assert len(retained) >= 2 and valid(after)
    assert retained[0]['after']['self_intersecting_faces'] > 0
    for attempt in retained:
        assert attempt['after']['self_intersecting_faces'] < attempt['input_intersecting_faces']
        assert attempt['verification']['verified']
        assert attempt['verification']['upper_bound_mm'] <= .011*.5
    assert item['verification']['unchanged_faces_skipped'] > 0
    # Independent full-surface check of the final patch proof on this fixture.
    assert reduction.verify_surface_bound(mesh, result, .011*.5,
                                          acceleration={'cuda_disabled': True})['verified']


def test_local_bound_includes_every_face_incident_to_a_moved_vertex():
    source = mm.makeUVSphere(1, 24, 24)
    candidate = mm.Mesh(source)
    candidate.points[mm.VertId(100)] += mm.Vector3f(.0001, 0, 0)
    candidate.invalidateCaches()
    proof = reduction.verify_local_surface_bound(source, candidate, .001,
                                                 acceleration={'cuda_disabled': True})
    assert proof['verified'] and proof['upper_bound_mm'] <= .001
    assert 0 < proof['patch_faces'][0] < source.topology.numValidFaces()
    assert proof['unchanged_faces_skipped'] > 0


def test_artifacts_are_removed_before_they_can_block_outer_surface_repair():
    mesh = mm.makeUVSphere(2, 24, 24)
    for offset in (0, .25):
        cavity = mm.makeCube(mm.Vector3f(.5, .5, .5), mm.Vector3f(offset, offset*.7, offset*.4))
        cavity.topology.flipOrientation()
        mesh.addMesh(cavity)
    # A legitimate positive tiny body stays when small-shell removal is off.
    mesh.addMesh(mm.makeCube(mm.Vector3f(.01, .01, .01), mm.Vector3f(4, 0, 0)))
    before = diagnostics(mesh, mm)
    assert before['self_intersecting_faces'] > 0
    report = {}
    result, after = engine.clean_reconstruction_artifacts(mesh, before, mm, Settings(.015), report, lambda _: None)
    assert valid(after) and after['components'] == 2
    assert report['reconstruction_artifacts']['accepted']
    assert len(report['reconstruction_artifacts']['removed']) == 2
    assert diagnostics(mesh, mm)['components'] == 4
    assert result is not mesh


def test_artifact_cleanup_does_not_assume_remaining_mesh_has_no_intersections():
    mesh = mm.makeUVSphere(1, 24, 24)
    other = mm.makeUVSphere(1, 24, 24)
    other.transform(mm.AffineXf3f.translation(mm.Vector3f(1, 0, 0)))
    mesh.addMesh(other)
    tiny = mm.makeCube(mm.Vector3f(.01, .01, .01), mm.Vector3f(4, 0, 0))
    mesh.addMesh(tiny)
    report = {}
    _, after = engine.clean_reconstruction_artifacts(mesh, diagnostics(mesh, mm), mm,
        Settings(.015, remove_small_shells=True), report, lambda _: None)
    assert after['self_intersecting_faces'] > 0 and not valid(after)
    assert after['components'] == 2
