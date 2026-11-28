"""Exercise the large-mesh decisions at small scale as well as the Windows Blender fixture."""
import pytest
import numpy as np
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn

from mesh_repair_blender.core import engine, reduction
from mesh_repair_blender.core.engine import Progress, Settings, diagnostics, valid


def cube(edge=None):
    mesh = mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(0, 0, 0))
    if edge:
        params = mm.SubdivideSettings()
        params.maxEdgeLen, params.maxAngleChangeAfterFlip = edge, 0
        mm.subdivideMesh(mesh, params)
    return mesh


def test_large_valid_surface_does_not_need_voxels(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'DENSE_MESH_FACES', 10)
    path = tmp_path/'closed.stl'
    mm.saveMesh(cube(.2), path)
    report = engine.run(path, settings=Settings(.02, restore_sharp_edges=False), work_dir=tmp_path/'work',
                        array_output=(tmp_path/'v.npy', tmp_path/'f.npy'))
    assert not report['dense_routing']['needs_voxels'] and 'voxel_passes' not in report
    assert report['after_hole_filling']['self_intersecting_faces'] == 0
    assert report['before']['self_intersecting_faces'] is None
    assert valid(report['after'])


def test_large_open_surface_goes_directly_to_winding(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'DENSE_MESH_FACES', 10)
    mesh = cube()
    mesh.topology.deleteFace(mm.FaceId(0))
    path = tmp_path/'open.stl'
    mm.saveMesh(mesh, path)
    from mesh_repair_blender.core import holes
    fill = holes.fill_simple_holes
    monkeypatch.setattr(holes, 'fill_simple_holes', lambda *a, **kw: fill(*a, **kw, max_patch_faces=0))
    report = engine.run(path, settings=Settings(.08, restore_sharp_edges=False, disable_cuda=True),
                        work_dir=tmp_path/'work', array_output=(tmp_path/'v.npy', tmp_path/'f.npy'))
    assert report['dense_routing']['needs_voxels'] and not report['local_repair']['attempted']
    assert report['after_hole_filling']['boundary_loops'] and report['after_hole_filling']['self_intersecting_faces'] is None
    assert report['algorithm'] == 'sparse_openvdb_winding' and valid(report['after'])
    assert all(p['voxel_size_mm'] == .08 for p in report['voxel_passes'])


def test_misplaced_closed_level_set_falls_back_at_same_step(monkeypatch):
    monkeypatch.setattr(engine, 'DENSE_MESH_FACES', 5)
    mesh = cube(.3)
    native = mm.doubleOffsetMesh

    def shifted(part, outward, inward, params):
        result = native(part, outward, inward, params)
        if params.signDetectionMode == mm.SignDetectionMode.OpenVDB:
            result.transform(mm.AffineXf3f.translation(mm.Vector3f(1, 0, 0)))
        return result
    monkeypatch.setattr(mm, 'doubleOffsetMesh', shifted)
    report = {'warnings': [], 'after_hole_filling': {'self_intersecting_faces': 2}}
    translation = mm.Vector3f(3, 4, 5)
    result = engine.sparse_repair(mesh, mm, Settings(.08, restore_sharp_edges=False, disable_cuda=True),
                                  Progress(lambda _: None), report, lambda _: None, translation)
    direct, fallback = report['voxel_passes']
    assert direct['reason'] == 'level_set_distance_exceeds_step' and not direct['accepted']
    assert fallback['voxel_size_mm'] == direct['voxel_size_mm'] == .08
    assert valid(diagnostics(result, mm))
    assert result.computeBoundingBox().min.x == pytest.approx(3, abs=.1)


@pytest.mark.parametrize('strict', [False, True])
def test_dense_refinement_is_checked_against_original_and_keeps_seed(monkeypatch, strict):
    monkeypatch.setattr(engine, 'DENSE_MESH_FACES', 10)
    original, seed, smaller = cube(.1), cube(.25), cube()
    calls = []
    def propose(mesh, *args, **options):
        calls.append(mesh.topology.numValidFaces())
        assert options['single_pass']
        return (seed if len(calls) == 1 else smaller), [{}]
    monkeypatch.setattr(reduction, 'propose_reduction', propose)
    if strict:
        def proof(ref, target, *args, **kwargs):
            assert ref is original
            return {'verified': target is seed, 'upper_bound_mm': 0.0}
        monkeypatch.setattr(reduction, 'verify_surface_bound', proof)
    else:
        smaller.transform(mm.AffineXf3f.translation(mm.Vector3f(.05, 0, 0)))
    report = {'after': diagnostics(original, mm), 'steps': [], 'warnings': [], 'acceleration': {}}
    result = engine.simplify_attempts(original, None, mm, Settings(.02, simplify_curved=True, strict_tolerance=strict),
                                      Progress(lambda _: None), report, lambda _: None, None)
    assert result is seed
    assert calls[:2] == [original.topology.numValidFaces(), seed.topology.numValidFaces()]
    assert calls.count(original.topology.numValidFaces()) == 1
    assert report['simplification']['accepted_attempt'] == 1
    assert report['simplification']['verification']['verified'] is strict


def test_rejected_dense_seeds_limit_full_mesh_visits(monkeypatch):
    monkeypatch.setattr(engine, 'DENSE_MESH_FACES', 10)
    original, broken = cube(.1), cube()
    broken.topology.deleteFace(mm.FaceId(0))
    monkeypatch.setattr(reduction, 'propose_reduction', lambda *a, **kw: (broken, [{}]))
    report = {'after': diagnostics(original, mm), 'steps': [], 'warnings': []}
    result = engine.simplify_attempts(original, None, mm, Settings(.02, simplify_curved=True),
                                      Progress(lambda _: None), report, lambda _: None, None)
    assert result is original
    assert [i['phase'] for i in report['simplification']['attempts']] == ['dense_seed', 'conservative_seed']
    assert not report['simplification']['accepted']


def test_closed_band_rejects_artificial_tunnels(monkeypatch):
    monkeypatch.setattr(mm, 'doubleOffsetMesh', lambda *args: mm.makeTorus(1, .2, 24, 12))
    monkeypatch.setattr(mm, 'findSelfCollidingTrianglesBS', lambda *args: pytest.fail('already rejected on genus'))
    report = {'voxel_passes': []}
    result = engine.try_closed_level_set(cube(), mm, Settings(.02), Progress(lambda _: None),
                                         report, lambda _: None, None)
    assert result is None and report['voxel_passes'][0]['reason'] == 'level_set_genus_changed'


def test_dense_refinement_cannot_spend_the_voxel_error_twice(monkeypatch):
    monkeypatch.setattr(engine, 'DENSE_MESH_FACES', 10)
    source, original, seed, shifted = cube(), cube(.1), cube(.25), cube()
    source.transform(mm.AffineXf3f.translation(mm.Vector3f(-.012, 0, 0)))
    shifted.transform(mm.AffineXf3f.translation(mm.Vector3f(.012, 0, 0)))

    def propose(base, *args, **options):
        return (seed if base is original else shifted), [{}]

    monkeypatch.setattr(reduction, 'propose_reduction', propose)
    report = {'after': diagnostics(original, mm), 'steps': [], 'warnings': []}
    result = engine.simplify_attempts(original, None, mm, Settings(.02, simplify_curved=True),
                                      Progress(lambda _: None), report, lambda _: None, None, source)
    assert result is seed
    compact = report['simplification']['attempts'][1]
    assert compact['verification']['max_deviation_mm'] < .019
    assert compact['reason'] == 'level_set_distance_exceeds_step'
    assert not compact['level_set_source_check']['within_step']


def test_dense_voxel_estimate_can_select_the_band_for_a_small_input(monkeypatch):
    monkeypatch.setattr(engine, 'DENSE_MESH_FACES', 1500)
    sphere = mm.makeUVSphere(1, 24, 24)
    verts = mn.getNumpyVerts(sphere)
    verts[np.argmax(verts[:, 0])] += [0, .6, 0]
    mesh = mn.meshFromFacesVerts(mn.getNumpyFaces(sphere.topology), verts)
    report = {'warnings': []}
    engine.sparse_repair(mesh, mm, Settings(.08, restore_sharp_edges=False, disable_cuda=True),
                         Progress(lambda _: None), report, lambda _: None)
    assert mesh.topology.numValidFaces() < 1500
    assert report['voxel_passes'][0]['backend'] == 'openvdb_closed_level_set'
    assert valid(report['reconstruction_validation'])


def test_inward_open_body_is_reoriented_after_closing(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'DENSE_MESH_FACES', 10)
    mesh = cube()
    mesh.topology.flipOrientation()
    mesh.topology.deleteFace(mm.FaceId(0))
    path = tmp_path/'inside-out.stl'
    mm.saveMesh(mesh, path)
    report = engine.run(path, settings=Settings(.02, restore_sharp_edges=False), work_dir=tmp_path/'job',
                        array_output=(tmp_path/'v.npy', tmp_path/'f.npy'))
    assert report['hole_filling']['closed_orientation_corrections'] == 1
    assert report['algorithm'] == 'surface_repair' and 'voxel_passes' not in report
    assert report['after']['signed_volume_mm3'] == pytest.approx(1)


def test_dense_seed_near_budget_stops_after_measured_acceptance(monkeypatch):
    monkeypatch.setattr(engine, 'DENSE_MESH_FACES', 10)
    original, seed = cube(.1), cube()
    monkeypatch.setattr(reduction, 'propose_reduction', lambda *a, **kw: (seed, [{}]))
    monkeypatch.setattr(reduction, 'measured_distance', lambda *args: .018)
    report = {'after': diagnostics(original, mm), 'steps': [], 'warnings': []}
    result = engine.simplify_attempts(original, None, mm, Settings(.02, simplify_curved=True),
                                      Progress(lambda _: None), report, lambda _: None, None)
    assert result is seed
    summary = report['simplification']
    assert len(summary['attempts']) == 1
    assert summary['search_stop'] == 'dense_seed_near_budget'


def test_many_existing_intersections_use_safe_caps_once(tmp_path, monkeypatch):
    from mesh_repair_blender.core import holes
    source = cube()
    source.addMesh(mm.makeCube(mm.Vector3f.diagonal(.8), mm.Vector3f.diagonal(.45)))
    source.topology.deleteFace(mm.FaceId(0))
    src = tmp_path/'source.stl'
    mm.saveMesh(source, src)
    calls = []
    fill = holes.fill_simple_holes

    def tracked(*args, **kwargs):
        calls.append(kwargs.get('defer_intersections', False))
        return fill(*args, **kwargs)

    monkeypatch.setattr(holes, 'fill_simple_holes', tracked)
    settings = Settings(.07, restore_sharp_edges=False, disable_cuda=True, local_fix_max_faces=1, local_fix_fraction=0)
    report = engine.run(src, settings=settings, work_dir=tmp_path/'job',
                        array_output=(tmp_path/'v.npy', tmp_path/'f.npy'))
    assert calls == [False]
    assert report['prepared']['self_intersecting_faces'] > 1
    assert valid(report['after'])
