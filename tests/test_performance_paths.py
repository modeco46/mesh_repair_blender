"""Native geometry regressions for the experimental surface-first paths."""
from types import SimpleNamespace

import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn

from mesh_repair_blender.core import components, engine, orientation, reduction, surface_paths
from mesh_repair_blender.core.engine import Settings, Progress, diagnostics, valid
from mesh_repair_blender.core.deviation import source_deviation
from mesh_repair_blender.core.holes import fill_simple_holes


def dented_sphere():
    mesh = mm.makeUVSphere(.1, 24, 24)
    vertices = mn.getNumpyVerts(mesh)
    vertices[np.argmax(vertices[:, 0])] += [0, .06, 0]
    return mn.meshFromFacesVerts(mn.getNumpyFaces(mesh.topology), vertices)


def report_for(mesh):
    return {'after': diagnostics(mesh, mm), 'steps': [], 'warnings': [],
            'acceleration': {'cuda_disabled': True}}


def test_bulk_labels_match_native_iterator_and_own_their_memory():
    mesh = mm.makeCube()
    mesh.addMesh(mm.makeUVSphere(1, 48, 48))
    mesh.topology.deleteFace(mm.FaceId(3))  # Invalid slots must survive too.
    mapping, _ = mm.MeshComponents.getAllComponentsMap(mm.MeshPart(mesh))
    expected = np.fromiter((int(x) for x in mapping), dtype=np.int32, count=mapping.size())
    actual = components.region_labels(mapping)
    mapping.clear()
    assert actual.flags.owndata and np.array_equal(actual, expected)


def test_unknown_component_layout_falls_back_without_reading_pointer():
    class Mapping:
        vec_ = SimpleNamespace(element_type_byte_size=8,
            data_pointer=lambda: pytest.fail('unsupported element size must not be dereferenced'))

        def size(self):
            return 3

        def __iter__(self):
            return iter([0, 1, -1])
    assert components.region_labels(Mapping()).tolist() == [0, 1, -1]


def test_noop_weld_skips_reorientation(monkeypatch):
    calls = []
    native = orientation.orient_consistently

    def orient(*args):
        calls.append(True)
        return native(*args)

    monkeypatch.setattr(orientation, 'orient_consistently', orient)
    result, report = orientation.prepare_orientation(mm.makeCube(), mm, weld_tolerance=.002)
    assert len(calls) == 1 and report['approximate_weld_united_vertices'] == 0
    assert report['collision_geometry_unchanged']
    assert report['components'] == diagnostics(result, mm)['components']
    assert 'orientation_regions' in report


def test_real_weld_reorients_and_invalidates_collision_reuse():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [.00001, 0, 0], [0, 1.00001, 0], [-1, 0, 0]], np.float32)
    mesh = mn.meshFromFacesVerts(np.array([[0, 1, 2], [3, 4, 5]], np.int32), vertices)
    _, report = orientation.prepare_orientation(mesh, mm, weld_tolerance=.002)
    assert report['approximate_weld_united_vertices'] > 0
    assert 'approximate_weld_followup' in report and not report['collision_geometry_unchanged']


def test_closed_mesh_hole_pass_reuses_geometry():
    mesh = mm.makeCube()
    result, report = fill_simple_holes(mesh, mm, .02)
    assert result is mesh and report['holes_before'] == report['filled'] == 0


@pytest.mark.parametrize('hole', [False, True])
def test_collision_reuse_only_when_geometry_did_not_change(tmp_path, hole):
    mesh = mm.makeCube()
    if hole:
        mesh.topology.deleteFace(mm.FaceId(0))
    path = tmp_path/'source.stl'
    mm.saveMesh(mesh, path)
    report = engine.run(path, settings=Settings(.02, disable_cuda=True, restore_sharp_edges=False),
                        work_dir=tmp_path/'job')
    assert valid(report['after'])
    assert report['post_hole_diagnostics'] == ('recomputed_after_geometry_change' if hole
                                              else 'reused_unchanged_source_collisions')


def test_internal_shell_repair_enables_union_without_voxels(tmp_path):
    source = dented_sphere()
    second = mm.makeUVSphere(.1, 24, 24)
    second.transform(mm.AffineXf3f.translation(mm.Vector3f(.04, 0, 0)))
    source.addMesh(second)
    assert diagnostics(source, mm)['self_intersecting_faces'] > 200
    path = tmp_path/'source.stl'
    mm.saveMesh(source, path)
    report = engine.run(path, settings=Settings(.02, disable_cuda=True, restore_sharp_edges=False),
                        work_dir=tmp_path/'job')
    assert report['algorithm'] == 'solid_union_after_local_shell_repair'
    assert report['union_attempt']['faces_inside_shell'] == 4
    assert report['shell_local_repair']['accepted']
    assert 'voxel_passes' not in report and valid(report['after'])
    assert diagnostics(mm.loadMesh(path), mm)['self_intersecting_faces'] > 200


def test_shell_threshold_counts_faces_not_pairs_and_keeps_original():
    mesh = dented_sphere()
    before = mn.getNumpyVerts(mesh).copy()
    report = {}
    result = surface_paths.repair_shells_before_union(mesh, diagnostics(mesh, mm), {0}, 4, mm,
        Settings(.02, local_fix_max_faces=3, local_fix_fraction=0), Progress(lambda _: None), report, lambda _: None)
    assert result is None and not report['shell_local_repair']['attempted']
    assert np.array_equal(before, mn.getNumpyVerts(mesh))


@pytest.mark.parametrize('quality,accepted', [(.02, True), (.011, False)])
def test_decimated_surface_rescue_checks_total_error(quality, accepted):
    original = mm.makeUVSphere(.1, 24, 24)
    candidate = dented_sphere()
    report = {'acceleration': {'cuda_disabled': True}}
    result, after = surface_paths.rescue_decimated_surface(original, candidate, diagnostics(candidate, mm),
        mm, Settings(quality), Progress(lambda _: None), report, lambda _: None)
    assert (result is not candidate) == accepted
    assert valid(after) == accepted
    if not accepted:
        assert report['repair']['reason'] == 'repair_exceeds_total_simplification_tolerance'
        assert report['total_deviation_mm'] > quality*.95
    else:
        assert reduction.verify_surface_bound(original, result, quality,
                                              acceleration={'cuda_disabled': True})['verified']


def test_simplification_checkpoint_keeps_repaired_candidate(monkeypatch):
    reference = mm.makeUVSphere(.1, 24, 24)
    subdivision = mm.SubdivideSettings()
    subdivision.maxEdgeLen = .01
    mm.subdivideMesh(reference, subdivision)
    candidate = dented_sphere()
    report = report_for(reference)
    received = []

    def propose(mesh, *args, checkpoint=None, **kwargs):
        stages = [{'pass': 1, 'before': mesh.topology.numValidFaces(),
                   'after': candidate.topology.numValidFaces()}]
        response = checkpoint(mm.Mesh(candidate), stages)
        if isinstance(response, mm.Mesh):
            assert valid(diagnostics(response, mm))
            received.append(response)
        return mm.Mesh(candidate), stages

    monkeypatch.setattr(reduction, 'propose_reduction', propose)
    result = engine.simplify_attempts(reference, None, mm, Settings(.02, simplify_curved=True),
        Progress(lambda _: None), report, lambda _: None, None)
    assert received and valid(diagnostics(result, mm))
    assert report['simplification']['accepted']
    assert report['simplification']['attempts'][0]['checkpoints'][0]['local_repair']['repair']['accepted']


def test_refinements_use_reduced_mesh_but_compare_to_original(monkeypatch):
    reference = mm.makeCube()
    params = mm.SubdivideSettings()
    params.maxEdgeLen = .1
    params.maxAngleChangeAfterFlip = 0
    mm.subdivideMesh(reference, params)
    first = mm.makeCube()
    first.transform(mm.AffineXf3f.translation(mm.Vector3f(.01, 0, 0)))
    over_budget = mm.makeCube()
    over_budget.transform(mm.AffineXf3f.translation(mm.Vector3f(.025, 0, 0)))
    inputs = []

    def propose(mesh, *args, checkpoint=None, **kwargs):
        inputs.append(mesh.topology.numValidFaces())
        candidate = first if len(inputs) == 1 else over_budget
        stages = [{'pass': 1, 'before': inputs[-1], 'after': 12}]
        checkpoint(candidate, stages)
        return candidate, stages

    monkeypatch.setattr(reduction, 'propose_reduction', propose)
    report = report_for(reference)
    result = engine.simplify_attempts(reference, None, mm, Settings(.02, simplify_curved=True),
        Progress(lambda _: None), report, lambda _: None, None)
    assert inputs[0] > 12 and inputs[1:] == [12, 12]
    assert result is first
    assert report['simplification']['search_stop'] == 'two_refinements_without_improvement'
    assert report['simplification']['attempts'][-1]['reason'] == 'sample_exceeds_tolerance'


def test_planar_screen_skips_curved_mesh_and_can_be_overridden():
    mesh = mm.makeUVSphere(1, 48, 48)
    screen = reduction.planar_opportunity(mesh, mm, Settings(.001))
    assert not screen['run'] and screen['estimated_reduction_fraction'] < .03
    assert reduction.planar_opportunity(mesh, mm, Settings(.001, planar_finish_always=True))['run']


def test_native_reducer_continues_from_repaired_checkpoint():
    mesh = mm.makeCube()
    params = mm.SubdivideSettings()
    params.maxEdgeLen, params.maxAngleChangeAfterFlip = .1, 0
    mm.subdivideMesh(mesh, params)
    calls = []

    def checkpoint(candidate, stages):
        calls.append(candidate.computeBoundingBox().center().x)
        if len(calls) == 1:
            candidate.transform(mm.AffineXf3f.translation(mm.Vector3f(.001, 0, 0)))
            return candidate
        return True

    result, _ = reduction.propose_reduction(mesh, mm, Settings(.02, simplify_curved=True),
        Progress(lambda _: None), checkpoint=checkpoint)
    assert len(calls) >= 2
    assert calls[1]-calls[0] == pytest.approx(.001, abs=1e-5)
    assert result.computeBoundingBox().center().x == pytest.approx(.001, abs=1e-5)


def test_near_deleted_shell_is_reported_without_hiding_deviation():
    source = mm.makeCube()
    target = mm.makeCube()
    target.transform(mm.AffineXf3f.translation(mm.Vector3f(-.02, 0, 0)))
    box = source.computeBoundingBox()
    region = {'reason': 'small_shell', 'bbox_min_mm': [box.max.x-.01, box.min.y-.01, box.min.z-.01],
              'bbox_max_mm': [box.max.x+.01, box.max.y+.01, box.max.z+.01]}
    report = source_deviation(source, target, mm, .015, None, {'cuda_disabled': True}, removed_regions=[region])
    assert report['vertices_over_quality'] > 0
    assert report['removed_shell_proximity']['vertices_over_quality'] > 0
    assert report['max_deviation_mm'] > report['warning_threshold_mm']
    assert not report['removed_shell_proximity_excluded']
