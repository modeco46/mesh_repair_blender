import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn

from mesh_repair_blender.core import engine, reduction
from mesh_repair_blender.core.engine import Settings, Progress, diagnostics, valid


def subdivided_cube(edge=1.5):
    mesh = mm.makeCube(mm.Vector3f(2, 2, 2), mm.Vector3f(0, 0, 0))
    params = mm.SubdivideSettings()
    params.maxEdgeLen = edge
    params.maxAngleChangeAfterFlip = 0
    mm.subdivideMesh(mesh, params)
    return mesh


def finish(reference, current, strict=False):
    report = {'after': diagnostics(current, mm), 'steps': [], 'warnings': [],
              'acceleration': {'cuda_disabled': True}}
    result = engine.finish_planar_faces(reference, current, mm,
        Settings(.015, collapse_coplanar=True, simplify_curved=True, strict_tolerance=strict),
        Progress(lambda _: None), report, lambda _: None)
    return result, report


@pytest.mark.parametrize('edge', [1.5, .25])
def test_planar_finish_removes_exact_coplanar_faces_even_on_small_bodies(edge):
    mesh = subdivided_cube(edge)
    before = mn.getNumpyVerts(mesh).copy()
    if edge == 1.5:
        assert mesh.topology.numValidFaces() < 50
    result, report = finish(mesh, mesh, strict=True)
    item = report['planar_finish']
    assert item['accepted'] and item['after'] == 12
    assert item['verification']['verified'] and item['verification']['upper_bound_mm'] < 1e-8
    assert valid(diagnostics(result, mm))
    assert np.array_equal(before, mn.getNumpyVerts(mesh))
    assert result.volume() == pytest.approx(mesh.volume())


def test_planar_finish_preserves_small_separate_shells():
    mesh = subdivided_cube(.25)
    mesh.addMesh(mm.makeCube(mm.Vector3f(.001, .001, .001), mm.Vector3f(4, 0, 0)))
    result, report = finish(mesh, mesh)
    assert report['planar_finish']['accepted']
    assert diagnostics(result, mm)['components'] == 2
    assert result.topology.numValidFaces() == 24


def test_final_pass_checks_total_deviation_not_just_its_own_change(monkeypatch):
    reference = subdivided_cube()
    current = mm.Mesh(reference)
    current.transform(mm.AffineXf3f.translation(mm.Vector3f(.01, 0, 0)))
    proposed = mm.makeCube(mm.Vector3f(2, 2, 2), mm.Vector3f(.02, 0, 0))
    monkeypatch.setattr(reduction, 'propose_reduction', lambda *a, **kw: (proposed, []))
    assert reduction.measured_distance(current, proposed, mm) < .015
    result, report = finish(reference, current)
    assert result is current and not report['planar_finish']['accepted']
    assert report['planar_finish']['reason'] == 'planar_finish_exceeds_total_tolerance'


def test_invalid_final_pass_rolls_back(monkeypatch):
    mesh = subdivided_cube()
    broken = mm.makeCube()
    broken.topology.deleteFace(mm.FaceId(0))
    monkeypatch.setattr(reduction, 'propose_reduction', lambda *a, **kw: (broken, []))
    result, report = finish(mesh, mesh)
    assert result is mesh and not report['planar_finish']['accepted']
    assert report['planar_finish']['reason'] == 'planar_finish_topology_changed'


def test_full_pipeline_runs_final_pass_with_both_options_enabled(tmp_path):
    path = tmp_path/'cube.stl'
    mm.saveMesh(subdivided_cube(.25), path)
    report = engine.run(path, settings=Settings(.015, collapse_coplanar=True, simplify_curved=True,
                        disable_cuda=True, restore_sharp_edges=False), work_dir=tmp_path/'job')
    assert 'planar_finish' in report
    assert report['after']['triangles'] == 12 and valid(report['after'])


def test_export_failure_rolls_back_only_final_planar_pass(tmp_path, monkeypatch):
    path = tmp_path/'cube.stl'
    source = subdivided_cube()
    mm.saveMesh(source, path)
    # Leave a real planar workload for the last pass.
    monkeypatch.setattr(engine, 'simplify_attempts', lambda mesh, *args: mesh)
    native = mm.loadMesh
    reads = []

    def load(filename, *args):
        result = native(filename, *args)
        if str(filename).endswith('result.stl'):
            reads.append(True)
            if len(reads) == 1:
                result.topology.deleteFace(mm.FaceId(0))
        return result

    monkeypatch.setattr(mm, 'loadMesh', load)
    report = engine.run(path, settings=Settings(.015, collapse_coplanar=True, simplify_curved=True,
                        disable_cuda=True), work_dir=tmp_path/'job')
    assert report['final_roundtrips'] == 2
    assert not report['planar_finish']['accepted']
    assert report['planar_finish']['reason'] == 'export_validation_failed'
    assert report['after']['triangles'] == source.topology.numValidFaces()
    assert valid(report['after'])
