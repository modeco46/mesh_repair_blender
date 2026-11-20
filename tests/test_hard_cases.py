"""Export seam failures, fixed error hotspots, and large-report work limits."""
import math
import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn

from mesh_repair_blender.core import engine, hotspots, reduction, planes
from mesh_repair_blender.core.engine import Settings, Progress, diagnostics, valid
from mesh_repair_blender.core.transfer import save_mesh_arrays


def dense_cube(edge=.1):
    mesh = mm.makeCube()
    settings = mm.SubdivideSettings()
    settings.maxEdgeLen, settings.maxAngleChangeAfterFlip = edge, 0
    mm.subdivideMesh(mesh, settings)
    return mesh


def test_indexed_transfer_preserves_coincident_but_distinct_vertices(tmp_path):
    mesh = mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(0, 0, 0))
    mesh.addMesh(mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(1, 1, 1)))
    assert valid(diagnostics(mesh, mm))
    paths = (tmp_path/'vertices.npy', tmp_path/'faces.npy')
    report = save_mesh_arrays(mesh, paths)
    vertices, faces = (np.load(p, allow_pickle=False) for p in paths)
    assert len(vertices) == 16 and len(np.unique(vertices, axis=0)) == 15
    reconstructed = mn.meshFromFacesVerts(faces, vertices)
    assert valid(diagnostics(reconstructed, mm)) and diagnostics(reconstructed, mm)['components'] == 2
    assert report['verified'] and not report['stl_vertex_welding']


def test_blender_output_does_not_reload_through_lossy_stl(tmp_path, monkeypatch):
    path = tmp_path/'input.stl'
    mm.saveMesh(mm.makeCube(), path)
    native = mm.loadMesh

    def load(filename, *args):
        assert not str(filename).endswith('result.stl')
        return native(filename, *args)

    monkeypatch.setattr(mm, 'loadMesh', load)
    arrays = (tmp_path/'vertices.npy', tmp_path/'faces.npy')
    report = engine.run(path, settings=Settings(disable_cuda=True), work_dir=tmp_path/'job', array_output=arrays)
    assert report['serialization']['verified'] and 'stl' not in report['artifacts']
    assert np.load(arrays[1]).shape == (12, 3)
    assert valid(report['after'])


def test_failed_stl_diagnostics_survive_in_partial_report(tmp_path, monkeypatch):
    path = tmp_path/'input.stl'
    mm.saveMesh(mm.makeCube(), path)
    native = mm.loadMesh

    def load(filename, *args):
        result = native(filename, *args)
        if str(filename).endswith('result.stl'):
            result.topology.deleteFace(mm.FaceId(0))
        return result

    monkeypatch.setattr(mm, 'loadMesh', load)
    with pytest.raises(RuntimeError, match='STL') as failure:
        engine.run(path, settings=Settings(disable_cuda=True), work_dir=tmp_path/'job')
    attempts = failure.value.report['export_attempts']
    assert len(attempts) == 1 and attempts[0]['boundary_loops'] == 1


def test_protection_follows_pack_mapping_and_native_decimation():
    mesh = dense_cube()
    vertices = mn.getNumpyVerts(mesh)
    point = vertices[np.argmax(vertices[:, 0]+vertices[:, 1]+vertices[:, 2])].copy()
    locked = hotspots.protected_vertices(mesh, point[None], mm)
    old = [tuple(mesh.points[v]) for v in locked]
    mapping = mesh.packOptimally(False)
    mapped = hotspots.remap_locked_vertices(locked, mapping, mm)
    new = [tuple(mesh.points[v]) for v in mapped]
    assert sorted(old) == sorted(new)
    reduced, stages = reduction.propose_reduction(mesh, mm, Settings(.02, simplify_curved=True),
        Progress(lambda _: None), protection_points=point[None])
    assert stages[0]['protected_hotspot_vertices'] > 0
    assert np.any(np.all(mn.getNumpyVerts(reduced) == point, axis=1))
    assert valid(diagnostics(reduced, mm))


def test_error_plateau_reports_locations_and_stops_after_protection(monkeypatch):
    reference = dense_cube()
    shifted = mm.makeCube()
    shifted.transform(mm.AffineXf3f.translation(mm.Vector3f(.05, 0, 0)))
    calls = []

    def propose(mesh, *args, checkpoint=None, protection_points=None, **kwargs):
        calls.append(protection_points)
        stages = [{'pass': 1, 'before': mesh.topology.numValidFaces(), 'after': 12}]
        checkpoint(mm.Mesh(shifted), stages)
        return mm.Mesh(shifted), stages

    monkeypatch.setattr(reduction, 'propose_reduction', propose)
    report = {'after': diagnostics(reference, mm), 'steps': [], 'warnings': [],
              'acceleration': {'cuda_disabled': True}}
    result = engine.simplify_attempts(reference, None, mm, Settings(.02, simplify_curved=True),
        Progress(lambda _: None), report, lambda _: None, None)
    summary = report['simplification']
    assert result is reference and not summary['accepted']
    assert len(calls) == 3 and calls[0] is None and calls[1] is not None and calls[2] is not None
    assert summary['search_stop'] == 'distance_plateau_after_local_protection'
    worst = summary['attempts'][0]['hotspots']['top20'][0]
    assert len(worst['coordinates_mm']) == 3 and worst['distance_mm'] > .04


def test_final_planar_pass_can_repair_a_few_collisions(monkeypatch):
    reference = mm.makeUVSphere(.1, 24, 24)
    vertices = mn.getNumpyVerts(reference)
    vertices[np.argmax(vertices[:, 0])] += [0, .06, 0]
    candidate = mn.meshFromFacesVerts(mn.getNumpyFaces(reference.topology), vertices)
    params = mm.SubdivideSettings()
    params.maxEdgeLen = .01
    mm.subdivideMesh(reference, params)
    monkeypatch.setattr(reduction, 'propose_reduction', lambda *a, **kw: (candidate, []))
    report = {'after': diagnostics(reference, mm), 'steps': [], 'warnings': [],
              'acceleration': {'cuda_disabled': True}}
    result = engine.finish_planar_faces(reference, reference, mm, Settings(.02, planar_finish_always=True),
        Progress(lambda _: None), report, lambda _: None)
    assert report['planar_finish']['accepted'] and valid(diagnostics(result, mm))
    assert report['planar_finish']['local_repair']['repair']['accepted']


def test_large_plane_report_is_sampled_and_labels_estimates():
    mesh = dense_cube()
    report = planes.triangle_distribution(mesh, mesh, mm, .02, sample_limit=100)
    assert report['sampled'] and report['counts_are_estimates']
    assert report['analyzed_triangles'] == 100 and report['total_triangles'] == mesh.topology.numValidFaces()
    assert not report['source_residuals_available']
    assert all(p['rougher_than_tolerance'] is None for p in report['planes'])


@pytest.mark.parametrize('angle', [1, 10, 45])
def test_bulk_crease_corners_match_native_edge_endpoints(angle):
    for mesh in (dense_cube(.25), mm.makeUVSphere(1, 48, 48)):
        creases = mm.findCreaseEdges(mesh.topology, mesh.points, math.radians(angle))
        assert reduction.crease_corners(mesh, creases, mm, mn) == reduction.crease_corners(
            mesh, creases, mm, mn, angle_deg=angle)


def test_screen_keeps_short_edge_opportunities_on_curved_surface():
    mesh = mm.makeUVSphere(1, 320, 320)
    assert reduction.planar_opportunity(mesh, mm, Settings(.018))['run']
