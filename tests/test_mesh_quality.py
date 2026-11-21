"""Known solids and malformed triangle soups, with independent output checks."""
import struct

import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn

from mesh_repair_blender.core import engine, reduction
from mesh_repair_blender.core.engine import Settings, Progress, diagnostics, valid
from mesh_repair_blender.core.solid_union import many_shell_union
from mesh_repair_blender.core.transfer import write_indexed_mesh


def panel_settings(q=.02):
    return Settings(q, restore_sharp_edges=False, disable_cuda=True, weld_close_vertices=True,
                    remove_small_shells=True, simplify_curved=True, collapse_coplanar=True)


def cube(size=2):
    return mm.makeCube(mm.Vector3f(size, size, size), mm.Vector3f(0, 0, 0))


def repair(path, q=.02):
    paths = (path.parent/'v.npy', path.parent/'f.npy')
    report = engine.run(path, work_dir=path.parent/'job', settings=panel_settings(q), array_output=paths)
    assert report['serialization']['exact_roundtrip']
    assert report['final_roundtrips'] == 0 and not (path.parent/'job'/'result.stl').exists()
    vertices, faces = (np.load(p, allow_pickle=False) for p in paths)
    assert vertices.dtype == np.float32 and faces.dtype == np.int32
    mesh = mn.meshFromFacesVerts(faces, vertices)
    assert valid(diagnostics(mesh, mm))
    return mesh, report


@pytest.mark.parametrize('size,q', [(2, .02), (20, .01)])
def test_open_box_matches_meshlab_cap_and_analytic_volume(tmp_path, size, q):
    mesh = cube(size)
    faces = mn.getNumpyFaces(mesh.topology)
    vertices = mn.getNumpyVerts(mesh)
    for face in np.flatnonzero(np.all(vertices[faces, 2] == size, axis=1)):
        mesh.topology.deleteFace(mm.FaceId(int(face)))
    source = tmp_path/'open.stl'
    mm.saveMesh(mesh, source)
    repaired, report = repair(source, q)
    assert report['hole_filling']['details'][0]['patch_triangles'] == 2
    assert report['algorithm'] == 'surface_repair' and 'voxel_passes' not in report
    assert not report['warnings']
    assert repaired.topology.numValidFaces() == 12 and repaired.volume() == pytest.approx(size**3)
    assert set(map(tuple, mn.getNumpyVerts(repaired))) == set(map(tuple, vertices))


@pytest.mark.parametrize('defect', ['duplicate', 'reversed_duplicate', 'collinear'])
def test_invalid_extra_triangle_cannot_replace_or_open_cube(tmp_path, defect):
    path = tmp_path/'dirty.stl'
    mm.saveMesh(cube(), path)
    data = path.read_bytes()
    record = data[84:134]
    if defect == 'reversed_duplicate':
        record = record[:24] + record[36:48] + record[24:36] + record[48:]
    elif defect == 'collinear':
        record = struct.pack('<12fH', 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0)
    path.write_bytes(data[:80] + struct.pack('<I', 13) + data[84:] + record)
    repaired, report = repair(path)
    field = 'removed_degenerate_faces' if defect == 'collinear' else 'removed_duplicate_faces'
    assert report['orientation'][field] == 1
    assert repaired.topology.numValidFaces() == 12 and repaired.volume() == pytest.approx(8)
    assert report['source_deviation']['max_deviation_mm'] == 0


def test_flat_drawing_and_thin_solid_are_distinguished(tmp_path):
    plane = mn.meshFromFacesVerts(np.array([[0, 1, 2]]), np.array([[9, 0, -1], [1, 3, 4], [2, 0, 1.]]))
    source = tmp_path/'sheet.stl'
    mm.saveMesh(plane, source)
    with pytest.raises(ValueError, match='Solidify') as failure:
        repair(source)
    assert failure.value.report['failure_reason'] == 'planar_input_without_thickness'
    assert not (tmp_path/'v.npy').exists()

    solid = mm.makeCube(mm.Vector3f(1, 2, .0001), mm.Vector3f(0, 0, 0))
    mm.saveMesh(solid, source)
    result, _ = repair(source)
    assert result.volume() == pytest.approx(.0002, rel=1e-5)


def assembly():
    solid = mm.makeCube(mm.Vector3f(1.8, .6, .1), mm.Vector3f(0, 0, 0))
    for row in range(4):
        for col in range(8):
            solid.addMesh(mm.makeCube(mm.Vector3f(.08, .08, .12),
                          mm.Vector3f(.06+col*.21, .045+row*.14, .07)))
    return solid


def test_33_crossing_solids_retain_all_posts_at_fine_step(tmp_path):
    source = tmp_path/'assembly.stl'
    mm.saveMesh(assembly(), source)
    mesh, report = repair(source, q=.01)
    assert report['algorithm'] == 'grouped_solid_union'
    assert 'voxel_passes' not in report and report['after']['components'] == 1
    assert mesh.volume() == pytest.approx(1.8*.6*.1+32*.08**2*.09, rel=1e-4)
    assert report['assembly_union']['max_operand_outside_mm'] < .001


def test_union_losing_posts_is_rejected_even_when_base_volume_passes(monkeypatch):
    mesh = assembly()
    base = mm.makeCube(mm.Vector3f(1.8, .6, .1), mm.Vector3f(0, 0, 0))
    monkeypatch.setattr(mm, 'uniteManyMeshes', lambda *args: base)
    report = {}
    result, _ = many_shell_union(mesh, diagnostics(mesh, mm), mm, panel_settings(.01),
                                 Progress(lambda _: None), report, lambda _: None)
    assert result is None and report['assembly_union']['reason'] == 'lost_exterior'
    assert mesh.topology.numValidFaces() == 396


def cylinder(wall):
    count = 32
    theta = 2*np.pi*np.arange(count)/count
    circle = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    outer = np.column_stack((circle, np.zeros(count)))
    lift = [0, 0, .01]
    verts = [*outer, *(outer+lift)]
    faces = []
    for i in range(count):
        j = (i+1) % count
        faces.extend([(i, j, i+count), (i+count, j, j+count)])
    if wall:
        inner = outer*.008
        verts.extend([*inner, *(inner+lift)])
        for i in range(count):
            j = (i+1) % count
            faces.extend([(j+2*count, i+2*count, j+3*count), (j+3*count, i+2*count, i+3*count),
                          (i, i+2*count, j), (j, i+2*count, j+2*count),
                          (i+count, j+count, i+3*count), (i+3*count, j+count, j+3*count)])
    else:
        verts.extend([[0, 0, 0], lift])
        for i in range(count):
            j = (i+1) % count
            faces.extend([(j, i, 2*count), (i+count, j+count, 2*count+1)])
    return mn.meshFromFacesVerts(np.asarray(faces, np.int32), np.asarray(verts, np.float32))


def test_missing_annular_end_face_restores_wall_and_preserves_bore(tmp_path):
    ring = cylinder(True)
    expected_volume = ring.volume()
    faces = mn.getNumpyFaces(ring.topology)
    z = mn.getNumpyVerts(ring)[:, 2]
    for face in np.flatnonzero(np.all(z[faces] == z.max(), axis=1)):
        ring.topology.deleteFace(mm.FaceId(int(face)))
    source = tmp_path/'open-bushing.stl'
    mm.saveMesh(ring, source)
    mesh, report = repair(source, .005)
    assert report['hole_filling']['details'][0]['method'] == 'joint_planar_contours'
    assert report['hole_filling']['filled'] == 2 and 'voxel_passes' not in report
    after = diagnostics(mesh, mm)
    assert after['triangles'] == 2*after['vertices']  # One handle on one closed surface.
    assert mesh.volume() == pytest.approx(expected_volume, rel=.02)
    assert mesh.computeBoundingBox().size().z == pytest.approx(.01, abs=.0001)


def test_small_through_hole_survives_even_when_filled_disc_passes_distance(monkeypatch):
    ring, disc = cylinder(True), cylinder(False)
    info = diagnostics(ring, mm)
    assert valid(info) and valid(diagnostics(disc, mm))
    assert reduction.measured_distance(ring, disc, mm) < .01
    assert abs(ring.volume()/disc.volume()-1) < .001
    monkeypatch.setattr(reduction, 'propose_reduction', lambda *args, **kwargs: (disc, []))
    report = {'after': info, 'steps': []}
    result = engine.finish_planar_faces(ring, ring, mm, panel_settings(), Progress(lambda _: None), report, print)
    assert result is ring and report['planar_finish']['reason'] == 'planar_finish_topology_changed'


def test_decimation_cannot_thin_a_wall_within_pointwise_tolerance(monkeypatch):
    slab = mm.makeCube(mm.Vector3f(2, 2, .016), mm.Vector3f(0, 0, 0))
    params = mm.SubdivideSettings()
    params.maxEdgeLen, params.maxAngleChangeAfterFlip = .3, 0
    mm.subdivideMesh(slab, params)
    thin = mm.makeCube(mm.Vector3f(2, 2, .008), mm.Vector3f(0, 0, 0))
    assert reduction.measured_distance(slab, thin, mm) < .01
    monkeypatch.setattr(reduction, 'propose_reduction', lambda *args, **kwargs: (thin, []))
    report = {'after': diagnostics(slab, mm), 'steps': [], 'warnings': []}
    tracker = Progress(lambda _: None)
    result = engine.simplify_attempts(slab, None, mm, panel_settings(), tracker, report, print, None)
    assert result is slab and report['simplification']['reason'] == 'volume_changed'
    result = engine.finish_planar_faces(slab, slab, mm, panel_settings(), tracker, report, print)
    assert result is slab and report['planar_finish']['reason'] == 'volume_changed'


def test_indexed_writer_keeps_coincident_vertices_and_rolls_back_partial_files(tmp_path):
    solid = cube()
    solid.addMesh(mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(2, 2, 2)))
    vertices, faces = tmp_path/'v.npy', tmp_path/'f.npy'
    faces.write_bytes(b'prior output')
    with pytest.raises(FileExistsError):
        write_indexed_mesh(solid, (vertices, faces))
    assert not vertices.exists() and faces.read_bytes() == b'prior output'
    faces.unlink()
    write_indexed_mesh(solid, (vertices, faces))
    points = np.load(vertices)
    assert len(points) == 16 and len(np.unique(points, axis=0)) == 15
    actual = mn.meshFromFacesVerts(np.load(faces), points)
    info = diagnostics(actual, mm)
    assert valid(info) and info['components'] == 2
