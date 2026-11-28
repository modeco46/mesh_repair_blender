import numpy as np
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn

from mesh_repair_blender.core.collisions import patch_neighbours, region_collisions
from mesh_repair_blender.core.holes import patch_collisions


def test_large_triangle_without_local_vertices_is_in_spatial_check():
    # Both endpoints of the long edge lie outside the ball of the edited face.
    faces = np.arange(6, dtype=np.int32).reshape(2, 3)
    points = np.float32([[-.1, -.1, 0], [.1, -.1, 0], [0, .1, 0],
                         [-90, 0, -1], [90, 0, -1], [0, 0, 70]])
    mesh = mn.meshFromFacesVerts(faces, points)
    mesh.addMesh(mm.makeCube(mm.Vector3f.diagonal(1), mm.Vector3f.diagonal(20)))
    touched = mm.FaceBitSet()
    touched.autoResizeSet(mm.FaceId(0))
    region = patch_neighbours(mesh, points[faces[:1]], touched, mm)
    assert region is not None and region.count() == 2
    hits = region_collisions(mesh, region, mm)
    assert hits is not None and hits == mm.findSelfCollidingTrianglesBS(mm.MeshPart(mesh))
    assert set(map(int, hits)) == {0, 1}
    assert region_collisions(mesh, mesh.topology.getValidFaces(), mm) is None


def test_local_patch_intersections_match_native_face_ids():
    rng = np.random.default_rng(17)
    hit_cases = 0
    for i in range(16):
        before = mm.makeUVSphere(1, 24, 24)
        # Nearly coincident sheets exercise collisions with unedited geometry.
        if i % 2:
            before.addMesh(mm.makeUVSphere(.985, 24, 24))
        before.deleteFaces(mn.faceBitSetFromBools(np.arange(before.topology.faceSize()) == 27))
        candidate = mm.Mesh(before)
        settings = mm.FillHoleNicelySettings()
        settings.triangulateOnly = True
        patch = mm.fillHoleNicely(candidate, candidate.topology.findHoleRepresentiveEdges()[0], settings)
        corner = candidate.topology.getTriVerts(next(iter(patch)))[0]
        if i % 4:
            candidate.points[corner] += mm.Vector3f(*rng.normal(0, .18, 3))
            candidate.invalidateCaches()
        affected = mm.FaceBitSet()
        if i % 4:
            moved = mm.VertBitSet()
            moved.autoResizeSet(corner)
            affected = mm.getIncidentFaces(candidate.topology, moved)
        affected |= patch
        expected = mm.findSelfCollidingTrianglesBS(mm.MeshPart(candidate)) & affected
        assert patch_collisions(before, candidate, patch, mm) == expected
        hit_cases += bool(expected.count())
    assert 0 < hit_cases < 16
