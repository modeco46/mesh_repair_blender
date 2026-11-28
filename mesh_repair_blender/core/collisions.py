"""Collision checks on exact, topology-preserving copies of spatial regions."""
import numpy as np


def region_collisions(mesh, region, mm):
    """Return hits in original face IDs, or None if a compact copy is unsuitable."""
    count = region.count()
    if not 0 < count <= min(32768, mesh.topology.numValidFaces()//4):
        return None
    try:
        face_map = mm.FaceMapOrHashMap.createMap()
        vert_map = mm.VertMapOrHashMap.createMap()
        mapping = mm.PartMapping()
        mapping.tgt2srcFaces, mapping.tgt2srcVerts = face_map, vert_map
        local = mesh.cloneRegion(region, False, mapping)
        faces, vertices = face_map.getMap(), vert_map.getMap()
        if (local.topology.numValidFaces() != count or local.topology.faceSize() != count or
                faces.size() != count or local.topology.numValidVerts() != vertices.size()):
            return None
        # A disconnected fan at a shared source vertex must not be split:
        # the SDK excludes some contacts according to vertex/edge identity.
        if len({int(v) for v in vertices}) != vertices.size():
            return None
        result = mm.FaceBitSet()
        for face in mm.findSelfCollidingTrianglesBS(mm.MeshPart(local)):
            result.autoResizeSet(faces[face])
        return result
    except (RuntimeError, MemoryError):
        return None


def patch_neighbours(mesh, triangles, touched, mm):
    """Enclose every changed triangle and query the updated whole-mesh AABB tree.

    A triangle meeting any changed triangle must have a box intersecting at
    least one query ball, including a long triangle with all vertices outside.
    Return None on a broad query so the caller can use the full native check.
    """
    span = np.ptp(triangles, axis=1).max()
    if not np.isfinite(span) or span <= 0:
        return None
    centers = triangles.mean(axis=1)
    cells = np.floor((centers-centers.min(axis=0))/(4*span)).astype(np.int64)
    groups, labels = np.unique(cells, axis=0, return_inverse=True)
    if len(groups) > 512:
        return None
    nearby = set()

    def found(face, ball):
        nearby.add(int(face))
        return mm.Processing.Continue if len(nearby) <= 32768 else mm.Processing.Stop

    part = mm.MeshPart(mesh)
    for i in range(len(groups)):
        verts = triangles[labels == i].reshape(-1, 3)
        low, high = verts.min(axis=0), verts.max(axis=0)
        center = ((low+high)*.5).astype(np.float32)
        epsilon = 16*np.finfo(np.float32).eps*max(1., float(np.abs(verts).max()))
        radius = np.linalg.norm(np.maximum(high-center, center-low))+epsilon
        mm.findBoxedTrisInBall(part, mm.Ball3f(mm.Vector3f(*center), float(radius*radius)), found)
        if len(nearby) > 32768:
            return None
    region = mm.FaceBitSet()
    for i in nearby:
        region.autoResizeSet(mm.FaceId(i))
    return region if (touched-region).count() == 0 else None
