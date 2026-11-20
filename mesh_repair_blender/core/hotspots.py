"""Find simplification error locations and pin their reference neighbourhoods."""
import numpy as np
from .compute import Projector
from .reduction import projection_arrays


def distance_hotspots(reference, candidate, tolerance, mm, acceleration=None, emit=lambda text: None,
                      max_protection_points=50_000):
    from meshlib import mrmeshnumpy as mn
    reference.pack()
    candidate.pack()
    points, records, total_bad, retained = [], [], 0, 0
    for source, target, direction in ((reference, candidate, 'repaired_to_candidate'),
                                      (candidate, reference, 'candidate_to_repaired')):
        vertices = mn.getNumpyVerts(source)
        target_faces = mn.getNumpyFaces(target.topology) if target is reference else None
        target_vertices = mn.getNumpyVerts(target) if target is reference else None
        projector = Projector(target, mm, acceleration, emit)
        for start in range(0, len(vertices), 131072):
            batch = vertices[start:start+131072]
            projections = mm.std_vector_MeshProjectionResult()
            projector.findProjections(projections, mn.fromNumpyArray(batch), None, None,
                                       float(np.finfo(np.float32).max), 0)
            faces, squared = projection_arrays(projections, mm)
            if (faces < 0).any() or not np.isfinite(squared).all():
                raise RuntimeError('Не удалось определить места превышения допуска.')
            distances = np.sqrt(np.maximum(squared, 0))
            bad = np.flatnonzero(distances > .9*tolerance)
            total_bad += len(bad)
            if len(bad):
                worst = bad[np.argsort(distances[bad])[-10:]]
                records.extend({'direction': direction, 'vertex_id': int(start+i),
                    'coordinates_mm': batch[i].tolist(), 'distance_mm': float(distances[i])} for i in worst)
                remaining = max_protection_points-retained
                if remaining > 0:
                    selected = bad[:remaining] if target_faces is None else bad[:remaining//3]
                    fixed = batch[selected] if target_faces is None else target_vertices[target_faces[faces[selected]]].reshape(-1, 3)
                    points.append(fixed.copy())
                    retained += len(fixed)
    fixed = np.unique(np.concatenate(points), axis=0) if points else np.empty((0, 3), np.float32)
    return {'threshold_mm': .9*tolerance, 'vertices_over_threshold': total_bad,
            'protection_points': len(fixed), 'protection_limit_reached': retained >= max_protection_points-2,
            'top20': sorted(records, key=lambda x: -x['distance_mm'])[:20]}, fixed


def protected_vertices(mesh, points, mm):
    from meshlib import mrmeshnumpy as mn
    if points is None or not len(points):
        return None
    projector = mm.PointsToMeshProjector()
    projector.updateMeshData(mesh)
    projections = mm.std_vector_MeshProjectionResult()
    projector.findProjections(projections, mn.fromNumpyArray(np.asarray(points, np.float32)), None, None,
                               float(np.finfo(np.float32).max), 0)
    face_ids, _ = projection_arrays(projections, mm)
    if (face_ids < 0).any():
        raise RuntimeError('Не удалось защитить область превышения допуска.')
    faces = mm.FaceBitSet()
    for index in np.unique(face_ids):
        faces.autoResizeSet(mm.FaceId(int(index)))
    locked = mm.getIncidentVerts(mesh.topology, faces)
    mm.expand(mesh.topology, locked, 2)
    return locked


def remap_locked_vertices(locked, pack, mm):
    """PackMapping.v maps old vertex ids to packed ids; do not reuse old ids."""
    result = mm.VertBitSet()
    for vertex in locked:
        mapped = pack.v.b[vertex]
        if mapped.valid():
            result.autoResizeSet(mapped)
    return result
