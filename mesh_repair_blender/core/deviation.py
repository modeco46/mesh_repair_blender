"""Chunked source-vertex distances, full sidecar map and compact JSON hotspots."""
import time
import numpy as np
from .compute import Projector
from .reduction import projection_arrays


def inside_flags(vertices, distances, target, mm, quality_mm):
    """Classify all vertices beyond tolerance with native signed-distance batches."""
    from meshlib import mrmeshnumpy as mn

    flags = np.zeros(len(vertices), dtype=bool)
    candidates = np.flatnonzero(distances > quality_mm)
    for start in range(0, len(candidates), 65536):
        indices = candidates[start:start + 65536]
        points = mm.VertCoords(mn.fromNumpyArray(vertices[indices].astype(np.float32)))
        signed = mm.findSignedDistances(target, points)
        values = np.fromiter(signed.vec_, dtype=np.float32, count=len(indices))
        if not np.isfinite(values).all():
            raise RuntimeError('Не удалось определить знак отклонения вершин.')
        flags[indices] = values < 0
    return flags


def source_deviation(source, target, mm, quality_mm, map_path, acceleration=None,
                     progress=lambda value: None, emit=lambda text: None):
    from meshlib import mrmeshnumpy as mn
    started = time.monotonic()
    source.pack()
    vertices = mn.getNumpyVerts(source)
    distances = np.empty(len(vertices), dtype=np.float32)
    projector = Projector(target, mm, acceleration, emit)
    for start in range(0, len(vertices), 131072):
        points = mn.fromNumpyArray(np.ascontiguousarray(vertices[start:start+131072], dtype=np.float32))
        projections = mm.std_vector_MeshProjectionResult()
        projector.findProjections(projections, points, None, None, float(np.finfo(np.float32).max), 0)
        ids, squared = projection_arrays(projections, mm)
        if len(squared) != len(vertices[start:start+131072]) or (ids < 0).any() or not np.isfinite(squared).all():
            raise RuntimeError('Не удалось вычислить карту отклонения: отсутствует проекция на поверхность.')
        distances[start:start+len(squared)] = np.sqrt(np.maximum(squared, 0))
        progress((start+len(squared))/len(vertices))
    threshold = quality_mm
    inside = inside_flags(vertices, distances, target, mm, quality_mm)
    classified = True
    outer = np.where(inside, 0, distances).astype(np.float32)     # internal source vertices do not count
    count = min(10, len(distances))
    top = np.argpartition(outer, len(outer)-count)[-count:]
    top = top[np.argsort(-outer[top], kind='stable')]
    # No millions of JSON entries: the map is a separate, lossless NumPy file.
    error = None
    try:
        if map_path is not None:
            with open(map_path, 'wb') as stream:
                np.savez(stream, vertices=vertices, distances_mm=outer, all_distances_mm=distances,
                         interior=inside, threshold_mm=threshold)
    except OSError as exc:
        error = str(exc)
    return {'heatmap_error': error,'mode': 'measured_not_proven', 'direction': 'source_vertices_to_result_surface',
            'max_deviation_mm': float(outer.max()), 'max_deviation_all_mm': float(distances.max()),
            'warning_threshold_mm': threshold, 'interior_classified': classified,
            'interior_vertices': int(inside.sum()),
            'interior_max_deviation_mm': float(distances[inside].max()) if inside.any() else 0.0,
            'vertex_count': len(vertices), 'exceeding_vertices': int(np.count_nonzero(outer > threshold)),
            'all_vertices_over_quality': int(np.count_nonzero(distances > quality_mm)),
            'vertices_over_two_steps': int(np.count_nonzero(outer > 2*quality_mm)),
            'vertices_over_quality': int(np.count_nonzero(outer > quality_mm)),
            'top10': [{'vertex_id': int(i), 'coordinates_mm': vertices[i].tolist(),
                       'deviation_mm': float(outer[i])} for i in top],
            'elapsed_seconds': time.monotonic()-started}


def recover_exterior_details(source, current, voxel_reference, mm, q, acceleration, report):
    """Recover isolated sub-voxel losses on an already simplified, closed surface."""
    from .engine import diagnostics, level_set_distances, valid

    checked = source_deviation(source, current, mm, q, None, acceleration)
    count = checked['vertices_over_quality']
    report.update(accepted=False, exterior_over_before=count, max_before_mm=checked['max_deviation_mm'])
    # The ten recorded peaks cover all offenders only in this case. Larger
    # mismatches require reconstruction; a local vertex adjustment cannot fix them.
    if not 0 < count <= len(checked['top10']) or checked['max_deviation_mm'] > 1.25*q:
        return current, None
    shifts = {}
    for peak in checked['top10']:
        if peak['deviation_mm'] <= q:
            continue
        point = mm.Vector3f(*peak['coordinates_mm'])
        projection = current.projectPoint(point)
        delta = point-projection.proj.point
        shift = np.asarray(list(delta*(1-.91*q/delta.length())))
        for vertex in current.topology.getTriVerts(projection.proj.face):
            shifts.setdefault(int(vertex), []).append(shift)
    candidate = mm.Mesh(current)
    moved = mm.VertBitSet()
    for vertex, offsets in shifts.items():
        v = mm.VertId(vertex)
        moved.autoResizeSet(v)
        candidate.points[v] = candidate.points[v]+mm.Vector3f(*np.mean(offsets, axis=0))
    candidate.invalidateCaches()
    changed = mm.getIncidentFaces(current.topology, moved)
    if any(mm.dot(current.dirDblArea(f), candidate.dirDblArea(f)) <= 0 for f in changed):
        report['reason'] = 'face_flipped_or_degenerate'
        return current, None
    after = diagnostics(candidate, mm)
    if not valid(after) or abs(after['signed_volume_mm3']-current.volume()) > .001*current.volume():
        report['reason'] = 'invalid_candidate'
        return current, None
    fidelity = level_set_distances(voxel_reference, candidate, mm, q)
    if not fidelity['within_step']:
        report.update(reason='voxel_reference_distance', voxel_reference=fidelity)
        return current, None
    measured = source_deviation(source, candidate, mm, q, None, acceleration)
    report.update(exterior_over_after=measured['vertices_over_quality'], max_after_mm=measured['max_deviation_mm'])
    if measured['vertices_over_quality']:
        report['reason'] = 'exterior_distance'
        return current, None
    report.update(accepted=True, moved_vertices=len(shifts), voxel_reference=fidelity)
    return candidate, after
