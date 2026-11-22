"""Oriented ray crossings for the regular grid of a nearly closed triangle mesh."""
import math
import time

import numpy as np

from .compute import Projector
from .reduction import ProjectionLayoutError, projection_arrays
from .winding import _float_view, _grid_boxes


class ScanUnavailable(RuntimeError):
    """The bounded scan cannot establish a consistent solid on this grid."""


def _triangles(mesh, mm):
    from meshlib import mrmeshnumpy as mn
    vertices = mn.getNumpyVerts(mesh)
    faces = mn.getNumpyFaces(mesh.topology)
    if len(faces) != mesh.topology.numValidFaces():
        valid_faces = np.flatnonzero(mn.getNumpyBitSet(mesh.topology.getValidFaces()))
        faces = faces[valid_faces]
    triangles = vertices[faces]
    fans = []
    for edge in mesh.topology.findHoleRepresentiveEdges():
        loop = mm.trackRightBoundaryLoop(mesh.topology, edge)
        rim = vertices[[int(mesh.topology.org(e)) for e in loop]]
        if len(rim) <= 2:
            continue  # A doubled, oppositely directed segment has zero solid angle.
        center = np.broadcast_to(rim.mean(axis=0), rim.shape)
        fans.append(np.stack((center, rim, np.roll(rim, -1, axis=0)), axis=1))
    # Auxiliary oriented fans close the counting chain. The field inside every
    # fan neighbourhood will be evaluated on the unmodified mesh by the SDK.
    return np.concatenate([triangles, *fans]) if fans else triangles


def _edge_function(ax, ay, bx, by):
    # Opposite directions of a shared edge use the exact same two products.
    return ax*by - ay*bx


def _crossings(mesh, mm, coords, *, work_limit=150_000_000):
    triangles = _triangles(mesh, mm)
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    starts = [c.searchsorted(low[:, i], side='left') for i, c in enumerate(coords)]
    widths = [c.searchsorted(high[:, i], side='right')-starts[i] for i, c in enumerate(coords)]
    # Sweep X or Y so a complete ray fits within one Z layer of the mesher.
    work = [int((widths[1]*widths[2]).sum()), int((widths[2]*widths[0]).sum())]
    axis = int(np.argmin(work))
    if work[axis] > work_limit:
        raise ScanUnavailable('Слишком много проекций граней на плоскость сканирования.')
    u, v = (axis+1) % 3, (axis+2) % 3
    ends = np.cumsum(widths[u]*widths[v], dtype=np.int64)
    keys, signs = [], []
    crossing_count = 0
    strides = (1, len(coords[0]), len(coords[0])*len(coords[1]))
    for offset in range(0, work[axis], 131072):
        ids = np.arange(offset, min(offset+131072, work[axis]), dtype=np.int64)
        face = ends.searchsorted(ids, side='right')
        cell = ids - (ends[face]-widths[u][face]*widths[v][face])
        iu = starts[u][face]+cell % widths[u][face]
        iv = starts[v][face]+cell // widths[u][face]
        points = triangles[face]
        x = points[:, :, u]-coords[u][iu, None]
        y = points[:, :, v]-coords[v][iv, None]
        weights = np.stack([_edge_function(x[:, a], y[:, a], x[:, b], y[:, b])
                            for a, b in ((1, 2), (2, 0), (0, 1))], axis=1)
        area = weights.sum(axis=1)
        positive = area > 0
        covered = area != 0
        for k, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
            w = weights[:, k]
            # Half-open top-left rule: exactly one incident triangle owns a
            # ray through a shared projected edge, for either face orientation.
            top_left = (y[:, b] > y[:, a]) | ((y[:, b] == y[:, a]) & (x[:, b] < x[:, a]))
            covered &= np.where(positive, (w > 0) | ((w == 0) & top_left),
                                (w < 0) | ((w == 0) & ~top_left))
        if not covered.any():
            continue
        crossing_count += np.count_nonzero(covered)
        if crossing_count > 16_000_000:
            raise ScanUnavailable('Превышен бюджет памяти для сортировки пересечений лучей.')
        hit = (weights[covered]*points[covered, :, axis]).sum(axis=1)/area[covered]
        ia = coords[axis].searchsorted(hit, side='left')
        if (ia >= len(coords[axis])).any():
            raise ScanUnavailable('Пересечение луча вне воксельной сетки.')
        keys.append(ia*strides[axis] + iu[covered]*strides[u] + iv[covered]*strides[v])
        signs.append(np.where(positive[covered], -1, 1).astype(np.int8))
    if not keys:
        raise ScanUnavailable('При сканировании не найдено пересечений с поверхностью.')
    keys, signs = np.concatenate(keys), np.concatenate(signs)
    order = np.argsort(keys, kind='stable')
    return axis, keys[order], signs[order], work[axis]


def scan_surface(mesh, mm, winding, params, boxes, completed, acceleration, *, budget=4_000_000, probe_signs=None):
    """Mesh the original grid. Project only endpoints of sign-changing edges."""
    from meshlib import mrmeshnumpy as mn
    start = time.monotonic()
    step = params.voxelSize
    grid = mm.calcOriginAndDimensions(mesh.computeBoundingBox(), step)
    dims = grid.dimensions
    if dims.x*dims.y > 16_000_000 or max(dims) > 2_000_000 or mesh.topology.numValidFaces() > 2_000_000:
        raise ScanUnavailable('Превышен бюджет памяти для послойного сканирования.')
    spacing = mm.Vector3f.diagonal(step)
    center = grid.origin+spacing*.5
    coords = [(c+np.arange(n, dtype=np.float64)*step).astype(np.float32).astype(np.float64)
              for c, n in zip(center, dims)]
    guard_boxes = _grid_boxes(boxes, center, step, dims)
    guard_samples = sum(math.prod(b-a for a, b in zip(lo, hi)) for lo, hi in guard_boxes)
    if guard_samples > 8_000_000:
        raise ScanUnavailable('Превышен бюджет памяти для сканирования контуров.')
    axis, keys, signs, tests = _crossings(mesh, mm, coords)
    # For a complete, oriented chain all rays return to exterior winding zero.
    nx, ny, nz = list(dims)
    plane = nx*ny
    stops = keys.searchsorted(np.arange(nz+1, dtype=np.int64)*plane)
    options = mm.DistanceToMeshOptions()
    options.maxDistSq = (1.001*step)**2
    options.nullOutsideMinMax = False
    options.windingNumberThreshold, options.windingNumberBeta = params.windingNumberThreshold, params.windingNumberBeta
    cap = math.sqrt(options.maxDistSq)
    guards = []
    for (x0, y0, z0), (x1, y1, z1) in guard_boxes:
        size = mm.Vector3i(x1-x0, y1-y0, z1-z0)
        buffer = mm.std_vector_float()
        buffer.resize(size.x*size.y*size.z)
        origin = center + mm.Vector3f(x0*step, y0*step, z0*step)
        winding.calcFromGridWithDistances(buffer, size, mm.AffineXf3f(mm.Matrix3f.scale(spacing), origin),
                                          options, mm.func_bool_from_float())
        values = _float_view(buffer, (size.z, size.y, size.x)).copy()
        if not np.isfinite(values).all():
            raise ScanUnavailable('Поле внутри контура содержит неопределённые значения.')
        guards.append((x0, x1, y0, y1, z0, z1, values))

    def sign_plane(z):
        if not 0 <= z < nz:
            return np.zeros((ny, nx), dtype=bool)
        counts = np.zeros((ny, nx), dtype=np.int32)
        region = slice(stops[z], stops[z+1])
        np.add.at(counts.ravel(), keys[region]-z*plane, signs[region])
        np.cumsum(counts, axis=1-axis, dtype=np.int32, out=counts)
        if (counts[:, -1] if axis == 0 else counts[-1]).any():
            raise ScanUnavailable('Луч не вернулся в наружную область сетки.')
        inside = counts > 0
        for x0, x1, y0, y1, z0, z1, field in guards:
            if z0 <= z < z1:
                inside[y0:y1, x0:x1] = field[z-z0] < 0
        return inside

    projector = Projector(mesh, mm, acceleration)
    marching = mm.MarchingCubesParams()
    marching.origin, marching.iso, marching.lessInside = grid.origin, 0, True
    mesher = mm.MarchingCubesByParts(dims, marching)
    depth = min(nz, max(2, budget//plane))
    progress = {'grid': list(dims), 'total_layers': nz, 'block_layers': depth, 'completed_layers': 0,
                'completed_blocks': 0, 'sweep_axis': 'XY'[axis], 'ray_triangle_tests': tests,
                'ray_crossings': len(keys), 'hole_box_samples': guard_samples, 'projected_samples': 0,
                'unique_grid_samples': 0, 'setup_seconds': time.monotonic()-start, 'field_seconds': 0.,
                'marching_cubes_seconds': 0., 'projection_device': 'cuda' if projector.gpu else 'cpu'}
    completed(progress.copy())
    previous, current, following = sign_plane(-1), sign_plane(0), sign_plane(1)
    overlap = None
    while True:
        z = mesher.nextZ()
        end = min(z+depth, nz)
        volume = mm.SimpleVolume()
        volume.dims, volume.voxelSize = mm.Vector3i(nx, ny, end-z), spacing
        data = volume.data.vec_
        data.resize(plane*(end-z))
        slab = _float_view(data, (end-z, ny, nx))
        started = time.monotonic()
        if overlap is not None:
            slab[0] = overlap
        for k in range(z+int(overlap is not None), end):
            field = slab[k-z]
            field.fill(cap)
            np.negative(field, out=field, where=current)
            active = (current != previous) | (current != following)
            for dim in (0, 1):
                a = (slice(1, None), slice(None)) if dim == 0 else (slice(None), slice(1, None))
                b = (slice(None, -1), slice(None)) if dim == 0 else (slice(None), slice(None, -1))
                change = current[a] != current[b]
                active[a] |= change
                active[b] |= change
            for x0, x1, y0, y1, z0, z1, exact in guards:
                if z0 <= k < z1:
                    field[y0:y1, x0:x1] = exact[k-z0]
                    active[y0:y1, x0:x1] = False
            points = np.flatnonzero(active)
            for first in range(0, len(points), 131072):
                indices = points[first:first+131072]
                query = np.empty((len(indices), 3), dtype=np.float32)
                query[:, 0], query[:, 1], query[:, 2] = coords[0][indices % nx], coords[1][indices // nx], coords[2][k]
                projections = mm.std_vector_MeshProjectionResult()
                projector.findProjections(projections, mn.fromNumpyArray(query), None, None, options.maxDistSq, 0.)
                try:
                    ids, squared = projection_arrays(projections, mm)
                except ProjectionLayoutError as exc:
                    raise ScanUnavailable(str(exc)) from exc
                if len(ids) != len(indices) or not np.isfinite(squared[ids >= 0]).all():
                    raise ScanUnavailable('Не удалось измерить расстояние в активных вокселях.')
                distances = np.sqrt(np.clip(np.where(ids >= 0, squared, options.maxDistSq), 0, options.maxDistSq))
                field.ravel()[indices] = np.copysign(distances, field.ravel()[indices])
            progress['projected_samples'] += len(points)
            progress['unique_grid_samples'] += plane
            if probe_signs is not None:
                probe_signs[k] = field < 0
            previous, current, following = current, following, sign_plane(k+2)
        overlap = slab[-1].copy() if end < nz else None
        progress['field_seconds'] += time.monotonic()-started
        started = time.monotonic()
        mesher.addPart(volume)
        progress['marching_cubes_seconds'] += time.monotonic()-started
        progress.update(completed_layers=end, completed_blocks=progress['completed_blocks']+1,
                        projection_device='cuda' if projector.gpu else 'cpu')
        if end == nz:
            break
        completed(progress.copy())
    started = time.monotonic()
    result = mm.Mesh.fromTriMesh(mesher.finalize())
    progress['mesh_build_seconds'] = time.monotonic()-started
    completed(progress.copy())
    return result


def guard_probe_disagreements(full, scan, grid, step, boxes):
    """Evaluate small ambiguous neighbourhoods with native winding on both grids."""
    changed = np.argwhere(full != scan)
    if not 0 < len(changed) <= 32:
        return None
    first = np.asarray(list(grid.origin)) + .5*step
    center = first + changed[:, ::-1]*step
    guards = np.stack((center-1.5*step, center+1.5*step), axis=1)
    return np.concatenate((boxes, guards))
