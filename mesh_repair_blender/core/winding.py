"""Bounded winding batches on a common grid, with native marching cubes."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import math
from threading import Event
import time

import numpy as np


def hole_field_boxes(mesh, mm, step):
    """Enclose possible off-surface closures of small open boundaries.

    Adding a fan to each oriented boundary gives a closed triangle chain with
    integer winding off the triangles, including on self-intersecting inputs.
    A fan's absolute winding is <= area / (4*pi*distance**2). Outside these
    boxes all fans together therefore contribute at most 1/8. The fans are
    used ONLY to bound the field; no cap geometry is added to the repair.
    """
    rims = mesh.topology.findHoleRepresentiveEdges()
    if not 0 < len(rims) <= 1000:
        return None
    boxes = []
    areas = []
    boundary_edges = 0
    for edge in rims:
        loop = mm.trackRightBoundaryLoop(mesh.topology, edge)
        boundary_edges += len(loop)
        if len(loop) < 2 or boundary_edges > 8192:
            return None
        points = np.array([list(mesh.points[mesh.topology.org(e)]) for e in loop])
        center = points.mean(axis=0)
        a = points-center
        areas.append(.5*np.linalg.norm(np.cross(a, np.roll(a, -1, axis=0)), axis=1).sum())
        boxes.append((points.min(axis=0), points.max(axis=0)))
    areas = np.asarray(areas)
    rim_bounds = np.asarray(boxes)
    area = float(areas.sum())
    limit = 4*math.pi*.125
    radius = math.sqrt(area/limit)
    padding = 2*math.sqrt(3)*step
    extent = np.asarray(list(mesh.computeBoundingBox().size())) + 2*step

    def expand(radii):
        bounds = rim_bounds.copy()
        bounds[:, 0] -= radii
        bounds[:, 1] += radii
        fraction = float(np.prod(bounds[:, 1]-bounds[:, 0], axis=1).sum()/np.prod(extent))
        return bounds, fraction

    boxes, share = expand(radius+padding)
    method = 'total_fan_area'
    if len(areas) > 1 and radius > 0 and (share > .15 or share*np.prod(extent/step) > 8_000_000):
        # Outside the guards let i be the nearest fan box. Every other fan j
        # is at least as far, and at least half the separation of boxes i and j
        # away (triangle inequality). Hence d_j >= max(r_i, separation_ij/2).
        # This bounds the sum of solid angles even for overlapping contours.
        radii = np.empty(len(areas))
        for i, (low, high) in enumerate(rim_bounds):
            gap = np.maximum(0, np.maximum(low-rim_bounds[:, 1], rim_bounds[:, 0]-high))
            half_gap_sq = .25*np.einsum('ij,ij->i', gap, gap)
            left, right = 0., radius
            for _ in range(12):
                mid = (left+right)*.5
                if np.sum(areas/np.maximum(mid*mid, half_gap_sq)) <= limit:
                    right = mid
                else:
                    left = mid
            radii[i] = right+padding
        boxes, share = expand(radii[:, None])
        radius = float(radii.max())-padding
        method = 'nearest_fan_separation'
    if not np.isfinite(share) or share > .15:
        return None
    return boxes, {'boundary_loops': len(rims), 'boundary_edges': boundary_edges,
                   'fan_area_bound_mm2': area, 'guard_radius_mm': radius+padding,
                   'guard_bound': method, 'box_volume_fraction_upper_bound': share}


def seam_field_boxes(result, mm, step, translation):
    """Small exact-field boxes in the input grid coordinate system."""
    guards = []
    cells = 0
    offset = np.asarray(list(translation)) if translation is not None else 0.
    for edge in result.topology.findHoleRepresentiveEdges():
        loop = mm.trackRightBoundaryLoop(result.topology, edge)
        if not 2 <= len(loop) <= 64:
            return None
        points = np.array([list(result.points[result.topology.org(e)]) for e in loop])-offset
        low, high = points.min(axis=0)-4*step, points.max(axis=0)+4*step
        cells += math.prod(map(int, np.ceil((high-low)/step)+10))
        if cells > 1_000_000:
            return None
        guards.append((low, high))
    return np.array(guards) if guards else None


def _float_view(vector, shape):
    # The supplied SDK vector owns the memory. Keep it alive and unmodified in
    # size for the complete lifetime of this view, including mesher.addPart().
    if vector.element_type_byte_size != 4 or len(vector) != math.prod(shape) or not vector.data_pointer():
        raise RuntimeError('MeshLib вернул несовместимый буфер воксельного поля.')
    return np.ctypeslib.as_array((ctypes.c_float*len(vector)).from_address(vector.data_pointer())).reshape(shape)


def compare_probe_surfaces(full, band, mm, step, repair_step):
    from .engine import topology_info
    reference, candidate = topology_info(full, mm), topology_info(band, mm)
    # Both projections and marching-cubes interpolation use float32. Bound
    # their numerical differences against the *requested* repair resolution.
    limit = min(step*5e-4, repair_step*.005)
    result = {'probe_matched': False, 'probe_distance_limit_mm': limit,
              'full_probe_volume_mm3': reference['signed_volume_mm3'],
              'full_probe_faces': reference['triangles'], 'band_probe_faces': candidate['triangles']}
    if not reference['triangles'] or not reference['topology_valid'] or any(
            reference[k] != candidate[k] for k in ('components', 'boundary_loops', 'topology_valid', 'triangles', 'vertices')):
        return result
    error = math.sqrt(mm.findMaxDistanceSq(mm.MeshPart(full), mm.MeshPart(band)))
    result.update(probe_matched=error <= limit, probe_surface_distance_mm=error)
    return result


def _grid_boxes(boxes, center, step, dims):
    if boxes is None:
        return []
    # Include an extra grid point at each end against float grid rounding.
    # Coarse checks can have larger voxels than the final guard was built for.
    scaled = (boxes-np.asarray(list(center)))/step
    lo = np.maximum(np.floor(scaled[:, 0]).astype(np.int64)-4, 0)
    hi = np.minimum(np.ceil(scaled[:, 1]).astype(np.int64)+5, list(dims))
    return [(tuple(map(int, a)), tuple(map(int, b))) for a, b in zip(lo, hi) if (a < b).all()]


def winding_slabs(mesh, mm, winding, params, completed, *, sharpen=False, budget=2_000_000, deadline=None,
                  boundary_boxes=None, probe_signs=None, prefetch=False):
    step = params.voxelSize
    grid = mm.calcOriginAndDimensions(mesh.computeBoundingBox(), step)
    dims = grid.dimensions
    spacing = mm.Vector3f.diagonal(step)
    basis = mm.Matrix3f.scale(spacing)
    first_center = grid.origin + spacing*.5
    boxes = _grid_boxes(boundary_boxes, first_center, step, dims)
    options = mm.DistanceToMeshOptions()
    options.maxDistSq = (1.001*step)**2
    options.nullOutsideMinMax = not params.closeHolesInHoleWindingNumber
    options.windingNumberThreshold = params.windingNumberThreshold
    options.windingNumberBeta = params.windingNumberBeta
    band = mm.DistanceToMeshOptions(options)
    band.maxDistSq = (2.01*step)**2
    band.nullOutsideMinMax = True
    cap = math.sqrt(options.maxDistSq)
    marching = mm.MarchingCubesParams()
    marching.origin, marching.iso, marching.lessInside = grid.origin, 0, True
    if sharpen:
        voxel_per_face = mm.Vector_VoxelId_FaceId()
        grid_to_mesh = mm.AffineXf3f()
        marching.outVoxelPerFaceMap, marching.outGridToMeshXf = voxel_per_face, grid_to_mesh
    mesher = mm.MarchingCubesByParts(dims, marching)
    plane = dims.x*dims.y
    depth = max(2, budget//plane)
    overlap = None
    progress = {'grid': [dims.x, dims.y, dims.z], 'total_layers': dims.z, 'block_layers': min(depth, dims.z),
                'completed_layers': 0, 'completed_blocks': 0, 'unique_grid_samples': 0,
                'winding_samples': 0, 'hole_box_samples': 0, 'field_seconds': 0., 'marching_cubes_seconds': 0.}
    fresh, extra = mm.std_vector_float(), mm.std_vector_float()

    def evaluate(buffer, size, origin, settings):
        # The CUDA binding writes to res.data(); it does not resize the vector.
        buffer.resize(size.x*size.y*size.z)
        winding.calcFromGridWithDistances(buffer, size, mm.AffineXf3f(basis, origin), settings,
                                          mm.func_bool_from_float())
        return _float_view(buffer, (size.z, size.y, size.x))

    # The mesher owns a copy of the slab. Only one thread at a time touches
    # the winding calculator or fresh; the executor joins before CPU fallback.
    def next_field(size, origin):
        launched.set()
        started = time.monotonic()
        values = evaluate(fresh, size, origin, options)
        return values, time.monotonic()-started

    pool = ThreadPoolExecutor(max_workers=1) if prefetch and not boxes and depth < dims.z else None
    launched = Event() if pool is not None else None
    pending = None
    progress['prefetch'] = pool is not None
    try:
        completed(progress.copy())
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Истёк лимит времени уточнения winding.')
            started = time.monotonic()
            z = mesher.nextZ()
            size = mm.Vector3i(dims.x, dims.y, min(depth, dims.z-z))
            volume = mm.SimpleVolume()
            volume.dims, volume.voxelSize = size, spacing
            start = z + int(overlap is not None)
            end = z+size.z
            origin = first_center + mm.Vector3f(0, 0, start*step)
            new_size = mm.Vector3i(dims.x, dims.y, end-start)
            if pending is None:
                values = evaluate(fresh, new_size, origin, band if boxes else options)
                field_time = time.monotonic()-started
            else:
                values, field_time = pending.result()
                pending = None
            copying = time.monotonic()
            progress['unique_grid_samples'] += values.size
            if boxes:
                # Native distance screening returns NaN outside the band before
                # running the expensive winding traversal. Restore the baseline's
                # clamped distance magnitudes on every retained grid sample.
                progress['winding_samples'] += int(np.count_nonzero(np.isfinite(values)))
                np.clip(values, -cap, cap, out=values)
                for (x0, y0, z0), (x1, y1, z1) in boxes:
                    lo, hi = max(z0, start), min(z1, end)
                    if lo >= hi:
                        continue
                    region = values[lo-start:hi-start, y0:y1, x0:x1]
                    missing = np.isnan(region)
                    if missing.any():
                        box_origin = origin + mm.Vector3f(x0*step, y0*step, (lo-start)*step)
                        field = evaluate(extra, mm.Vector3i(x1-x0, y1-y0, hi-lo), box_origin, options)
                        np.copyto(region, field, where=missing)
                        progress['hole_box_samples'] += field.size
                        progress['winding_samples'] += field.size
                        del field
            else:
                progress['winding_samples'] += values.size
            data = volume.data.vec_
            data.resize(plane*size.z)
            slab = _float_view(data, (size.z, size.y, size.x))
            if overlap is not None:
                slab[0] = overlap
            slab[start-z:] = values
            if probe_signs is not None:
                probe_signs[start:end] = values < 0
            overlap = values[-1].copy() if end < dims.z else None
            del values, slab
            progress['field_seconds'] += field_time + (time.monotonic()-copying)
            if pool is not None and end < dims.z:
                next_size = mm.Vector3i(dims.x, dims.y, min(depth-1, dims.z-end))
                next_origin = first_center + mm.Vector3f(0, 0, end*step)
                launched.clear()
                pending = pool.submit(next_field, next_size, next_origin)
                # addPart can hold the GIL. Let the worker enter the SDK first.
                launched.wait()
            started = time.monotonic()
            mesher.addPart(volume)
            progress['marching_cubes_seconds'] += time.monotonic()-started
            progress['completed_layers'] = end
            progress['completed_blocks'] += 1
            if end == dims.z:
                break
            completed(progress.copy())
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    started = time.monotonic()
    result = mm.Mesh.fromTriMesh(mesher.finalize())
    progress['mesh_build_seconds'] = time.monotonic()-started
    completed(progress.copy())
    if sharpen:
        detail = mm.SharpenMarchingCubesMeshSettings()
        detail.dims, detail.gridToMeshXf = dims, grid_to_mesh
        detail.voxelClamp = params.voxelClamp
        detail.minNewVertDev = step*params.minNewVertDev
        detail.maxNewRank2VertDev = step*params.maxNewRank2VertDev
        detail.maxNewRank3VertDev = step*params.maxNewRank3VertDev
        detail.maxOldVertPosCorrection = step*params.maxOldVertPosCorrection
        mm.sharpenMarchingCubesMesh(mm.MeshPart(mesh), result, voxel_per_face, detail)
    return result
