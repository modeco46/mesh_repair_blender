"""Bounded winding batches on a common grid, with native marching cubes."""
import ctypes
import math
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
    if not 0 < len(rims) <= 64:
        return None
    boxes = []
    area = 0.
    boundary_edges = 0
    for edge in rims:
        loop = mm.trackRightBoundaryLoop(mesh.topology, edge)
        boundary_edges += len(loop)
        if len(loop) < 2 or boundary_edges > 2048:
            return None
        points = np.array([list(mesh.points[mesh.topology.org(e)]) for e in loop])
        center = points.mean(axis=0)
        a = points-center
        area += .5*np.linalg.norm(np.cross(a, np.roll(a, -1, axis=0)), axis=1).sum()
        boxes.append((points.min(axis=0), points.max(axis=0)))
    # A single global area bound also covers overlapping and non-simple rims.
    # Two cell diagonals cover all corner samples used to triangulate a crossing.
    radius = math.sqrt(area/(4*math.pi*.125)) + 2*math.sqrt(3)*step
    boxes = np.asarray(boxes)
    boxes[:, 0] -= radius
    boxes[:, 1] += radius
    extent = np.asarray(list(mesh.computeBoundingBox().size())) + 2*step
    share = float(np.prod(boxes[:, 1]-boxes[:, 0], axis=1).sum()/np.prod(extent))
    if not np.isfinite(share) or share > .15:
        return None
    return boxes, {'boundary_loops': len(rims), 'boundary_edges': boundary_edges,
                   'fan_area_bound_mm2': float(area), 'guard_radius_mm': radius,
                   'box_volume_fraction_upper_bound': share}


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
                  boundary_boxes=None, probe_signs=None):
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
    completed(progress.copy())
    fresh, extra = mm.std_vector_float(), mm.std_vector_float()

    def evaluate(buffer, size, origin, settings):
        # The CUDA binding writes to res.data(); it does not resize the vector.
        buffer.resize(size.x*size.y*size.z)
        winding.calcFromGridWithDistances(buffer, size, mm.AffineXf3f(basis, origin), settings,
                                          mm.func_bool_from_float())
        return _float_view(buffer, (size.z, size.y, size.x))

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
        values = evaluate(fresh, new_size, origin, band if boxes else options)
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
        progress['field_seconds'] += time.monotonic()-started
        started = time.monotonic()
        mesher.addPart(volume)
        progress['marching_cubes_seconds'] += time.monotonic()-started
        progress['completed_layers'] = end
        progress['completed_blocks'] += 1
        if end == dims.z:
            break
        completed(progress.copy())
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
