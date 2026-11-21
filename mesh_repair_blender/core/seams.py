"""Close only tiny cracks produced by marching cubes, with a local surface bound."""
import numpy as np

from .reduction import triangle_upper_bounds


def close_voxel_handle(mesh, prepared, current, voxel_reference, mm, q, report):
    """A single new handle on a genus-zero body can be a marching-cubes seam."""
    from .engine import diagnostics, level_set_distances, same_closed_surface_topology, valid
    from .reduction import verify_surface_bound
    if not (prepared['closed'] and prepared['topology_valid'] and prepared['components'] == current['components'] == 1
            and 2*prepared['vertices']-prepared['triangles'] == 4 and 2*current['vertices'] == current['triangles']):
        return mesh, current
    options = mm.DetectTunnelSettings()
    options.maxTunnelLength = 4*q
    item = report['voxel_handle'] = {'accepted': False, 'max_loop_length_mm': options.maxTunnelLength}
    old = mm.detectTunnelFaces(mm.MeshPart(mesh), options)
    if not 0 < old.count() <= 64:
        item['reason'] = 'no_subvoxel_handle'
        return mesh, current
    candidate = mm.Mesh(mesh)
    fill = mm.FillHoleNicelySettings()
    fill.triangulateOnly = True
    new = mm.patchMesh(candidate, old, fill)
    if not 0 < new.count() <= 64:
        item['reason'] = 'unexpected_patch_size'
        return mesh, current
    proof = verify_surface_bound(mesh.cloneRegion(old), candidate.cloneRegion(new), q, max_checks=16000,
                                 acceleration=report.get('acceleration'))
    item.update(removed_faces=old.count(), added_faces=new.count(), verification=proof)
    if not proof['verified']:
        item['reason'] = 'unconfirmed_surface_distance'
        return mesh, current
    candidate.pack()
    checked = diagnostics(candidate, mm)
    if not valid(checked) or not same_closed_surface_topology(prepared, checked):
        item['reason'] = 'invalid_surface_or_genus'
        return mesh, current
    item['voxel_reference'] = level_set_distances(voxel_reference, candidate, mm, q)
    if not item['voxel_reference']['within_step']:
        item['reason'] = 'voxel_reference_distance'
        return mesh, current
    item['accepted'] = True
    return candidate, checked


def fill_voxel_seams(mesh, mm, tolerance):
    """Keep every existing face/vertex; bound each new triangle to an old rim edge.

    Distance to one segment is convex. If all three vertices of a new triangle
    are within tolerance of the SAME old boundary edge, so is its entire surface.
    The reverse distance is zero because none of the old surface is removed.
    A large real hole or an uncertain bound rejects the candidate transaction.
    """
    from meshlib import mrmeshnumpy as mn
    item = {'accepted': False, 'tolerance_mm': tolerance, 'details': []}
    loops = list(mesh.topology.findHoleRepresentiveEdges())
    item['holes_before'] = len(loops)
    if not 0 < len(loops) <= 200:
        item['reason'] = 'seam_hole_count_limit'
        return mesh, item
    candidate = mm.Mesh(mesh)

    def point(vert):
        v = candidate.points[vert]
        return (v.x, v.y, v.z)

    maximum = 0.0
    try:
        for edge in loops:
            if candidate.topology.left(edge).valid():
                continue
            loop = mm.trackRightBoundaryLoop(candidate.topology, edge)
            ids = [candidate.topology.org(e) for e in loop]
            if not 3 <= len(ids) <= 64 or len({int(v) for v in ids}) != len(ids):
                item['reason'] = 'non_simple_seam'
                return mesh, item
            rim = np.asarray([point(v) for v in ids], dtype=np.float64)
            patch = mm.FaceBitSet()
            params = mm.FillHoleParams()
            params.outNewFaces = patch
            params.makeDegenerateBand = False
            mm.fillHole(candidate, edge, params)
            if not patch.count() or patch.count() > 128:
                item['reason'] = 'seam_patch_limit'
                return mesh, item
            triangles = np.asarray([[point(v) for v in candidate.topology.getTriVerts(f)]
                                    for f in patch], dtype=np.float64)
            bounds = np.full(len(triangles), np.inf)
            for i, a in enumerate(rim):
                b = rim[(i+1) % len(rim)]
                segments = np.broadcast_to(np.stack((a, b, a)), triangles.shape)
                bounds = np.minimum(bounds, triangle_upper_bounds(triangles, segments))
            margin = 256*np.finfo(float).eps*max(1., float(np.abs(rim).max()), float(np.abs(triangles).max()))
            bound = float(bounds.max()) + margin
            item['details'].append({'boundary_edges': len(ids), 'patch_faces': patch.count(),
                                    'upper_bound_mm': bound})
            if not np.isfinite(bound) or bound > tolerance:
                item['reason'] = 'seam_exceeds_tolerance'
                return mesh, item
            maximum = max(maximum, bound)
        # Check the contract independently: old geometry must remain unchanged.
        before_v, after_v = mn.getNumpyVerts(mesh), mn.getNumpyVerts(candidate)
        before_f, after_f = mn.getNumpyFaces(mesh.topology), mn.getNumpyFaces(candidate.topology)
        valid_faces = mn.getNumpyBitSet(mesh.topology.getValidFaces())[:len(before_f)]
        if (len(after_v) < len(before_v) or len(after_f) < len(before_f)
                or not np.array_equal(before_v, after_v[:len(before_v)])
                or not np.array_equal(before_f[valid_faces], after_f[:len(before_f)][valid_faces])):
            item['reason'] = 'seam_changed_existing_geometry'
            return mesh, item
        if not candidate.topology.isClosed() or not candidate.topology.checkValidity():
            item['reason'] = 'seam_topology_invalid'
            return mesh, item
        candidate.invalidateCaches()
        item.update(accepted=True, holes_after=0, verification={
            'verified': True, 'method': 'new_triangles_to_unchanged_boundary_segments',
            'upper_bound_mm': maximum, 'tolerance_mm': tolerance})
        return candidate, item
    except (RuntimeError, MemoryError) as exc:
        item.update(reason='seam_native_error', detail=str(exc))
        return mesh, item
