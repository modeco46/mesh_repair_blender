"""Bounded local repairs for dense marching-cubes surfaces."""
import time
import numpy as np

from .reduction import triangle_upper_bounds


def relax_voxel_patch(mesh, info, collisions, mm, tolerance):
    """Edit an owned reconstruction's small collision patch, rolling back on failure."""
    from meshlib import mrmeshnumpy as mn
    from .engine import valid
    started = time.monotonic()
    region = mm.getIncidentVerts(mesh.topology, collisions)
    mm.expand(mesh.topology, region, 2)
    faces = mm.getIncidentFaces(mesh.topology, region)
    if faces.count() > 8192:
        return None, {'accepted': False, 'reason': 'patch_too_large'}
    # Include every neighbour used by relaxation. Only the selected central
    # vertices may move, including when a patch comprises several components.
    triangles = np.array([[int(v) for v in mesh.topology.getTriVerts(f)] for f in faces], dtype=np.int32)
    ids, local_faces = np.unique(triangles, return_inverse=True)
    vertices = np.array([list(mesh.points[mm.VertId(int(v))]) for v in ids], dtype=np.float32)
    patch = mn.meshFromFacesVerts(local_faces.reshape(-1, 3).astype(np.int32), vertices)
    if patch.topology.numValidFaces() != len(triangles) or not np.array_equal(mn.getNumpyVerts(patch), vertices):
        return None, {'accepted': False, 'reason': 'patch_mapping_changed'}
    selected = np.array([region.test(mm.VertId(int(v))) for v in ids])
    params = mm.MeshRelaxParams()
    params.region = mn.vertBitSetFromBools(selected)
    params.iterations, params.force = 20, .25
    params.limitNearInitial, params.maxInitialDist = True, .98*tolerance
    mm.relax(patch, params)
    original = vertices[selected]
    moved = mn.getNumpyVerts(patch)[selected]
    # Corresponding points on each triangle have identical barycentric weights.
    # No point on either surface can move farther than its most displaced vertex.
    maximum = float(np.linalg.norm(moved.astype(float)-original, axis=1).max())
    margin = 256*np.finfo(float).eps*max(1., float(np.abs(original).max()), float(np.abs(moved).max()))
    proof = {'verified': maximum+margin <= tolerance, 'upper_bound_mm': maximum+margin, 'tolerance_mm': tolerance,
             'method': 'unchanged_topology_vertex_displacement', 'max_vertex_displacement_mm': maximum}
    item = {'attempted': True, 'accepted': False, 'method': 'bounded_patch_relax', 'verification': proof,
            'initial_intersecting_faces': collisions.count(), 'patch_faces': len(triangles), 'tolerance_mm': tolerance}
    if not proof['verified']:
        return None, item
    targets = [mm.VertId(int(v)) for v in ids[selected]]

    def apply(points):
        for vertex, point in zip(targets, points):
            mesh.points[vertex] = mm.Vector3f(*point)
        mesh.updateCaches(region)

    try:
        apply(moved)
        hits = None
        if info['triangles'] >= 1_000_000:
            from .collisions import patch_neighbours, region_collisions
            # All original collisions are in faces. The only new intersections
            # can involve faces incident to vertices explicitly edited above.
            coords = vertices.astype(float)
            coords[selected] = moved
            try:
                nearby = patch_neighbours(mesh, coords[local_faces.reshape(-1, 3)], faces, mm)
                if nearby is not None:
                    hits = region_collisions(mesh, nearby, mm)
            except (RuntimeError, MemoryError):
                pass
        item['collision_check'] = 'changed_faces_and_spatial_neighbours' if hits is not None else 'whole_mesh'
        if hits is None:
            hits = mm.findSelfCollidingTrianglesBS(mm.MeshPart(mesh))
        remaining = hits.count()
        after = dict(info, self_intersecting_faces=remaining, signed_volume_mm3=mesh.volume())
        item['after'] = after
        if valid(after):
            item['accepted'] = True
            return after, item
        return None, item
    finally:
        if not item['accepted']:
            apply(original)
        item['elapsed_seconds'] = time.monotonic()-started


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
