"""Transactional, bounded filling of simple boundary loops before voxelization."""
import time
import numpy as np


def patch_collisions(before, candidate, patch, mm):
    """Check changed faces and their spatial neighbours, including moved rim faces.

    The region contains every triangle whose box overlaps the box of the edits.
    Thus any new intersection has both operands in the region. Using the SDK's
    self-collision query preserves its handling of shared edges/vertices.
    """
    from meshlib import mrmeshnumpy as mn
    old_v, new_v = mn.getNumpyVerts(before), mn.getNumpyVerts(candidate)
    old_f, new_f = mn.getNumpyFaces(before.topology), mn.getNumpyFaces(candidate.topology)
    changed_v = np.ones(len(new_v), dtype=bool)
    n = min(len(old_v), len(new_v))
    changed_v[:n] = np.any(old_v[:n] != new_v[:n], axis=1)
    changed_f = np.ones(len(new_f), dtype=bool)
    n = min(len(old_f), len(new_f))
    changed_f[:n] = np.any(old_f[:n] != new_f[:n], axis=1)
    changed_f |= np.any(changed_v[new_f], axis=1)
    affected = (mn.faceBitSetFromBools(changed_f) | patch) & candidate.topology.getValidFaces()
    ids = np.fromiter((int(f) for f in affected), dtype=np.int32, count=affected.count())
    if not len(ids):
        return mm.FaceBitSet()
    points = new_v[new_f[ids]]
    lo, hi = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    margin = max(1., float(np.abs(points).max())) * np.finfo(np.float32).eps * 8
    nearby = np.empty(len(new_f), dtype=bool)
    for start in range(0, len(new_f), 65536):
        block = slice(start, start+65536)
        tri = new_v[new_f[block]]
        nearby[block] = np.all(tri.max(axis=1) >= lo-margin, axis=1) & np.all(tri.min(axis=1) <= hi+margin, axis=1)
    region = mn.faceBitSetFromBools(nearby) & candidate.topology.getValidFaces()
    return mm.findSelfCollidingTrianglesBS(mm.MeshPart(candidate, region)) & affected


def planar_boundary(mesh, mm, loop, points, quality_mm):
    # A latitude on a sphere is also planar. Look for a supporting plane or
    # a short rim bounded entirely by perpendicular side walls (e.g. a box lid).
    centered = points-points.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 2 or singular[1] < 1e-10:
        return False
    normal = axes[-1]
    if np.max(np.abs(centered @ normal)) > max(1e-6, quality_mm*.001):
        return False
    perpendicular_walls = True
    for edge in loop:
        face = mesh.topology.right(edge)
        if face.valid():
            n = mesh.normal(face)
            values = np.array([n.x, n.y, n.z], float)
            length = np.linalg.norm(values)
            if length and abs(values @ normal)/length > 1-1e-6:
                return True
            if not length or abs(values @ normal)/length > 1e-5:
                perpendicular_walls = False
        else:
            perpendicular_walls = False
    return perpendicular_walls


def coplanar_groups(mesh, loops, mm, quality, max_edges):
    """Find rims on one supported plane of the same body, e.g. the two lips of a tube."""
    from meshlib import mrmeshnumpy as mn
    if not 2 <= len(loops) <= 32 or mm.MeshComponents.getNumComponents(mm.MeshPart(mesh)) != 1:
        return []
    vertices = mn.getNumpyVerts(mesh)
    groups = []
    for index, edge in enumerate(loops):
        loop = mm.trackRightBoundaryLoop(mesh.topology, edge)
        if not 3 <= len(loop) <= max_edges:
            continue
        ids = [int(mesh.topology.org(e)) for e in loop]
        points = vertices[ids]
        if len(set(ids)) != len(ids) or not planar_boundary(mesh, mm, loop, points, quality):
            continue
        # Equal supporting planes, not merely parallel planes on opposite sides.
        origin = points.mean(axis=0)
        centered = points-origin
        normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
        precision = min(quality*.001, 4*np.finfo(np.float32).eps*max(1., np.abs(points).max()))
        area_vector = .5*np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)
        for group in groups:
            if (abs(normal @ group['normal']) > 1-1e-8 and
                    np.max(np.abs((points-group['origin']) @ group['normal'])) <= precision):
                break
        else:
            group = {'origin': origin, 'normal': normal, 'indices': [], 'area_vector': np.zeros(3)}
            groups.append(group)
        group['indices'].append(index)
        group['area_vector'] += area_vector
    return [group for group in groups if len(group['indices']) > 1]


def fill_planar_group(mesh, edges, expected_area, max_faces, mm):
    from meshlib import mrmeshnumpy as mn
    candidate = mm.Mesh(mesh)
    mm.fillContours2D(candidate, mm.std_vector_Id_EdgeTag(edges))
    patch = candidate.topology.getValidFaces() - mesh.topology.getValidFaces()
    # Opposite rim windings subtract the bore area. Capping each ring with a
    # separate disk would add those areas and turn a through-hole into a plug.
    if (not 0 < patch.count() <= max_faces or expected_area <= 0 or
            not np.isclose(candidate.area(patch), expected_area, rtol=2e-5, atol=0) or
            candidate.topology.findNumHoles() != mesh.topology.findNumHoles()-len(edges) or
            not candidate.topology.checkValidity() or
            not np.array_equal(mn.getNumpyVerts(candidate), mn.getNumpyVerts(mesh)) or
            patch_collisions(mesh, candidate, patch, mm).count()):
        raise ValueError('Площадь, границы или пересечения совместной заплатки не прошли проверку.')
    return candidate, patch.count()


def fill_simple_holes(mesh, mm, quality_mm, progress=lambda value: None, emit=lambda text: None,
                      max_diameter_mm=10.0, max_boundary_edges=2048,
                      max_patch_faces=50000, time_budget_seconds=60.0,
                      defer_intersections=False):
    """Defer colliding patches only when the caller owns repair/rollback."""
    from meshlib import mrmeshnumpy as mn
    started = time.monotonic()
    work = mesh  # Every patch is tried on a private copy below.
    loops = list(mesh.topology.findHoleRepresentiveEdges())
    report = {'holes_before': len(loops), 'filled': 0, 'rejected': 0, 'skipped': 0,
              'max_diameter_mm': max_diameter_mm, 'max_boundary_edges': max_boundary_edges,
              'max_patch_faces': max_patch_faces, 'time_budget_seconds': time_budget_seconds,
              'pending_local_fix': 0, 'accepted_with_local_fix': 0,
              'patch_self_intersecting_faces': 0, 'details': []}
    target_edge = max(2*quality_mm, .05)
    grouped = set()
    for group in coplanar_groups(mesh, loops, mm, quality_mm, max_boundary_edges):
        indices = group['indices']
        grouped.update(indices)  # Never replace a failed ring-shaped cap with separate disks.
        area = float(np.linalg.norm(group['area_vector']))
        item = {'holes': [i+1 for i in indices], 'method': 'joint_planar_contours', 'expected_area_mm2': area}
        report['details'].append(item)
        if time.monotonic()-started >= time_budget_seconds:
            report['skipped'] += len(indices)
            item['reason'] = 'hole_fill_time_budget'
            continue
        try:
            work, count = fill_planar_group(work, [loops[i] for i in indices], area, max_patch_faces, mm)
            report['filled'] += len(indices)
            item.update(accepted=True, patch_triangles=count)
        except (RuntimeError, ValueError) as exc:
            item.update(reason='planar_group_rejected', detail=str(exc))
            report['rejected'] += len(indices)
    for index, edge in enumerate(loops):
        if index in grouped:
            continue
        progress(index/max(1, len(loops)))
        item = {'hole': index+1}
        report['details'].append(item)
        if time.monotonic()-started >= time_budget_seconds:
            report['skipped'] += sum(i not in grouped for i in range(index, len(loops)))
            item['reason'] = 'hole_fill_time_budget'
            break
        # Accepted edits do not pack/reindex the mesh. A previous patch may have
        # closed a touching loop too; check its representative before proceeding.
        if work.topology.left(edge).valid():
            item['reason'] = 'already_closed'
            continue
        loop = mm.trackRightBoundaryLoop(work.topology, edge)
        if len(loop) > max_boundary_edges:
            item.update(reason='hole_size_limit', boundary_edges=len(loop))
            report['skipped'] += 1
            continue
        ids = [work.topology.org(e) for e in loop]
        if len(ids) < 3 or len(set(map(int, ids))) != len(ids):
            item['reason'] = 'non_simple_boundary'
            report['skipped'] += 1
            continue
        pts = np.asarray([list(work.points[v]) for v in ids], dtype=float)
        diameter = float(np.linalg.norm(np.ptp(pts, axis=0)))
        flat = planar_boundary(work, mm, loop, pts, quality_mm)
        estimated = len(ids)-2 if flat else int(np.ceil(diameter**2 / target_edge**2 * 2))
        item.update(boundary_edges=len(ids), diameter_mm=diameter, estimated_patch_faces=estimated)
        if (diameter > max_diameter_mm and not (flat and len(ids) <= 30)) or estimated > max_patch_faces:
            item['reason'] = 'hole_size_limit'
            report['skipped'] += 1
            continue
        attempt = time.monotonic()
        try:
            # This copy also rolls back modified boundary triangles/coordinates,
            # not merely newly created faces. No partial failed patch can leak.
            candidate = mm.Mesh(work)
            before_holes = candidate.topology.findNumHoles()
            settings = mm.FillHoleNicelySettings()
            settings.triangulateOnly = flat
            item['method'] = 'planar_triangulation' if flat else 'curvature_patch'
            settings.subdivideSettings.maxEdgeLen = target_edge
            settings.subdivideSettings.maxEdgeSplits = min(20000, max_patch_faces//2)
            settings.smoothCurvature = True
            patch = mm.fillHoleNicely(candidate, edge, settings)
            remaining = candidate.topology.findNumHoles()
            if (not patch.count() or remaining >= before_holes or not candidate.topology.checkValidity()
                    or not np.isfinite(mn.getNumpyVerts(candidate)).all()):
                item['reason'] = 'patch_not_closed'
            elif patch.count() > max_patch_faces:
                item['reason'] = 'patch_work_limit'
            else:
                collisions = patch_collisions(work, candidate, patch, mm).count()
                if collisions and not flat:
                    # Smoothing may push a lid through a nearby wall. Try the
                    # original rim again without moving or subdividing it.
                    alternative = mm.Mesh(work)
                    rim = mm.FillHoleNicelySettings()
                    rim.triangulateOnly = True
                    metric = mm.FillHoleParams()
                    metric.metric = mm.getUniversalMetric(alternative)
                    rim.triangulateParams = metric
                    item['curvature_patch_intersections'] = collisions
                    try:
                        cap = mm.fillHoleNicely(alternative, edge, rim)
                        if (cap.count() == len(ids)-2 and alternative.topology.findNumHoles() == before_holes-1
                                and alternative.topology.checkValidity()
                                and np.array_equal(mn.getNumpyVerts(alternative), mn.getNumpyVerts(work))
                                and not patch_collisions(work, alternative, cap, mm).count()):
                            candidate, patch, collisions = alternative, cap, 0
                            remaining = before_holes-1
                            item['method'] = 'fixed_rim_triangulation'
                    except RuntimeError as exc:
                        item['fixed_rim_error'] = str(exc)
                # A planar cap must be a clean triangulation. Crossing a second
                # rim or another part is not a reason to cut the original walls.
                if collisions:
                    item.update(reason='patch_self_intersections', self_intersecting_faces=collisions)
                    report['patch_self_intersecting_faces'] += collisions
                if not collisions or (defer_intersections and not flat):
                    work = candidate
                    item.update(accepted=True, patch_triangles=patch.count())
                    report['filled'] += before_holes-remaining
                    if collisions:
                        item['pending_local_fix'] = True
                        report['pending_local_fix'] += 1
        except RuntimeError as exc:
            item.update(reason='patch_native_error', detail=str(exc))
        if not item.get('accepted'):
            report['rejected'] += 1
        item['elapsed_seconds'] = time.monotonic()-attempt
    report.update(holes_after=work.topology.findNumHoles(), closed_after=work.topology.isClosed())
    # Only after the lid exists is signed volume a reliable orientation test
    # for an originally open body. All accepted patches own a private mesh.
    if report['filled'] and report['closed_after']:
        from .orientation import orient_closed_components
        report['closed_orientation_corrections'] = orient_closed_components(work, mm)
    report['elapsed_seconds'] = time.monotonic()-started
    progress(1)
    if loops:
        emit(f"Заполнение дыр: {len(loops)} → {report['holes_after']}; принято {report['filled']}, "
             f"отклонено {report['rejected']}, пропущено {report['skipped']}; {report['elapsed_seconds']:.2f} с.")
    return work, report
