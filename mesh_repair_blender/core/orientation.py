"""Exact welding and consistent winding, without resampling the surface.

A two-sheet parity graph solves orientation in native SciPy code. Unlike a Python
BFS per shell it is linear in graph size even for thousands of tiny components.
Non-manifold edges are not traversed. Open components never use signed volume.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def finite_vertices(verts):
    count = int(np.count_nonzero(~np.isfinite(verts).all(axis=1)))
    if count:
        raise ValueError(f'В координатах сетки обнаружены NaN/inf: {count} вершин. Исправьте координаты в исходном файле.')


def clean_triangles(verts, faces, *, keep_coincident_faces=False):
    """Weld exact coordinates, then remove null and repeated triangles before building adjacency."""
    finite_vertices(verts)
    unique, inverse = np.unique(verts, axis=0, return_inverse=True)
    welded = inverse.reshape(-1)[faces].astype(np.int32)
    keep = np.empty(len(welded), dtype=bool)
    for start in range(0, len(welded), 65536):
        block = slice(start, start + 65536)
        tri = unique[welded[block]].astype(np.float64)
        keep[block] = np.any(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) != 0, axis=1)
    degenerate = int(np.count_nonzero(~keep))
    if degenerate:
        welded = welded[keep]
    if not len(welded):
        raise ValueError('Все треугольники имеют нулевую площадь.')
    duplicates = 0
    if not keep_coincident_faces:
        keys = np.sort(welded, axis=1).view(np.dtype((np.void, 3 * welded.itemsize))).ravel()
        _, indices = np.unique(keys, return_index=True)
        duplicates = len(welded) - len(indices)
        if duplicates:
            welded = welded[np.sort(indices)]  # Retain the original winding and ordering.
    return unique, welded, degenerate, duplicates


def is_flat_input(vertices, faces):
    """Test represented geometry, using round-off only (independent of the voxel step)."""
    # Pick a well-conditioned reference triangle. All inputs here are packed.
    sample = faces[np.linspace(0, len(faces) - 1, min(32, len(faces)), dtype=np.intp)]
    tri = vertices[sample].astype(np.float64)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    index = int(np.argmax(lengths))
    if not lengths[index]:
        return False  # clean_triangles handles zero-area inputs.
    normal = normals[index] / lengths[index]
    origin = tri[index, 0]
    for start in range(0, len(vertices), 65536):
        delta = vertices[start:start + 65536].astype(np.float64) - origin
        # Dot-product rounding bound; does not scale with absolute scene position.
        rounding = 32 * np.finfo(float).eps * np.abs(delta).sum(axis=1)
        if np.any(np.abs(np.einsum('ij,j->i', delta, normal)) > rounding):
            return False
    return True


def edge_pairs(faces, nv):
    # Flatten edge-major to avoid a 3N x 2 intermediate plus a repeated face-id array.
    a = faces.T.reshape(-1)
    b = faces[:, [1, 2, 0]].T.reshape(-1)
    key = np.minimum(a, b).astype(np.int64) * nv + np.maximum(a, b)
    order = np.argsort(key, kind='stable')
    sorted_key = key[order]
    first = np.r_[0, np.flatnonzero(sorted_key[1:] != sorted_key[:-1]) + 1]
    count = np.diff(np.r_[first, len(order)])
    pairs = first[count == 2]
    x, y = order[pairs], order[pairs + 1]
    return x, y, a[x] == a[y], order[first[count != 2]], count


def orient_consistently(verts, faces):
    n = len(faces)
    if not n:
        raise ValueError('Сетка не содержит невырожденных треугольников.')
    x, y, bad, boundary, edge_counts = edge_pairs(faces, len(verts))
    fa, fb = x % n, y % n
    # Vertices i and i+n encode the two possible orientations of face i.
    # Connected components of this double cover encode all parity constraints.
    row = np.concatenate((fa, fa + n)).astype(np.int32)
    col = np.concatenate((fb + bad * n, fb + (~bad) * n)).astype(np.int32)
    graph = coo_matrix((np.ones(len(row), np.int8), (row, col)), shape=(2*n, 2*n)).tocsr()
    _, parity = connected_components(graph, directed=False)
    conflict = parity[:n] == parity[n:]
    # A non-orientable component joins the two sheets. Leave only those
    # faces untouched; the other components still have a valid parity solution.
    flip = (parity[:n] > parity[n:]) & ~conflict
    _, labels = np.unique(np.minimum(parity[:n], parity[n:]), return_inverse=True)
    labels = labels.astype(np.int32)
    ncomp = int(labels.max()) + 1
    closed = np.ones(ncomp, bool)
    unoriented = np.zeros(ncomp, bool)
    unoriented[labels[conflict]] = True
    closed[unoriented] = False
    closed[labels[boundary % n]] = False
    # An ambiguous edge can involve more than the first face: mark all touched
    # components open, so volume is never used to orient a non-manifold component.
    if np.any(edge_counts > 2):
        all_a = faces.T.reshape(-1)
        all_b = faces[:, [1, 2, 0]].T.reshape(-1)
        keys = np.minimum(all_a, all_b).astype(np.int64)*len(verts) + np.maximum(all_a, all_b)
        _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        closed[labels[np.flatnonzero(counts[inv] != 2) % n]] = False
    out = faces.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    volume = np.zeros(ncomp)
    area = np.zeros(ncomp)
    changed_area = np.zeros(ncomp)
    # Translation improves volume conditioning; closed volume is origin-independent.
    origin = (verts.min(axis=0).astype(float) + verts.max(axis=0)) * .5
    for start in range(0, n, 131072):
        sl = slice(start, start+131072)
        tri = verts[out[sl]].astype(np.float64) - origin
        normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        weight = np.linalg.norm(normal, axis=1)
        area += np.bincount(labels[sl], weights=weight, minlength=ncomp)
        changed_area += np.bincount(labels[sl], weights=weight*flip[sl], minlength=ncomp)
        volume += np.bincount(labels[sl], weights=np.einsum('ij,ij->i', tri[:, 0], normal)/6, minlength=ncomp)
    # Closed shells become positive solids; open patches keep the area-majority
    # input orientation (a heuristic, explicitly not an inside/outside proof).
    reverse = np.where(closed, volume < 0, changed_area > area*.5)
    mask = reverse[labels] & ~conflict
    out[mask] = out[mask][:, [0, 2, 1]]
    return out, {'accepted': True, 'partial': bool(conflict.any()),
                 'conflicting_faces': int(conflict.sum()),
                 'unoriented_components': int(unoriented.sum()),
                 'unoriented_component_faces': np.bincount(labels, minlength=ncomp)[unoriented].tolist(),
                 'disoriented_edges': int(bad.sum()),
                 'components': ncomp, 'closed_components': int(closed.sum()),
                 'open_components': int((~closed).sum()),
                 'nonmanifold_edges': int(np.count_nonzero(edge_counts > 2)),
                 'negative_closed_components': int(np.count_nonzero(closed & (volume < 0)))}


def prepare_orientation(mesh, mm, progress=lambda value: None, weld_tolerance=0):
    from meshlib import mrmeshnumpy as mn
    started = time.monotonic()
    mesh = mm.Mesh(mesh)
    mesh.pack()
    before = mm.MeshComponents.getNumComponents(mm.MeshPart(mesh))
    verts, faces = mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology)
    finite_vertices(verts)
    progress(.05)
    # Separate closed solids can share a complete interface. Removing just one
    # side opens a solid; the Boolean stage owns such interfaces.
    verts, faces, degenerate, duplicates = clean_triangles(
        verts, faces, keep_coincident_faces=before > 1 and mesh.topology.isClosed())
    progress(.3)
    faces, report = orient_consistently(verts, faces)
    report.update(components_before=before, components_after=before,
                  removed_degenerate_faces=0, removed_duplicate_faces=0, approximate_weld_mm=weld_tolerance)
    progress(.7)
    if report['accepted']:
        settings = mm.MeshBuilder.BuildSettings()
        skipped = mm.int_output()
        settings.skippedFaceCount = skipped
        candidate = mn.meshFromFacesVerts(faces, verts, settings)
        if skipped.value or candidate.topology.numValidFaces() != len(faces):
            report.update(accepted=False, reason='orientation_rebuild_dropped_faces')
        else:
            after = mm.MeshComponents.getNumComponents(mm.MeshPart(candidate))
            if after < before or report['disoriented_edges'] or report['negative_closed_components'] or degenerate or duplicates or (candidate.topology.isClosed() and not mesh.topology.isClosed()):
                mesh = candidate
                mesh.pack()
                report.update(removed_degenerate_faces=degenerate, removed_duplicate_faces=duplicates)
            if weld_tolerance:
                count_before = mesh.topology.numValidFaces()
                united = mm.uniteCloseVertices(mesh, weld_tolerance)
                report['approximate_weld_united_vertices'] = united
                if united:
                    mesh, followup = prepare_orientation(mesh, mm)
                    report['approximate_weld_followup'] = followup
                report['approximate_weld_removed_faces'] = count_before-mesh.topology.numValidFaces()
            report['components_after'] = mm.MeshComponents.getNumComponents(mm.MeshPart(mesh))
            report['closed_after'] = mesh.topology.isClosed()
    progress(1)
    report['elapsed_seconds'] = time.monotonic() - started
    return mesh, report


def orient_closed_components(mesh, mm):
    """Safety pass for winding reconstruction: never flip an open component."""
    from meshlib import mrmeshnumpy as mn
    mesh.pack()
    mapping, count = mm.MeshComponents.getAllComponentsMap(mm.MeshPart(mesh))
    labels = np.fromiter((int(x) for x in mapping), dtype=np.int32, count=mapping.size())
    closed = np.ones(count, bool)
    edges = mesh.topology.findLeftBdEdges()
    adjacent = np.fromiter((int(mesh.topology.right(e)) for e in edges), dtype=np.int32, count=edges.count())
    closed[labels[adjacent]] = False
    verts, faces = mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology)
    origin = (verts.min(axis=0).astype(float)+verts.max(axis=0))*.5
    volume = np.zeros(count)
    for start in range(0,len(faces),131072):
        sl = slice(start,start+131072)
        tri = verts[faces[sl]].astype(float)-origin
        values = np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]))/6
        volume += np.bincount(labels[sl], weights=values, minlength=count)
    negative = closed & (volume < 0)
    if negative.any():
        region = mn.faceBitSetFromBools(negative[labels])
        mesh.topology.flipOrientation(mm.getIncidentEdges(mesh.topology,region))
    return int(negative.sum())
