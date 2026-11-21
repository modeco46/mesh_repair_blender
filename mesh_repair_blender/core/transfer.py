"""Deliver a checked native mesh to Blender with its indexed topology intact."""
import numpy as np


def write_indexed_mesh(mesh, paths):
    from meshlib import mrmeshnumpy as mn

    mesh.pack()
    # Native Vector3f coordinates are promoted to float64 by the NumPy binding.
    vertices = np.asarray(mn.getNumpyVerts(mesh), dtype=np.float32)
    faces = mn.getNumpyFaces(mesh.topology)
    created = []
    try:
        for path, data in zip(paths, (vertices, faces)):
            # These job-private files become visible to Blender via the atomic
            # worker result message only after both have been verified.
            with path.open('xb') as stream:
                created.append(path)
                np.save(stream, data, allow_pickle=False)
            restored = np.load(path, mmap_mode='r', allow_pickle=False)
            try:
                if restored.shape != data.shape or restored.dtype != data.dtype:
                    raise OSError('Изменился формат записанной сетки.')
                for start in range(0, len(data), 131072):
                    block = slice(start, start + 131072)
                    if not np.array_equal(restored[block], data[block]):
                        raise OSError('Записанные координаты или индексы отличаются от проверенной сетки.')
            finally:
                del restored  # Close the Windows mapping before rollback.
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {'format': 'indexed_arrays', 'exact_roundtrip': True, 'vertices': len(vertices), 'triangles': len(faces)}
