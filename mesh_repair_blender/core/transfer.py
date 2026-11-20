"""Lossless indexed-mesh transport for Blender, avoiding STL vertex welding."""
from pathlib import Path
import os
import numpy as np


def save_mesh_arrays(mesh, paths):
    from meshlib import mrmeshnumpy as mn
    mesh.pack()
    arrays = (np.asarray(mn.getNumpyVerts(mesh), dtype=np.float32),
              np.asarray(mn.getNumpyFaces(mesh.topology), dtype=np.int32))
    for path, data in zip(paths, arrays):
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.part')
        try:
            with temporary.open('wb') as stream:
                np.save(stream, data, allow_pickle=False)
            restored = np.load(temporary, mmap_mode='r', allow_pickle=False)
            try:
                if restored.shape != data.shape or restored.dtype != data.dtype:
                    raise RuntimeError('Изменился формат массива сетки при записи.')
                for start in range(0, len(data), 262144):
                    if not np.array_equal(data[start:start+262144], restored[start:start+262144]):
                        raise RuntimeError('Массив сетки не прошёл проверку записи.')
            finally:
                del restored  # Release the Windows mmap before rename/unlink.
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return {'format': 'indexed_numpy_arrays', 'verified': True,
            'verification': 'exact_coordinate_and_index_roundtrip',
            'stl_vertex_welding': False, 'vertices': len(arrays[0]), 'triangles': len(arrays[1])}
