"""Result-vertex to original-surface distances, computed only on request."""
import time

import numpy as np


def measure(source, vertices_path, output, disable_cuda=False, progress=lambda event: None):
    from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
    from core.compute import Projector
    from core.reduction import projection_arrays

    started = time.monotonic()
    original = mm.loadMesh(source)
    if not original.topology.numValidFaces():
        raise ValueError("Исходник не содержит треугольников.")
    vertices = np.load(vertices_path, mmap_mode="r", allow_pickle=False)
    acceleration = {"cuda_disabled": disable_cuda}
    projector = Projector(original, mm, acceleration)
    distances = np.empty(len(vertices), dtype=np.float32)
    for start in range(0, len(vertices), 131072):
        chunk = np.ascontiguousarray(vertices[start:start + 131072], dtype=np.float32)
        projections = mm.std_vector_MeshProjectionResult()
        projector.findProjections(projections, mn.fromNumpyArray(chunk), None, None,
                                  float(np.finfo(np.float32).max), 0)
        ids, squared = projection_arrays(projections, mm)
        if len(squared) != len(chunk) or (ids < 0).any() or not np.isfinite(squared).all():
            raise RuntimeError("Не удалось измерить отклонения от исходной поверхности.")
        distances[start:start + len(chunk)] = np.sqrt(np.maximum(squared, 0))
        progress({"percent": int(100 * (start + len(chunk)) / len(vertices)), "stage": "Карта отклонений"})
    np.save(output, distances, allow_pickle=False)
    return {"direction": "result_vertices_to_source_surface", "mode": "measured_not_proven",
            "max_mm": float(distances.max()), "elapsed_seconds": time.monotonic() - started,
            "acceleration": acceleration}
