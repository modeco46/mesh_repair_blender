"""Fast Blender <-> MeshRepair geometry transfer without UI operators.

The worker consumes binary STL and returns the validated indexed mesh as two
uncompressed NPY arrays. A result vertex is identified by its index.
"""
from __future__ import annotations

from pathlib import Path
import struct

import numpy as np


_STL_RECORD = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
])


def mm_per_blender_unit(scene, mode: str) -> float:
    """Return the number of millimetres represented by one Blender unit."""
    if mode == "MILLIMETERS":
        return 1.0
    if mode == "METERS":
        return 1000.0
    if mode != "SCENE":
        raise ValueError(f"Unknown unit mode: {mode}")
    scale = float(scene.unit_settings.scale_length)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scene unit scale must be a positive finite number.")
    return scale * 1000.0


def _write_triangle_block(stream, vertices: np.ndarray, faces: np.ndarray, start: int, stop: int) -> None:
    triangle = np.asarray(vertices[faces[start:stop]], dtype=np.float32)
    edge_a = triangle[:, 1] - triangle[:, 0]
    edge_b = triangle[:, 2] - triangle[:, 0]
    normal = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normal, axis=1)
    good = lengths > 0
    normal[good] /= lengths[good, None]
    normal[~good] = 0
    records = np.zeros(stop - start, dtype=_STL_RECORD)
    records["normal"] = normal
    records["vertices"] = triangle
    stream.write(records.tobytes())


def write_selected_stl(context, destination, unit_mode: str, *, chunk_size: int = 262_144):
    """Write selected evaluated mesh objects in world space to binary STL.

    Modifiers and Geometry Nodes are evaluated.  Negative object scales flip
    triangle winding so the world-space orientation stays correct.
    """
    objects = [obj for obj in context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise ValueError("Select at least one mesh object.")
    factor = mm_per_blender_unit(context.scene, unit_mode)
    destination = Path(destination)
    total = 0
    depsgraph = context.evaluated_depsgraph_get()
    with destination.open("wb") as stream:
        stream.write(b"MeshRepair Blender".ljust(80, b"\0"))
        stream.write(struct.pack("<I", 0))
        for source in objects:
            evaluated = source.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
            try:
                mesh.calc_loop_triangles()
                triangle_count = len(mesh.loop_triangles)
                if not triangle_count:
                    continue
                vertices = np.empty((len(mesh.vertices), 3), dtype=np.float32)
                mesh.vertices.foreach_get("co", vertices.reshape(-1))
                matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
                vertices = vertices.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
                vertices = np.asarray(vertices * factor, dtype=np.float32)
                faces = np.empty((triangle_count, 3), dtype=np.int32)
                mesh.loop_triangles.foreach_get("vertices", faces.reshape(-1))
                if np.linalg.det(matrix[:3, :3]) < 0:
                    faces[:, [1, 2]] = faces[:, [2, 1]]
                for start in range(0, triangle_count, chunk_size):
                    _write_triangle_block(stream, vertices, faces, start, min(start + chunk_size, triangle_count))
                total += triangle_count
            finally:
                evaluated.to_mesh_clear()
        if not total:
            raise ValueError("The selected objects contain no triangles.")
        if total > 0xFFFFFFFF:
            raise ValueError("Binary STL cannot contain more than 4,294,967,295 triangles.")
        stream.seek(80)
        stream.write(struct.pack("<I", total))
    return {"objects": len(objects), "triangles": total, "mm_per_blender_unit": factor}


def create_mesh_object(context, vertices_path, faces_path, name: str, mm_per_unit: float):
    """Create a Blender object from worker arrays using bulk collection writes."""
    import bpy

    vertices = np.load(vertices_path, mmap_mode="r")
    faces = np.load(faces_path, mmap_mode="r")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("The worker returned invalid geometry arrays.")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("The worker returned an empty mesh.")
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(vertices))
    coordinates = np.asarray(vertices, dtype=np.float32) / float(mm_per_unit)
    mesh.vertices.foreach_set("co", coordinates.reshape(-1))
    loop_count = len(faces) * 3
    mesh.loops.add(loop_count)
    mesh.loops.foreach_set("vertex_index", np.asarray(faces, dtype=np.int32).reshape(-1))
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(0, loop_count, 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(len(faces), 3, dtype=np.int32))
    mesh.polygons.foreach_set("use_smooth", np.zeros(len(faces), dtype=np.bool_))
    mesh.update(calc_edges=True)
    if mesh.validate(clean_customdata=False):
        bpy.data.meshes.remove(mesh)
        raise ValueError("Blender изменил геометрию при проверке импортированной сетки.")
    result = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(result)
    for obj in context.selected_objects:
        obj.select_set(False)
    result.select_set(True)
    context.view_layer.objects.active = result
    return result
