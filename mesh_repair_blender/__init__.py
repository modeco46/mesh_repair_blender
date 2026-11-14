"""MeshRepair for Blender 5.3."""

bl_info = {
    "name": "Mesh Repair",
    "author": "modeco46",
    "version": (1, 1, 0),
    "blender": (5, 3, 0),
    "location": "3D Viewport > Sidebar > Mesh Repair",
    "description": "Jewelry mesh repair with native CPU parallelism and CUDA",
    "category": "Mesh",
    "doc_url": "https://github.com/modeco46/mesh_repair_blender",
}


def register():
    from .addon import register as _register
    _register()


def unregister():
    from .addon import unregister as _unregister
    _unregister()
