"""Bulk transfer of native component labels, with a checked SDK layout."""
import ctypes
import numpy as np


def region_labels(mapping):
    """Return owned int32 labels; never retain a view into a temporary C++ vector.

    The pinned SDK exposes element size and a data pointer. Unknown layouts use
    its public iterator instead. Reading millions of RegionId Python objects was
    more expensive than the native component search itself.
    """
    count = mapping.size()
    if not count:
        return np.empty(0, dtype=np.int32)
    try:
        vector = mapping.vec_
        if vector.element_type_byte_size == 4 and vector.data_pointer():
            buffer = (ctypes.c_int32 * count).from_address(vector.data_pointer())
            view = np.ctypeslib.as_array(buffer)
            if all(int(view[i]) == int(vector[i]) for i in {0, count//2, count-1}):
                return view.copy()
    except (AttributeError, TypeError):
        pass
    return np.fromiter((int(value) for value in mapping), dtype=np.int32, count=count)


def component_labels(mesh, mm):
    mapping, count = mm.MeshComponents.getAllComponentsMap(mm.MeshPart(mesh))
    return region_labels(mapping), count
