from types import SimpleNamespace

import pytest

from mesh_repair_blender.bridge import mm_per_blender_unit


def scene(scale):
    return SimpleNamespace(unit_settings=SimpleNamespace(scale_length=scale))


def test_unit_conversion():
    assert mm_per_blender_unit(scene(1), "MILLIMETERS") == 1
    assert mm_per_blender_unit(scene(1), "METERS") == 1000
    assert mm_per_blender_unit(scene(.001), "SCENE") == 1


@pytest.mark.parametrize("scale", [0, -1, float("nan")])
def test_invalid_scene_scale(scale):
    with pytest.raises(ValueError):
        mm_per_blender_unit(scene(scale), "SCENE")
