"""Persistent per-vertex distances and reversible Solid viewport colouring."""
import numpy as np

DISTANCE = "MeshRepair deviation (mm)"
COLOR = "MeshRepair deviation colors"
_VIEW = None


def available(obj):
    return bool(obj and obj.type == "MESH" and DISTANCE in obj.data.attributes and COLOR in obj.data.color_attributes)


def active(obj=None):
    return _VIEW is not None and (obj is None or _VIEW["object"] == obj)


def colors(distances, quality):
    # Linear RGB: blue -> cyan -> green (tolerance) -> yellow -> red (2*tolerance).
    palette = np.array([[.015, .08, .8], [0, .7, 1], [.03, .8, .06], [1, .75, 0], [.95, .015, .005]])
    t = np.clip(np.asarray(distances) / (2 * quality), 0, 1)
    rgba = np.ones((len(t), 4), dtype=np.float32)
    for channel in range(3):
        rgba[:, channel] = np.interp(t, np.linspace(0, 1, 5), palette[:, channel])
    return rgba


def attach(obj, values, quality):
    values = np.asarray(values, dtype=np.float32)
    if len(values) != len(obj.data.vertices) or not np.isfinite(values).all():
        raise ValueError("Геометрия результата изменилась. Выполните ремонт заново.")
    for name in (DISTANCE, COLOR):
        existing = obj.data.attributes.get(name)
        if existing:
            obj.data.attributes.remove(existing)
    attribute = obj.data.attributes.new(DISTANCE, "FLOAT", "POINT")
    attribute.data.foreach_set("value", values)
    attribute = obj.data.color_attributes.new(name=COLOR, type="FLOAT_COLOR", domain="POINT")
    attribute.data.foreach_set("color", colors(values, quality).ravel())
    obj["mesh_repair_map_quality_mm"] = quality
    obj["mesh_repair_map_max_mm"] = float(values.max())
    obj.data.update()


def hide():
    global _VIEW
    if _VIEW is None:
        return
    for space, settings in _VIEW["spaces"]:
        try:
            for key, value in settings.items():
                setattr(space.shading, key, value)
        except ReferenceError:
            pass
    try:
        obj = _VIEW["object"]
        previous = obj.data.color_attributes.get(_VIEW["color"])
        if previous:
            obj.data.color_attributes.active_color = previous
    except ReferenceError:
        pass
    _VIEW = None


def show(context, obj):
    global _VIEW
    hide()
    if not available(obj):
        raise ValueError("У объекта нет карты отклонений.")
    previous = obj.data.color_attributes.active_color
    state = {"object": obj, "color": previous.name if previous else "", "spaces": []}
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                shading = space.shading
                settings = {key: getattr(shading, key) for key in ("type", "color_type", "light", "show_shadows", "show_cavity")}
                state["spaces"].append((space, settings))
                shading.type, shading.color_type, shading.light = "SOLID", "VERTEX", "FLAT"
                shading.show_shadows = shading.show_cavity = False
                area.tag_redraw()
    obj.data.color_attributes.active_color = obj.data.color_attributes.get(COLOR)
    _VIEW = state
