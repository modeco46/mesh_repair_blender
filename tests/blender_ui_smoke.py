"""Exercise panel layout and modal completion in Blender without a GUI event loop."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import bpy


class ContextProxy:
    def __init__(self):
        self.window_manager = WindowManagerProxy()

    def __getattr__(self, name):
        return getattr(bpy.context, name)


class WindowManagerProxy:
    def modal_handler_add(self, operator):
        pass  # Background Blender has no UI event loop; TIMER is delivered below.

    def __getattr__(self, name):
        return getattr(bpy.context.window_manager, name)


class Layout:
    def __init__(self, events):
        self.events = events

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.events.append((name, args, kwargs))
            return Layout(self.events)
        return record


def main():
    root = Path(os.environ.get("MESH_REPAIR_ADDON_ROOT", Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(root))
    import mesh_repair_blender
    from mesh_repair_blender import addon, heatmap

    mesh_repair_blender.register()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.unit_settings.scale_length = .001
    bpy.ops.mesh.primitive_cube_add(size=2)
    source = bpy.context.active_object
    bpy.context.scene.mesh_repair_settings.disable_cuda = True
    context = ContextProxy()
    operator = SimpleNamespace(report=lambda *args: print(args))
    for operation in ("check", "repair", "heatmap"):
        assert addon._begin(context, operation, operator) == {"RUNNING_MODAL"}
        job = addon._ACTIVE
        assert job["process"].wait(timeout=60) == 0
        modal = addon._MeshRepairModal()
        modal._mesh_repair_job_id = job["id"]
        # A real Blender Event does not expose the originating timer.
        assert modal.modal(context, SimpleNamespace(type="TIMER")) == {"FINISHED"}
        runtime = context.window_manager.mesh_repair_runtime
        assert not runtime.busy and not runtime.stage.startswith("Ошибка"), runtime.message
    result = runtime.result_object
    assert result != source and source.hide_get()
    assert not any(face.use_smooth for face in result.data.polygons)
    assert heatmap.available(result) and heatmap.active(result)
    assert abs(result.dimensions.x - 2) < 1e-5
    events = []
    addon.MESHREPAIR_PT_panel.draw(SimpleNamespace(layout=Layout(events)), context)
    properties = [args[1] for name, args, _ in events if name == "prop"]
    assert properties == ["quality_mm", "show_advanced"], properties
    operators = [args[0] for name, args, _ in events if name == "operator"]
    assert operators[0] == "mesh_repair.check_selected"
    assert "mesh_repair.save_report" not in operators
    assert operators.index("mesh_repair.repair_selected") > 0
    context.scene.mesh_repair_settings.show_advanced = True
    events.clear()
    addon.MESHREPAIR_PT_panel.draw(SimpleNamespace(layout=Layout(events)), context)
    operators = [args[0] for name, args, _ in events if name == "operator"]
    assert operators[-1] == "mesh_repair.save_report"
    # Refuse a stale snapshot after manual geometry edits.
    result.data.vertices[0].co.x += 1
    try:
        addon._validate_map_geometry(addon._LAST_REPAIR)
    except ValueError:
        pass
    else:
        raise AssertionError("Edited geometry must not accept the old snapshot")
    # A genuine worker failure must replace the previous successful report.
    bpy.ops.object.select_all(action="DESELECT")
    bad_mesh = bpy.data.meshes.new("degenerate")
    bad_mesh.from_pydata([(0, 0, 0)]*3, [], [(0, 1, 2)])
    bad_object = bpy.data.objects.new("degenerate", bad_mesh)
    context.collection.objects.link(bad_object)
    bad_object.select_set(True)
    context.view_layer.objects.active = bad_object
    for missing_result in (False, True):
        old_report = runtime.report_text
        assert old_report
        assert addon._begin(context, "repair", operator) == {"RUNNING_MODAL"}
        assert runtime.report_text == ""
        job = addon._ACTIVE
        assert job["process"].wait(timeout=60) != 0
        if missing_result:
            # A native crash may not leave a structured result file at all.
            job["result_file"].unlink()
            job["log_stream"].write(b"native failure diagnostic\n")
        modal = addon._MeshRepairModal()
        modal._mesh_repair_job_id = job["id"]
        assert modal.modal(context, SimpleNamespace(type="TIMER")) == {"FINISHED"}
        assert runtime.stage == "Ошибка" and runtime.report_text != old_report
        failure = json.loads(bpy.data.texts[runtime.report_text].as_string())
        assert failure["status"] == "error" and failure["settings"]["quality_mm"] > 0
        assert not job["directory"].exists()  # Report survives temporary-directory cleanup.
        if missing_result:
            assert "native failure diagnostic" in failure["worker_log_tail"]
        else:
            assert failure["traceback"]
    mesh_repair_blender.unregister()
    print("MESH_REPAIR_UI_SMOKE_OK")


if __name__ == "__main__":
    main()
