"""Blender UI and cancellable process orchestration for MeshRepair."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
import uuid

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ExportHelper

from .bridge import create_mesh_object, write_selected_stl
from . import heatmap


with Path(__file__).with_name("blender_manifest.toml").open("rb") as manifest:
    _ADDON_VERSION = tomllib.load(manifest)["version"]

_ACTIVE = None
_CACHED_JOB_DIR = None
_LAST_REPAIR = None


class MeshRepairSettings(PropertyGroup):
    quality_mm: FloatProperty(
        name="Допуск, мм", description="Шаг ремонта и предел упрощения", default=0.02,
        min=0.001, max=0.5, precision=3,
    )
    collapse_coplanar: BoolProperty(name="Схлопывать плоскости", default=True)
    coplanar_angle_deg: FloatProperty(
        name="Угол плоскостей", description="Допустимый угол для схлопывания, градусы",
        default=1.0, min=0.0, max=45.0,
    )
    simplify_curved: BoolProperty(name="Упрощать всю поверхность", default=True)
    strict_tolerance: BoolProperty(
        name="Строго доказывать допуск", default=False,
        description="Медленная численная верхняя граница для всей поверхности",
    )
    weld_close_vertices: BoolProperty(
        name="Склеивать близкие вершины", default=True,
        description="Порог склейки равен 0,1 от выбранного допуска",
    )
    restore_sharp_edges: BoolProperty(name="Восстанавливать острые рёбра", default=False)
    remove_small_shells: BoolProperty(name="Удалять оболочки < 50 граней", default=True)
    disable_cuda: BoolProperty(name="Без CUDA", default=False)
    hide_sources: BoolProperty(name="Скрыть исходники после ремонта", default=True)
    temp_directory: StringProperty(
        name="Временная папка", subtype="DIR_PATH", default="",
        description="Пустое поле использует системную временную папку",
    )
    show_advanced: BoolProperty(name="Расширенные настройки", default=False)


class MeshRepairRuntime(PropertyGroup):
    busy: BoolProperty(default=False, options={"SKIP_SAVE"})
    progress: IntProperty(default=0, min=0, max=100, subtype="PERCENTAGE", options={"SKIP_SAVE"})
    stage: StringProperty(default="Готово", options={"SKIP_SAVE"})
    message: StringProperty(default="", options={"SKIP_SAVE"})
    last_log: StringProperty(default="", options={"SKIP_SAVE"})
    report_text: StringProperty(default="", options={"SKIP_SAVE"})
    result_object: PointerProperty(type=bpy.types.Object, options={"SKIP_SAVE"})
    operation: StringProperty(default="", options={"SKIP_SAVE"})
    retry_available: BoolProperty(default=False, options={"SKIP_SAVE"})
    has_result: BoolProperty(default=False, options={"SKIP_SAVE"})
    triangles_before: IntProperty(default=0, options={"SKIP_SAVE"})
    triangles_after: IntProperty(default=0, options={"SKIP_SAVE"})
    components: IntProperty(default=0, options={"SKIP_SAVE"})
    boundary_loops: IntProperty(default=0, options={"SKIP_SAVE"})
    intersections: IntProperty(default=0, options={"SKIP_SAVE"})
    warning_count: IntProperty(default=0, options={"SKIP_SAVE"})
    backend: StringProperty(default="", options={"SKIP_SAVE"})


def _runtime(context):
    return context.window_manager.mesh_repair_runtime


def _redraw(context):
    # PropertyGroup writes from a modal timer do not invalidate the N-panel.
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _settings_dict(settings, force_no_sharp=False):
    return {
        "quality_mm": settings.quality_mm,
        "collapse_coplanar": settings.collapse_coplanar,
        "coplanar_angle_deg": settings.coplanar_angle_deg,
        "simplify_curved": settings.simplify_curved,
        "strict_tolerance": settings.strict_tolerance,
        "weld_close_vertices": settings.weld_close_vertices,
        "restore_sharp_edges": settings.restore_sharp_edges and not force_no_sharp,
        "remove_small_shells": settings.remove_small_shells,
        "disable_cuda": settings.disable_cuda,
        "temp_directory": "",
    }


def _python_prefix():
    explicit = os.environ.get("MESH_REPAIR_PYTHON")
    if explicit and Path(explicit).is_file():
        return [explicit]
    executable = Path(sys.executable)
    if executable.name.lower().startswith("python"):
        return [str(executable)]
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    blender_dir = Path(bpy.app.binary_path).resolve().parent
    names = ("python.exe", f"python{version}", f"python{sys.version_info.major}", "python")
    roots = (
        blender_dir / f"{bpy.app.version[0]}.{bpy.app.version[1]}" / "python" / "bin",
        Path(sys.prefix) / "bin",
    )
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return [str(candidate)]
    return [bpy.app.binary_path, "--background", "--factory-startup", "--python",
            str(Path(__file__).with_name("worker.py")), "--"]


def _command(job, operation):
    worker = Path(__file__).with_name("worker.py")
    prefix = _python_prefix()
    if len(prefix) == 1:
        prefix.append(str(worker))
    command = prefix + [
        operation,
        "--source", str(job["source"]),
        "--settings", str(job["settings"]),
        "--job-dir", str(job["core_dir"]),
        "--progress-file", str(job["progress_file"]),
        "--result-file", str(job["result_file"]),
    ]
    if operation == "repair":
        command += ["--vertices", str(job["vertices"]), "--faces", str(job["faces"])]
    elif operation == "heatmap":
        command += ["--vertices", str(job["vertices"]), "--distances", str(job["distances"])]
    return command


def _read_progress(job, runtime):
    path = job["progress_file"]
    if not path.exists():
        return
    with path.open("rb") as stream:
        stream.seek(job["progress_offset"])
        data = stream.read()
        job["progress_offset"] = stream.tell()
    pending = job["progress_pending"] + data
    lines = pending.split(b"\n")
    job["progress_pending"] = lines.pop()
    for line in lines:
        if not line:
            continue
        try:
            event = json.loads(line.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeDecodeError):
            continue
        if event.get("type") == "progress":
            value = event.get("value", {})
            job["last_worker_progress"] = value
            runtime.progress = max(runtime.progress, int(value.get("percent", 0)))
            runtime.stage = str(value.get("stage", runtime.stage))
        elif event.get("type") == "log":
            runtime.last_log = str(event.get("value", ""))[-1024:]


def _store_report(name, report):
    report["blender_addon_version"] = _ADDON_VERSION
    base = f"MeshRepair Report - {name}"
    text = bpy.data.texts.new(base)
    text.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return text


def _update_summary(runtime, report, operation):
    if operation == "check":
        before = after = report.get("info", {})
    else:
        before = report.get("before", {})
        after = report.get("after", report.get("reconstruction_validation", {}))
    runtime.triangles_before = int(before.get("triangles", 0) or 0)
    runtime.triangles_after = int(after.get("triangles", 0) or 0)
    runtime.components = int(after.get("components", 0) or 0)
    runtime.boundary_loops = int(after.get("boundary_loops", 0) or 0)
    runtime.intersections = int(after.get("self_intersecting_faces", 0) or 0)
    runtime.warning_count = len(report.get("warnings", []))
    acceleration = report.get("acceleration", {})
    devices = [str(report.get("compute_device", "")), *acceleration.get("projection_backends", [])]
    runtime.backend = "CPU + CUDA" if any("cuda" in value.lower() for value in devices) else "CPU"
    runtime.has_result = bool(before or after)


def _finish_job(context, job):
    global _ACTIVE, _CACHED_JOB_DIR, _LAST_REPAIR
    runtime = _runtime(context)
    _read_progress(job, runtime)
    job["log_stream"].close()
    result = None
    report = None
    retain_job = False
    try:
        if job["result_file"].is_file():
            result = json.loads(job["result_file"].read_text(encoding="utf-8"))
        if job.get("cancelled"):
            # The panel can already say Stop; retain the actual worker stage.
            cancelled = {
                "status": "cancelled", "operation": job["operation"],
                "settings": json.loads(job["settings"].read_text(encoding="utf-8")),
                "blender_source_objects": job["source_names"], "blender_mm_per_unit": job["mm_per_unit"],
                "elapsed_seconds": time.monotonic() - job["started"],
                "last_progress": job.get("last_worker_progress"), "last_log": runtime.last_log,
            }
            runtime.report_text = _store_report(job["result_name"], cancelled).name
            runtime.stage = "Остановлено"
            runtime.message = "Расчёт остановлен пользователем."
            return
        if not result:
            runtime.stage = "Ошибка"
            runtime.message = (f"Процесс завершился с кодом {job['process'].returncode}. "
                               "Подробности сохранены в JSON-отчёте.")
            log_path = job["directory"] / "worker.log"
            log_tail = ""
            if log_path.is_file():
                with log_path.open("rb") as stream:
                    stream.seek(max(0, log_path.stat().st_size - 16_384))
                    log_tail = stream.read().decode("utf-8", errors="replace")
            report = {"status": "error", "error": runtime.message,
                      "operation": job["operation"], "returncode": job["process"].returncode,
                      "settings": json.loads(job["settings"].read_text(encoding="utf-8")),
                      "blender_source_objects": job["source_names"], "worker_log_tail": log_tail}
            runtime.report_text = _store_report(job["result_name"], report).name
            return
        status = result.get("status")
        if job["operation"] == "heatmap":
            if status != "ok":
                raise RuntimeError(result.get("message", "Не удалось построить карту."))
            import numpy as np

            target = job["result_object"]
            _validate_map_geometry(job)
            heatmap.attach(target, np.load(job["distances"], allow_pickle=False), job["quality_mm"])
            heatmap.show(context, target)
            runtime.progress, runtime.stage = 100, "Карта отклонений"
            return
        if result.get("report") and Path(result["report"]).is_file():
            report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
            report["blender_source_objects"] = job["source_names"]
            report["blender_mm_per_unit"] = job["mm_per_unit"]
            text = _store_report(job["result_name"], report)
            runtime.report_text = text.name
            _update_summary(runtime, report, job["operation"])
        if status == "retry_without_sharpening":
            runtime.retry_available = True
            runtime.stage = "Нужен повтор"
            runtime.message = result.get("message", "Повторите без Sharpening.")
            return
        if status != "ok":
            runtime.stage = "Ошибка"
            runtime.message = result.get("message", "Неизвестная ошибка worker.")
            return
        if job["operation"] == "repair":
            heatmap.hide()
            repaired = create_mesh_object(context, job["vertices"], job["faces"],
                                          job["result_name"], job["mm_per_unit"])
            repaired["mesh_repair_report"] = runtime.report_text
            repaired["mesh_repair_quality_mm"] = job["quality_mm"]
            runtime.result_object = repaired
            job["result_object"] = repaired
            _LAST_REPAIR = job
            if job["hide_sources"]:
                for name in job["source_names"]:
                    source = bpy.data.objects.get(name)
                    if source is not None:
                        source.hide_set(True)
            runtime.message = f"Создан объект {repaired.name}."
            if _CACHED_JOB_DIR is not None:
                shutil.rmtree(_CACHED_JOB_DIR, ignore_errors=True)
            _CACHED_JOB_DIR = job["directory"]
            retain_job = True
        else:
            runtime.message = "Проверка завершена; геометрия не изменена."
        runtime.progress = 100
        runtime.stage = "Готово"
    except Exception as exc:
        runtime.stage = "Ошибка импорта"
        runtime.message = str(exc)
        if report is not None and runtime.report_text:
            report.update(status="import_error", error=str(exc))
            text = bpy.data.texts[runtime.report_text]
            text.clear()
            text.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        runtime.busy = False
        if job.get("timer") is not None:
            context.window_manager.event_timer_remove(job["timer"])
        if not retain_job:
            shutil.rmtree(job["directory"], ignore_errors=True)
        _ACTIVE = None
        _redraw(context)


def _begin(context, operation, operator, force_no_sharp=False):
    global _ACTIVE
    if _ACTIVE is not None:
        operator.report({"WARNING"}, "MeshRepair уже выполняет задачу.")
        return {"CANCELLED"}
    settings = context.scene.mesh_repair_settings
    runtime = _runtime(context)
    runtime.busy = True
    runtime.operation = operation
    runtime.progress = 0
    runtime.stage = "Подготовка геометрии"
    runtime.message = ""
    runtime.last_log = ""
    runtime.retry_available = False
    if operation != "heatmap":
        runtime.has_result = False
        runtime.report_text = ""
    _redraw(context)
    try:
        parent = None
        if settings.temp_directory:
            parent = Path(bpy.path.abspath(settings.temp_directory)).expanduser().resolve()
            parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="meshrepair-blender-", dir=parent))
        job = {
            "id": uuid.uuid4().hex,
            "started": time.monotonic(),
            "operation": operation,
            "directory": directory,
            "source": directory / "source.stl",
            "settings": directory / "settings.json",
            "core_dir": directory / "core-job",
            "progress_file": directory / "progress.jsonl",
            "result_file": directory / "result.json",
            "vertices": directory / "result-vertices.npy",
            "faces": directory / "result-faces.npy",
            "distances": directory / "distances.npy",
            "progress_offset": 0,
            "progress_pending": b"",
            "source_names": [obj.name for obj in context.selected_objects if obj.type == "MESH"],
            "quality_mm": settings.quality_mm,
            "hide_sources": settings.hide_sources,
        }
        active = context.view_layer.objects.active
        job["result_name"] = ((active.name if active and active.type == "MESH" else "Mesh") + "_repaired")[:63]
        if operation == "heatmap":
            if _LAST_REPAIR is None:
                raise ValueError("Сначала исправьте модель.")
            for key in ("source", "settings", "vertices", "result_object", "quality_mm", "mm_per_unit"):
                job[key] = _LAST_REPAIR[key]
            _validate_map_geometry(job)
        else:
            export = write_selected_stl(context, job["source"], "SCENE")
            job["mm_per_unit"] = export["mm_per_blender_unit"]
            job["settings"].write_text(json.dumps(_settings_dict(settings, force_no_sharp), indent=2), encoding="utf-8")
        log_stream = (directory / "worker.log").open("wb")
        job["log_stream"] = log_stream
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        job["process"] = subprocess.Popen(_command(job, operation), stdout=log_stream, stderr=subprocess.STDOUT,
                                          env=environment, creationflags=flags)
        job["timer"] = context.window_manager.event_timer_add(0.2, window=context.window)
        operator._mesh_repair_job_id = job["id"]
        context.window_manager.modal_handler_add(operator)
        _ACTIVE = job
        runtime.stage = "Запуск MeshRepair"
        _redraw(context)
        return {"RUNNING_MODAL"}
    except Exception as exc:
        if "job" in locals():
            _ACTIVE = None
            if job.get("timer") is not None:
                context.window_manager.event_timer_remove(job["timer"])
            process = job.get("process")
            if process is not None and process.poll() is None:
                process.terminate()
            stream = job.get("log_stream")
            if stream is not None and not stream.closed:
                stream.close()
        runtime.busy = False
        runtime.stage = "Ошибка запуска"
        runtime.message = str(exc)
        if "directory" in locals():
            shutil.rmtree(directory, ignore_errors=True)
        operator.report({"ERROR"}, str(exc))
        _redraw(context)
        return {"CANCELLED"}


class _MeshRepairModal:
    _mesh_repair_job_id = ""

    def modal(self, context, event):
        job = _ACTIVE
        if job is None or job["id"] != self._mesh_repair_job_id:
            return {"CANCELLED"}
        if event.type == "ESC":
            job["cancelled"] = True
            job["process"].terminate()
        # Blender's Event exposes the type, not the timer that generated it.
        # Polling on another TIMER is harmless; the job id above owns this modal.
        if event.type == "TIMER":
            _read_progress(job, _runtime(context))
            _redraw(context)
            if job["process"].poll() is not None:
                _finish_job(context, job)
                return {"FINISHED"}
        return {"PASS_THROUGH"}


class MESH_REPAIR_OT_repair_selected(_MeshRepairModal, Operator):
    bl_idname = "mesh_repair.repair_selected"
    bl_label = "Исправить выбранное"
    bl_description = "Создать исправленный объект из выбранных mesh-объектов"
    bl_options = {"REGISTER"}

    force_no_sharp: BoolProperty(default=False, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _ACTIVE is None and any(obj.type == "MESH" for obj in context.selected_objects)

    def execute(self, context):
        return _begin(context, "repair", self, self.force_no_sharp)


class MESH_REPAIR_OT_check_selected(_MeshRepairModal, Operator):
    bl_idname = "mesh_repair.check_selected"
    bl_label = "Проверить"
    bl_description = "Диагностировать выбранные объекты без изменения геометрии"

    @classmethod
    def poll(cls, context):
        return _ACTIVE is None and any(obj.type == "MESH" for obj in context.selected_objects)

    def execute(self, context):
        return _begin(context, "check", self)


class MESH_REPAIR_OT_stop(Operator):
    bl_idname = "mesh_repair.stop"
    bl_label = "Остановить"
    bl_description = "Завершить отдельный процесс MeshRepair"

    @classmethod
    def poll(cls, context):
        return _ACTIVE is not None

    def execute(self, context):
        if _ACTIVE is not None:
            _ACTIVE["cancelled"] = True
            _ACTIVE["process"].terminate()
            runtime = _runtime(context)
            runtime.stage = "Остановка..."
            runtime.message = "Ожидание завершения процесса."
            _redraw(context)
        return {"FINISHED"}


class MESH_REPAIR_OT_save_report(Operator, ExportHelper):
    bl_idname = "mesh_repair.save_report"
    bl_label = "Сохранить JSON-отчёт"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        name = _runtime(context).report_text
        return bool(name and bpy.data.texts.get(name))

    def execute(self, context):
        text = bpy.data.texts.get(_runtime(context).report_text)
        Path(self.filepath).write_text(text.as_string(), encoding="utf-8")
        self.report({"INFO"}, f"Отчёт сохранён: {self.filepath}")
        return {"FINISHED"}


def _map_target(context):
    obj = context.view_layer.objects.active
    return obj if heatmap.available(obj) else _runtime(context).result_object


def _validate_map_geometry(job):
    import numpy as np

    obj = job["result_object"]
    reference = np.load(job["vertices"], mmap_mode="r", allow_pickle=False)
    coordinates = np.empty((len(obj.data.vertices), 3), dtype=np.float32)
    obj.data.vertices.foreach_get("co", coordinates.ravel())
    if (obj.mode != "OBJECT" or obj.modifiers or not np.array_equal(np.asarray(obj.matrix_world), np.eye(4))
            or coordinates.shape != reference.shape
            or not np.array_equal(coordinates, np.asarray(reference) / float(job["mm_per_unit"]))):
        raise ValueError("Результат изменён после ремонта. Для точной карты выполните ремонт заново.")


class MESH_REPAIR_OT_view_deviation(_MeshRepairModal, Operator):
    bl_idname = "mesh_repair.view_deviation"
    bl_label = "Посмотреть отклонения"
    bl_description = "Расстояния вершин результата до исходной поверхности; синий — 0, зелёный — допуск, красный — 2 допуска"

    @classmethod
    def poll(cls, context):
        obj = _map_target(context)
        return _ACTIVE is None and (heatmap.available(obj) or (
            obj is not None and _LAST_REPAIR is not None and obj == _LAST_REPAIR.get("result_object")))

    def execute(self, context):
        obj = _map_target(context)
        if heatmap.active(obj):
            heatmap.hide()
            _runtime(context).stage = "Готово"
        elif heatmap.available(obj):
            heatmap.show(context, obj)
            _runtime(context).stage = "Карта отклонений"
        else:
            return _begin(context, "heatmap", self)
        _redraw(context)
        return {"FINISHED"}


class MESH_REPAIR_OT_details(Operator):
    bl_idname = "mesh_repair.details"
    bl_label = "Подробности"

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=480)

    def draw(self, context):
        import textwrap

        for line in textwrap.wrap(_runtime(context).message, width=60):
            self.layout.label(text=line)

    def execute(self, context):
        return {"FINISHED"}


def _stage_label(runtime):
    return "Ошибка" if runtime.stage.startswith("Ошибка") else runtime.stage


def _metric(layout, label, value):
    row = layout.row(align=True)
    row.label(text=label)
    row.alignment = "EXPAND"
    right = row.row()
    right.alignment = "RIGHT"
    right.label(text=str(value))


class MESHREPAIR_PT_panel(Panel):
    bl_label = "Mesh Repair"
    bl_idname = "MESHREPAIR_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mesh Repair"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mesh_repair_settings
        runtime = _runtime(context)
        selected = sum(1 for obj in context.selected_objects if obj.type == "MESH")
        layout.operator_context = "INVOKE_DEFAULT"
        layout.operator("mesh_repair.check_selected", icon="VIEWZOOM")
        box = layout.box()
        _metric(box, "Выбрано", selected)
        width = getattr(context.region, "width", 220) / max(1., context.preferences.system.ui_scale)
        for line in textwrap.wrap(_stage_label(runtime), width=max(12, int((width - 30) / 7))):
            box.label(text=line)
        if runtime.busy:
            elapsed = int(time.monotonic() - _ACTIVE["started"]) if _ACTIVE else 0
            box.progress(factor=runtime.progress / 100, type="BAR",
                         text=f"{runtime.progress}%   {elapsed//60}:{elapsed%60:02}")
            box.operator("mesh_repair.stop", icon="CANCEL")
        if runtime.has_result:
            _metric(box, "Грани до", f"{runtime.triangles_before:,}")
            if runtime.triangles_after != runtime.triangles_before:
                _metric(box, "Грани после", f"{runtime.triangles_after:,}")
            _metric(box, "Оболочки", runtime.components)
            _metric(box, "Дыры", runtime.boundary_loops)
            _metric(box, "Пересечения", f"{runtime.intersections:,}")
            _metric(box, "Вычисления", runtime.backend)
            if runtime.warning_count:
                _metric(box, "Предупреждения", runtime.warning_count)
        if runtime.message and (runtime.stage.startswith("Ошибка") or runtime.retry_available):
            box.operator("mesh_repair.details", icon="INFO")
        column = layout.column()
        column.enabled = not runtime.busy
        column.prop(settings, "quality_mm")
        row = column.row()
        row.scale_y = 1.7
        row.operator("mesh_repair.repair_selected", icon="MOD_REMESH")
        if runtime.retry_available:
            retry = column.operator("mesh_repair.repair_selected", text="Повтор без острых рёбер", icon="FILE_REFRESH")
            retry.force_no_sharp = True
        target = _map_target(context)
        if target is not None:
            column.operator("mesh_repair.view_deviation", text="Скрыть отклонения" if heatmap.active(target)
                            else "Посмотреть отклонения", icon="SHADING_RENDERED")
        if heatmap.active(target) and target is not None:
            legend = layout.box()
            quality = target.get("mesh_repair_map_quality_mm", .02)
            legend.label(text="Результат → исходник")
            _metric(legend, "Синий", "0 мм")
            _metric(legend, "Зелёный", f"{quality:g} мм")
            _metric(legend, "Красный ≥", f"{2 * quality:g} мм")
            _metric(legend, "Максимум", f"{target.get('mesh_repair_map_max_mm', 0):.4g} мм")
        layout.prop(settings, "show_advanced", text="Расширенные настройки",
                    icon="TRIA_DOWN" if settings.show_advanced else "TRIA_RIGHT", emboss=False)
        if settings.show_advanced:
            advanced = layout.box().column()
            advanced.enabled = not runtime.busy
            advanced.prop(settings, "restore_sharp_edges")
            advanced.prop(settings, "collapse_coplanar")
            if settings.collapse_coplanar:
                advanced.prop(settings, "coplanar_angle_deg")
            advanced.prop(settings, "simplify_curved")
            advanced.prop(settings, "strict_tolerance")
            advanced.prop(settings, "weld_close_vertices")
            advanced.prop(settings, "remove_small_shells")
            advanced.prop(settings, "disable_cuda")
            advanced.prop(settings, "hide_sources")
            advanced.label(text="Временная папка")
            advanced.prop(settings, "temp_directory", text="")
            if runtime.report_text:
                advanced.separator()
                advanced.label(text="Отчёт", icon="TEXT")
                advanced.operator("mesh_repair.save_report", text="Сохранить JSON", icon="EXPORT")


_CLASSES = (
    MeshRepairSettings,
    MeshRepairRuntime,
    MESH_REPAIR_OT_repair_selected,
    MESH_REPAIR_OT_check_selected,
    MESH_REPAIR_OT_stop,
    MESH_REPAIR_OT_save_report,
    MESH_REPAIR_OT_view_deviation,
    MESH_REPAIR_OT_details,
    MESHREPAIR_PT_panel,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh_repair_settings = PointerProperty(type=MeshRepairSettings)
    bpy.types.WindowManager.mesh_repair_runtime = PointerProperty(type=MeshRepairRuntime)


def unregister():
    global _ACTIVE, _CACHED_JOB_DIR, _LAST_REPAIR
    heatmap.hide()
    _LAST_REPAIR = None
    if _ACTIVE is not None:
        try:
            _ACTIVE["process"].terminate()
            _ACTIVE["log_stream"].close()
            shutil.rmtree(_ACTIVE["directory"], ignore_errors=True)
        except Exception:
            pass
        _ACTIVE = None
    if _CACHED_JOB_DIR is not None:
        shutil.rmtree(_CACHED_JOB_DIR, ignore_errors=True)
        _CACHED_JOB_DIR = None
    if hasattr(bpy.types.WindowManager, "mesh_repair_runtime"):
        del bpy.types.WindowManager.mesh_repair_runtime
    if hasattr(bpy.types.Scene, "mesh_repair_settings"):
        del bpy.types.Scene.mesh_repair_settings
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
