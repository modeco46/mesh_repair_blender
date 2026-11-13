"""Blender UI and cancellable process orchestration for MeshRepair."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ExportHelper

from .bridge import create_mesh_object, write_selected_stl


_ACTIVE = None


class MeshRepairSettings(PropertyGroup):
    quality_mm: FloatProperty(
        name="Допуск, мм", description="Шаг ремонта и предел упрощения", default=0.02,
        min=0.001, max=0.5, precision=3,
    )
    unit_mode: EnumProperty(
        name="Единицы входа",
        description="Как переводить координаты Blender в миллиметры для MeshRepair",
        items=(
            ("MILLIMETERS", "1 BU = 1 мм", "Координаты объекта уже заданы в миллиметрах"),
            ("SCENE", "Единицы сцены", "Использовать Unit Scale сцены"),
            ("METERS", "1 BU = 1 м", "Один Blender Unit равен одному метру"),
        ),
        default="MILLIMETERS",
    )
    collapse_coplanar: BoolProperty(name="Схлопывать плоскости", default=False)
    coplanar_angle_deg: FloatProperty(
        name="Угол плоскостей", description="Допустимый угол для схлопывания, градусы",
        default=1.0, min=0.0, max=45.0,
    )
    simplify_curved: BoolProperty(name="Упрощать всю поверхность", default=False)
    strict_tolerance: BoolProperty(
        name="Строго доказывать допуск", default=False,
        description="Медленная численная верхняя граница для всей поверхности",
    )
    weld_close_vertices: BoolProperty(
        name="Склеивать близкие вершины", default=False,
        description="Порог склейки равен 0,1 от выбранного допуска",
    )
    restore_sharp_edges: BoolProperty(name="Восстанавливать острые рёбра", default=True)
    remove_small_shells: BoolProperty(name="Удалять оболочки < 50 граней", default=False)
    disable_cuda: BoolProperty(name="Без CUDA", default=False)
    hide_sources: BoolProperty(name="Скрыть исходники после ремонта", default=False)
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
            runtime.progress = max(runtime.progress, int(value.get("percent", 0)))
            runtime.stage = str(value.get("stage", runtime.stage))
        elif event.get("type") == "log":
            runtime.last_log = str(event.get("value", ""))[-1024:]


def _store_report(name, report):
    base = f"MeshRepair Report - {name}"
    text = bpy.data.texts.new(base)
    text.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return text


def _update_summary(runtime, report, operation):
    if operation == "check":
        before = after = report.get("info", {})
    else:
        before, after = report.get("before", {}), report.get("after", {})
    runtime.triangles_before = int(before.get("triangles", 0) or 0)
    runtime.triangles_after = int(after.get("triangles", 0) or 0)
    runtime.components = int(after.get("components", 0) or 0)
    runtime.boundary_loops = int(after.get("boundary_loops", 0) or 0)
    runtime.intersections = int(after.get("self_intersecting_faces", 0) or 0)
    runtime.warning_count = len(report.get("warnings", []))
    acceleration = report.get("acceleration", {})
    runtime.backend = str(report.get("compute_device") or
                          ("CUDA: " + acceleration.get("gpu_name", "GPU")
                           if acceleration.get("cuda_available") else "CPU native parallel"))
    runtime.has_result = True


def _finish_job(context, job):
    global _ACTIVE
    runtime = _runtime(context)
    _read_progress(job, runtime)
    job["log_stream"].close()
    result = None
    try:
        if job["result_file"].is_file():
            result = json.loads(job["result_file"].read_text(encoding="utf-8"))
        if job.get("cancelled"):
            runtime.stage = "Остановлено"
            runtime.message = "Расчёт остановлен пользователем."
            return
        if not result:
            runtime.stage = "Ошибка"
            runtime.message = f"Worker завершился с кодом {job['process'].returncode}. См. worker.log."
            return
        status = result.get("status")
        report = None
        if result.get("report") and Path(result["report"]).is_file():
            report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
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
            repaired = create_mesh_object(context, job["vertices"], job["faces"],
                                          job["result_name"], job["mm_per_unit"])
            repaired["mesh_repair_report"] = runtime.report_text
            repaired["mesh_repair_quality_mm"] = context.scene.mesh_repair_settings.quality_mm
            if context.scene.mesh_repair_settings.hide_sources:
                for name in job["source_names"]:
                    source = bpy.data.objects.get(name)
                    if source is not None:
                        source.hide_set(True)
            runtime.message = f"Создан объект {repaired.name}."
        else:
            runtime.message = "Проверка завершена; геометрия не изменена."
        runtime.progress = 100
        runtime.stage = "Готово"
    except Exception as exc:
        runtime.stage = "Ошибка импорта"
        runtime.message = str(exc)
    finally:
        runtime.busy = False
        if job.get("timer") is not None:
            context.window_manager.event_timer_remove(job["timer"])
        shutil.rmtree(job["directory"], ignore_errors=True)
        _ACTIVE = None


def _begin(context, operation, operator, force_no_sharp=False):
    global _ACTIVE
    if _ACTIVE is not None:
        operator.report({"WARNING"}, "MeshRepair уже выполняет задачу.")
        return {"CANCELLED"}
    settings = context.scene.mesh_repair_settings
    runtime = _runtime(context)
    runtime.busy = True
    runtime.progress = 0
    runtime.stage = "Подготовка геометрии"
    runtime.message = ""
    runtime.last_log = ""
    runtime.retry_available = False
    runtime.has_result = False
    try:
        parent = None
        if settings.temp_directory:
            parent = Path(bpy.path.abspath(settings.temp_directory)).expanduser().resolve()
            parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="meshrepair-blender-", dir=parent))
        job = {
            "id": uuid.uuid4().hex,
            "operation": operation,
            "directory": directory,
            "source": directory / "source.stl",
            "settings": directory / "settings.json",
            "core_dir": directory / "core-job",
            "progress_file": directory / "progress.jsonl",
            "result_file": directory / "result.json",
            "vertices": directory / "result-vertices.npy",
            "faces": directory / "result-faces.npy",
            "progress_offset": 0,
            "progress_pending": b"",
            "source_names": [obj.name for obj in context.selected_objects if obj.type == "MESH"],
        }
        active = context.view_layer.objects.active
        job["result_name"] = ((active.name if active and active.type == "MESH" else "Mesh") + "_repaired")[:63]
        export = write_selected_stl(context, job["source"], settings.unit_mode)
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
        _ACTIVE = job
        operator._mesh_repair_job_id = job["id"]
        context.window_manager.modal_handler_add(operator)
        runtime.stage = "Запуск MeshRepair"
        return {"RUNNING_MODAL"}
    except Exception as exc:
        if "job" in locals():
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
        if event.type == "TIMER":
            _read_progress(job, _runtime(context))
            if job["process"].poll() is not None:
                _finish_job(context, job)
                return {"FINISHED"}
        return {"PASS_THROUGH"}


class MESHREPAIR_OT_repair(_MeshRepairModal, Operator):
    bl_idname = "mesh_repair.repair_selected"
    bl_label = "Исправить выбранное"
    bl_description = "Создать исправленный объект из выбранных mesh-объектов"
    bl_options = {"REGISTER"}

    force_no_sharp: BoolProperty(default=False, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _ACTIVE is None and any(obj.type == "MESH" for obj in context.selected_objects)

    def invoke(self, context, event):
        return _begin(context, "repair", self, self.force_no_sharp)


class MESHREPAIR_OT_check(_MeshRepairModal, Operator):
    bl_idname = "mesh_repair.check_selected"
    bl_label = "Проверить"
    bl_description = "Диагностировать выбранные объекты без изменения геометрии"

    @classmethod
    def poll(cls, context):
        return _ACTIVE is None and any(obj.type == "MESH" for obj in context.selected_objects)

    def invoke(self, context, event):
        return _begin(context, "check", self)


class MESHREPAIR_OT_stop(Operator):
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
        return {"FINISHED"}


class MESHREPAIR_OT_save_report(Operator, ExportHelper):
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
        layout.label(text=f"Выбрано mesh-объектов: {selected}", icon="MESH_DATA")
        if runtime.busy:
            layout.prop(runtime, "progress", text=runtime.stage, slider=True)
            layout.operator("mesh_repair.stop", icon="CANCEL")
        else:
            row = layout.row(align=True)
            row.operator("mesh_repair.check_selected", icon="VIEWZOOM")
            row.operator("mesh_repair.repair_selected", icon="MOD_REMESH")
            if runtime.retry_available:
                retry = layout.operator("mesh_repair.repair_selected", text="Повторить без острых рёбер", icon="FILE_REFRESH")
                retry.force_no_sharp = True
        if runtime.message:
            box = layout.box()
            box.label(text=runtime.stage)
            box.label(text=runtime.message[:120], icon="INFO" if runtime.stage != "Ошибка" else "ERROR")
            if runtime.last_log:
                box.label(text=runtime.last_log[:120])
        column = layout.column()
        column.enabled = not runtime.busy
        column.prop(settings, "quality_mm")
        column.prop(settings, "unit_mode")
        column.prop(settings, "restore_sharp_edges")
        column.prop(settings, "collapse_coplanar")
        if settings.collapse_coplanar and not settings.simplify_curved:
            column.prop(settings, "coplanar_angle_deg")
        column.prop(settings, "simplify_curved")
        column.prop(settings, "show_advanced", toggle=True)
        if settings.show_advanced:
            advanced = column.box()
            advanced.prop(settings, "strict_tolerance")
            advanced.prop(settings, "weld_close_vertices")
            advanced.prop(settings, "remove_small_shells")
            advanced.prop(settings, "disable_cuda")
            advanced.prop(settings, "hide_sources")
            advanced.prop(settings, "temp_directory")
        if runtime.has_result:
            box = layout.box()
            box.label(text="Последний результат")
            box.label(text=f"Треугольники: {runtime.triangles_before:,} → {runtime.triangles_after:,}")
            box.label(text=f"Оболочки: {runtime.components}; дыр: {runtime.boundary_loops}; пересечений: {runtime.intersections}")
            box.label(text=f"Вычисления: {runtime.backend}")
            if runtime.warning_count:
                box.label(text=f"Предупреждений: {runtime.warning_count}", icon="ERROR")
        if runtime.report_text:
            layout.label(text=f"Отчёт в Text: {runtime.report_text}", icon="TEXT")
            layout.operator("mesh_repair.save_report", icon="EXPORT")


_CLASSES = (
    MeshRepairSettings,
    MeshRepairRuntime,
    MESHREPAIR_OT_repair,
    MESHREPAIR_OT_check,
    MESHREPAIR_OT_stop,
    MESHREPAIR_OT_save_report,
    MESHREPAIR_PT_panel,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh_repair_settings = PointerProperty(type=MeshRepairSettings)
    bpy.types.WindowManager.mesh_repair_runtime = PointerProperty(type=MeshRepairRuntime)


def unregister():
    global _ACTIVE
    if _ACTIVE is not None:
        try:
            _ACTIVE["process"].terminate()
            _ACTIVE["log_stream"].close()
            shutil.rmtree(_ACTIVE["directory"], ignore_errors=True)
        except Exception:
            pass
        _ACTIVE = None
    if hasattr(bpy.types.WindowManager, "mesh_repair_runtime"):
        del bpy.types.WindowManager.mesh_repair_runtime
    if hasattr(bpy.types.Scene, "mesh_repair_settings"):
        del bpy.types.Scene.mesh_repair_settings
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
