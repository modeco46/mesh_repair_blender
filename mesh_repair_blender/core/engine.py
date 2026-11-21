"""One-click repair. MeshLib owns the native, multithreaded geometry work."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import math
import os
import tempfile
import time
import numpy as np
from .compute import capabilities, cuda_module


DENSE_WINDING_VOXEL_LIMIT = 64_000_000
DENSE_MESH_FACES = 1_000_000
MAX_SIMPLIFICATION_VOLUME_CHANGE = .02


@dataclass(frozen=True)
class Settings:
    quality_mm: float = 0.02
    collapse_coplanar: bool = False
    coplanar_angle_deg: float = 1.0
    simplify_curved: bool = False
    strict_tolerance: bool = False
    weld_close_vertices: bool = False
    restore_sharp_edges: bool = True
    local_fix_max_faces: int = 200
    local_fix_fraction: float = .001
    remove_small_shells: bool = False
    disable_cuda: bool = False
    temp_directory: str = ''
    sharpening_fallback_max_seconds: float = 120.0
    union_max_seconds: float = 240.0
    voxel_warn_seconds: float = 600.0
    voxel_max_cells: float = 6e9  # Warning threshold; streamed grids are not resident in full.
    # Full-field retries are opt-in: CUDA kernels may not call back for minutes.
    accurate_winding_max_seconds: float = 0.0

    def validate(self):
        if not math.isfinite(self.quality_mm) or not .001 <= self.quality_mm <= .5:
            raise ValueError("Качество должно быть от 0,001 до 0,5 мм.")
        if not math.isfinite(self.coplanar_angle_deg) or not 0 <= self.coplanar_angle_deg <= 45:
            raise ValueError("Угол копланарности должен быть от 0 до 45 градусов.")
        if self.local_fix_max_faces < 0 or not math.isfinite(self.local_fix_fraction) or not 0 <= self.local_fix_fraction <= 1:
            raise ValueError('Некорректный порог локального ремонта.')
        if not math.isfinite(self.sharpening_fallback_max_seconds) or self.sharpening_fallback_max_seconds < 0:
            raise ValueError('Некорректный предел времени повтора без Sharpening.')
        if not math.isfinite(self.union_max_seconds) or self.union_max_seconds < 0:
            raise ValueError('Некорректный предел времени объединения оболочек.')
        if not math.isfinite(self.voxel_warn_seconds) or self.voxel_warn_seconds < 0:
            raise ValueError('Некорректный порог предупреждения о времени вокселизации.')
        if not math.isfinite(self.voxel_max_cells) or self.voxel_max_cells <= 0:
            raise ValueError('Некорректный порог предупреждения о размере воксельной сетки.')
        if not math.isfinite(self.accurate_winding_max_seconds) or self.accurate_winding_max_seconds < 0:
            raise ValueError('Некорректный предел времени уточнения winding.')


class SharpeningRetryRequired(RuntimeError):
    """A costly retry needs one user click; retain the diagnostic report."""
    def __init__(self, report):
        self.report = report
        seconds = report['sharpening']['fallback']['estimated_seconds']
        super().__init__(f'Полный результат Sharpening невалиден. Оценка повтора: {seconds:.0f} с. '
                         'Нажмите «Повторить без Sharpening».')


class Progress:
    """Monotonic stage-weighted progress, driven by actual SDK callbacks."""
    def __init__(self, emit, log=None, timings=None, device=None, checkpoint=None):
        self.emit = emit
        self.percent = 0
        self.last_time = 0
        self.stage = ""
        self.stage_started = time.monotonic()
        self.log, self.timings, self.device = log, timings, device
        self.checkpoint = checkpoint or (lambda fields: None)

    def update(self, percent, stage):
        percent = max(self.percent, min(100, int(percent)))
        now = time.monotonic()
        if stage != self.stage:
            if self.stage and self.timings is not None:
                item = {'name': self.stage, 'elapsed_seconds': now-self.stage_started,
                        'compute_device': self.device(self.stage) if self.device else 'cpu_native_parallel'}
                self.timings.append(item)
                if self.log:
                    self.log(f"{item['name']}: {item['elapsed_seconds']:.2f} с; compute_device={item['compute_device']}.")
            self.stage_started = now
        if stage != self.stage or percent == 100 or (percent > self.percent and now - self.last_time >= .1):
            self.percent, self.stage, self.last_time = percent, stage, now
            self.emit({"percent": percent, "stage": stage})
        return True

    def callback(self, start, end, stage):
        self.update(start, stage)
        return lambda value: self.update(start + (end - start) * max(0, min(1, value)), stage)


def atomic_json(path, data, temp_dir=None):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".meshrepair-", suffix=".json", dir=temp_dir or path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(clean_json(data), f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def output_path(source):
    source = Path(source)
    for number in range(10000):
        suffix = "" if number == 0 else f"_{number}"
        path = source.with_name(source.stem + "_repaired" + suffix + ".stl")
        if not path.exists() and not Path(str(path) + ".json").exists():
            return path
    raise ValueError("В папке слишком много результатов с одинаковым именем.")


def topology_info(mesh, mm):
    """Cheap routing counters; deliberately no self-collision search."""
    return {
        "vertices": mesh.topology.numValidVerts(),
        "triangles": mesh.topology.numValidFaces(),
        "closed": mesh.topology.isClosed(),
        "topology_valid": mesh.topology.checkValidity(),
        "boundary_loops": mesh.topology.findNumHoles(),
        "components": mm.MeshComponents.getNumComponents(mm.MeshPart(mesh)),
        "signed_volume_mm3": mesh.volume(),
    }


def diagnostics(mesh, mm, callback=None):
    part = mm.MeshPart(mesh)
    collisions = mm.findSelfCollidingTrianglesBS(part, callback) if callback else mm.findSelfCollidingTrianglesBS(part)
    return dict(topology_info(mesh, mm), self_intersecting_faces=collisions.count())


def valid(info):
    return (info["triangles"] > 0 and info["closed"] and info["topology_valid"]
            and info["self_intersecting_faces"] == 0
            and math.isfinite(info["signed_volume_mm3"]) and info["signed_volume_mm3"] > 0)


def same_closed_surface_topology(before, after):
    # All faces are triangles. On a closed manifold E=3F/2, so equal
    # 2V-F means equal Euler characteristic (and total genus at fixed shell count).
    return (before['components'] == after['components'] and
            2 * before['vertices'] - before['triangles'] == 2 * after['vertices'] - after['triangles'])


def union_small(mesh, mm, tracker, deadline=None):
    """Small component sets only; larger sets go to sparse reconstruction, never fail by count.

    deadline is an optional time.monotonic() limit; the native boolean is cancelled through its callback."""
    regions = mm.MeshComponents.getAllComponents(mm.MeshPart(mesh))
    result = None
    for index, region in enumerate(regions):
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError('Превышен лимит времени объединения оболочек.')
        part = mesh.cloneRegion(region)
        if not part.topology.isClosed():
            raise RuntimeError("Открытая оболочка требует объёмного восстановления.")
        volume = part.volume()
        if not math.isfinite(volume) or volume == 0:
            raise RuntimeError('Оболочка нулевого или неопределённого объёма не может участвовать в Union.')
        if volume < 0:
            part.topology.flipOrientation()
        # The caller's post-hole diagnostic already excluded intersections in
        # every operand. Rechecking each component repeats that complete search.
        if result is None:
            result = part
        else:
            if not result.computeBoundingBox().intersects(part.computeBoundingBox()):
                result.addMesh(part)
                continue
            volumes = result.volume(), abs(volume)
            params = mm.BooleanParameters()
            progress_callback = tracker.callback(20 + 20 * index / len(regions),
                                                 20 + 20 * (index + 1) / len(regions), "Объединение поверхностей")
            params.cb = progress_callback if deadline is None else \
                (lambda value, cb=progress_callback: bool(cb(value)) and time.monotonic() < deadline)
            operation = mm.boolean(result, part, mm.BooleanOperation.Union, params)
            if not operation.valid():
                raise RuntimeError(operation.errorString)
            result = operation.mesh
            # A solid union contains both operands and cannot exceed their
            # sum. Allow numerical error from MeshLib's Boolean coordinates.
            combined = result.volume()
            slack = 5e-5 * sum(volumes)
            if not max(volumes)-slack <= combined <= sum(volumes)+slack:
                raise RuntimeError('Объём Union не согласуется с объёмами исходных тел.')
    return result


def shell_intersection_split(mesh, mm):
    """Colliding triangle pairs inside one shell versus between different shells."""
    pairs = mm.findSelfCollidingTriangles(mm.MeshPart(mesh))
    if not len(pairs):
        return 0, 0
    mapping, _ = mm.MeshComponents.getAllComponentsMap(mm.MeshPart(mesh))
    labels = np.fromiter((int(value) for value in mapping), dtype=np.int64)
    first = np.fromiter((int(pair.aFace) for pair in pairs), dtype=np.int64, count=len(pairs))
    second = np.fromiter((int(pair.bFace) for pair in pairs), dtype=np.int64, count=len(pairs))
    same = int((labels[first] == labels[second]).sum())
    return same, len(pairs) - same


def union_between_shells(mesh, info, mm, settings, tracker, report, emit):
    """Closed shells that only cross each other (bail, prongs, stones) are merged exactly by a boolean union.

    Voxelising the whole model for this case is slow, inflates the triangle count and leaves internal cavities."""
    item = report['union_attempt'] = {'attempted': False, 'accepted': False,
                                      'self_intersecting_faces': info['self_intersecting_faces']}
    if info['self_intersecting_faces'] > 500_000:
        item['reason'] = 'too_many_intersections'
        return None, None
    started = time.monotonic()
    inside, between = shell_intersection_split(mesh, mm)
    item.update(pairs_inside_shell=inside, pairs_between_shells=between, split_seconds=time.monotonic()-started)
    if inside or not between:
        item['reason'] = 'intersections_inside_shell' if inside else 'no_pairs'
        emit(f'Пересечения внутри одной оболочки ({inside}); булево объединение оболочек не применяется.' if inside
             else 'Пары пересекающихся граней не найдены; булево объединение не применяется.')
        return None, None
    item['attempted'] = True
    emit(f"Пересечения только между оболочками ({between:,} пар, оболочек {info['components']}): булево объединение без вокселей.")
    started = time.monotonic()
    try:
        repaired = union_small(mesh, mm, tracker, deadline=started + settings.union_max_seconds)
        if time.monotonic()-started >= settings.union_max_seconds:
            raise RuntimeError('Превышен лимит времени объединения оболочек.')
        after = diagnostics(repaired, mm)
    except (RuntimeError, MemoryError) as exc:
        item.update(reason='union_failed', error=str(exc), elapsed_seconds=time.monotonic()-started)
        report['warnings'].append(f'Булево объединение оболочек не удалось ({exc}); выполнено воксельное восстановление.')
        emit('Булево объединение не удалось; воксельное восстановление.')
        return None, None
    item['elapsed_seconds'] = time.monotonic()-started
    if not valid(after):
        item.update(reason='union_invalid', after=after)
        emit('Результат объединения не прошёл проверку; воксельное восстановление.')
        return None, None
    item.update(accepted=True, after=after)
    report['algorithm'] = 'solid_union'
    return repaired, after


def drop_cavities(mesh, after, mm, report, emit, quality_mm):
    """Internal voids and zero-thickness shells are removed after voxel reconstruction.

    They appear where self-intersecting or doubled internal surfaces give a zero winding number. Their signed
    volume is negative (void) or numerically zero (flat sheet thinner than a voxel). They make decimation change
    the shell count and an STL round trip split coincident layers into holes. Cavities inside the part are not
    preserved by design; at least one shell with real positive volume must remain."""
    if after['components'] < 2:
        return mesh, after, None
    regions = mm.MeshComponents.getAllComponents(mm.MeshPart(mesh))
    volumes = [(region, float(mesh.volume(region))) for region in regions]
    negligible = 1e-3*quality_mm**3
    voids = [(region, volume) for region, volume in volumes if volume <= negligible]
    if not voids or len(voids) == len(volumes):
        return mesh, after, None
    drop = mm.FaceBitSet()
    for region, _ in voids:
        drop |= region
    cleaned = mesh.cloneRegion(mesh.topology.getValidFaces()-drop)
    report['removed_cavities'] = [{'triangles': region.count(), 'volume_mm3': volume} for region, volume in voids]
    emit(f"Удалено внутренних полостей и пластин нулевой толщины: {len(voids)} (граней {drop.count():,}).")
    return cleaned, dict(topology_info(cleaned, mm), self_intersecting_faces=0), mesh


def clean_reconstruction_artifacts(mesh, info, mm, settings, report, emit):
    """Apply the existing cavity/small-shell policy before repairing collisions.

    Otherwise a tiny interior fragment scheduled for deletion can block a valid
    outer surface or force a full-grid retry. Re-diagnose the remaining mesh;
    deleting components never implies that its intersections disappeared.
    """
    if not info['closed'] or not info['topology_valid'] or info['components'] < 2:
        return mesh, info
    started = time.monotonic()
    regions = mm.MeshComponents.getAllComponents(mm.MeshPart(mesh))
    drop = mm.FaceBitSet()
    removed = []
    negligible = 1e-3*settings.quality_mm**3
    for region in regions:
        count, volume = region.count(), float(mesh.volume(region))
        cavity = volume <= negligible
        if cavity or (settings.remove_small_shells and count < 50):
            drop |= region
            box = mesh.computeBoundingBox(region)
            removed.append({'triangles': count, 'volume_mm3': volume,
                            'center_mm': list(box.center()), 'bbox_min_mm': list(box.min),
                            'bbox_max_mm': list(box.max),
                            'reason': 'cavity_or_zero_thickness' if cavity else 'small_shell'})
    item = report['reconstruction_artifacts'] = {'before': info, 'accepted': False, 'removed': []}
    if 0 < drop.count() < info['triangles']:
        candidate = mesh.cloneRegion(mesh.topology.getValidFaces()-drop)
        after = diagnostics(candidate, mm) if 'self_intersecting_faces' in info else topology_info(candidate, mm)
        if after['closed'] and after['topology_valid'] and after['triangles']:
            item.update(accepted=True, removed=removed, after=after)
            emit(f"Воксельные полости/мелкие оболочки: удалено {len(removed)}.")
            mesh, info = candidate, after
    item['elapsed_seconds'] = time.monotonic()-started
    return mesh, info


def level_set_distances(mesh, candidate, mm, q, translation=None):
    # "rigidB2A" maps the tested vertices into the target surface's coordinates.
    part, surface = mm.MeshPart(mesh), mm.MeshPart(candidate)
    to_world = mm.AffineXf3f.translation(translation) if translation is not None else None
    to_local = mm.AffineXf3f.translation(-translation) if translation is not None else None
    cap_sq = (2*q)**2
    into_source = math.sqrt(mm.findMaxDistanceSqOneWay(part, surface, to_local, cap_sq))
    into_result = math.sqrt(mm.findMaxDistanceSqOneWay(surface, part, to_world, cap_sq))
    return {'result_to_source_mm': into_source, 'source_to_result_mm': into_result, 'measurement_cap_mm': 2*q,
            'tolerance_mm': q, 'distance_method': 'bidirectional_vertices_to_surface',
            'within_step': all(math.isfinite(d) and d <= q for d in (into_source, into_result))}


def try_closed_level_set(mesh, mm, settings, tracker, report, emit, translation):
    """Voxelize a closed body in OpenVDB's surface band, then check it in output coordinates."""
    q = settings.quality_mm
    params = mm.OffsetParameters()
    params.voxelSize, params.signDetectionMode = q, mm.SignDetectionMode.OpenVDB
    params.callBack = tracker.callback(42, 62, 'Воксельный ремеш OpenVDB')
    closing = .35*q
    item = {'backend': 'openvdb_closed_level_set', 'compute_device': 'cpu_native_parallel', 'closing_offset_mm': closing,
            'probe': False, 'timing': False, 'sharpening': False, 'voxel_size_mm': q, 'accepted': False}
    report['voxel_passes'].append(item)
    tracker.checkpoint({'voxel_pass': dict(item)})
    started = time.monotonic()
    try:
        # A zero isosurface at a grazing self-contact can create hundreds of
        # tiny tunnels. Dilate/erode symmetrically within one voxel, then check
        # connectivity, handle count and both measured surface distances.
        candidate = mm.doubleOffsetMesh(mm.MeshPart(mesh), closing, -closing, params)
        if translation is not None:
            candidate.transform(mm.AffineXf3f.translation(translation))
        tracker.update(63, 'Проверка поверхности OpenVDB')
        after = item['candidate'] = diagnostics(candidate, mm)
        if not valid(after) or after['components'] != 1:
            item['reason'] = 'invalid_level_set_surface'
            return None
        original = {'vertices': mesh.topology.numValidVerts(), 'triangles': mesh.topology.numValidFaces(),
                    'components': 1}
        if not same_closed_surface_topology(original, after):
            item['reason'] = 'level_set_genus_changed'
            return None
        # A valid solid alone does not prove that VDB kept a thin detail or a
        # folded sheet. Measure every vertex in both directions at the same q.
        # Cap only clearly failing distances so the native queries can exit early.
        item.update(level_set_distances(mesh, candidate, mm, q, translation))
        if not item['within_step']:
            item['reason'] = 'level_set_distance_exceeds_step'
            return None
        item['accepted'] = True
        report.update(algorithm='closed_openvdb_level_set', reconstruction_validation=after)
        report['voxel_estimate'].update(storage='openvdb_surface_band', method='bbox_and_surface_area',
                                        probe_skipped='closed_level_set')
        return candidate
    except (RuntimeError, MemoryError) as exc:
        item.update(reason='level_set_failed', detail=str(exc))
        return None
    finally:
        item['elapsed_seconds'] = time.monotonic()-started
        emit(f"Ремеш OpenVDB: {item['elapsed_seconds']:.2f} с; " + ('принят.' if item['accepted']
             else f"отклонён [{item.get('reason')}], переход к winding с тем же шагом."))


def sparse_repair(mesh, mm, settings, tracker, report, emit, translation=None):
    """Validate the full result and reuse winding data for a bounded retry."""
    from .orientation import orient_closed_components
    report.update(algorithm='sparse_openvdb_winding', voxel_size_mm=settings.quality_mm)
    report['sparse_closed_orientation_corrections'] = orient_closed_components(mesh, mm)
    sharp = report['sharpening'] = {'requested': settings.restore_sharp_edges, 'accepted': False}
    report['compute_device'] = 'cpu_native_parallel'
    estimate = max(1, int(mesh.area()/settings.quality_mm**2))
    bbox = mesh.computeBoundingBox()
    extent = bbox.max - bbox.min

    def grid_for(step):
        return [max(1, int(math.ceil(float(value)/step))+2) for value in (extent.x, extent.y, extent.z)]

    grid = grid_for(settings.quality_mm)
    if max(grid) >= 2**31 or math.prod(grid) >= 2**63:
        raise RuntimeError('Размеры воксельной сетки выходят за диапазон индексов MeshLib. '
                           'Проверьте масштаб модели и допуск в миллиметрах.')
    cells = float(grid[0])*grid[1]*grid[2]
    report['voxel_estimate'] = {'triangles': estimate, 'method': 'bbox_cells_timed_probe',
                              'grid': grid, 'cells': cells, 'estimated_seconds': None,
                              'storage': 'streamed_layers' if cells > DENSE_WINDING_VOXEL_LIMIT else 'openvdb',
                              'estimate_scope': 'voxel_reconstruction_only'}
    estimated_seconds = cells*1.3e-6   # conservative CPU figure until the timed probe calibrates it
    emit(f'Воксельная сетка {grid[0]}×{grid[1]}×{grid[2]} ≈ {cells/1e6:,.0f} млн ячеек; '
         f'поверхность ≈ {estimate:,} треугольников (площадь / шаг²).')
    if cells > settings.voxel_max_cells:
        message = (f'Объём ограничивающего параллелепипеда: {cells/1e9:.1f} млрд ячеек. '
                   'Плотная итоговая поверхность требует оперативной памяти.')
        report['warnings'].append(message)
        emit(message)
    report['voxel_passes'] = []
    large_source = (mesh.topology.numValidFaces() >= DENSE_MESH_FACES or 'dense_routing' in report)
    # Two triangles per voxel-sized surface square already make a dense
    # isosurface even when the damaged input itself has only tens of thousands.
    dense_surface = large_source or 2*estimate >= DENSE_MESH_FACES
    closed = mesh.topology.isClosed()
    # These rims crossed another wall when smoothed. The winding sign can
    # distinguish the resulting folded surface after the fixed-rim cap closes it.
    folded_rims = closed and any(h.get('method') == 'fixed_rim_triangulation'
                                 for h in report.get('hole_filling', {}).get('details', []))
    if (dense_surface and not folded_rims and not settings.restore_sharp_edges and closed and
            mm.MeshComponents.getNumComponents(mm.MeshPart(mesh)) == 1):
        checked = report.get('after_hole_filling', {})
        crossing = checked.get('has_self_intersection')
        if crossing is None:
            count = checked.get('self_intersecting_faces')
            crossing = count > 0 if count is not None else mm.findSelfCollidingTriangles(
                mm.MeshPart(mesh), outCollidingPairs=None)
        if crossing:
            direct = try_closed_level_set(mesh, mm, settings, tracker, report, emit, translation)
            if direct is not None:
                return direct
    winding = None
    cuda = None if settings.disable_cuda else cuda_module()

    def reconstruct(sharpen, step, probe=False, stage=None, span=None, winding_beta=2.0, deadline=None):
        nonlocal winding, cuda
        params = mm.GeneralOffsetParameters() if sharpen else mm.OffsetParameters()
        params.voxelSize = step
        params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
        params.windingNumberBeta = winding_beta
        stage = stage or ('Проба острых рёбер (шаг ×4)' if probe else 'Восстановление поверхности')
        start, end = span or ((32, 41) if probe else (42, 65))
        grid_cells = math.prod(grid_for(step))
        timing = stage.startswith('Оценка времени')
        # Calibrate the backend selected for the full grid. Otherwise a small
        # dense OpenVDB probe can predict hours for a much faster streamed run.
        streaming = (grid_cells > DENSE_WINDING_VOXEL_LIMIT or
                     (timing and cells > DENSE_WINDING_VOXEL_LIMIT))
        # The native CPU FunctionVolume can evaluate the sign only in a thin
        # surface band of a closed mesh. The CUDA full-field call visits the box.
        narrow_band = streaming and not sharpen and closed
        if not narrow_band and cuda is not None and winding is None:
            try:
                winding = cuda.FastWindingNumber(mesh)
            except (RuntimeError, MemoryError) as exc:
                report['warnings'].append(f'CUDA недоступна, используется CPU: {exc}')
                cuda = None
        params.fwn = None if narrow_band else winding
        params.memoryEfficient = True
        params.closeHolesInHoleWindingNumber = not narrow_band
        callback = tracker.callback(start, end, stage+(' (по слоям)' if streaming else ''))
        params.callBack = callback if deadline is None else \
            (lambda value: bool(callback(value)) and time.monotonic() < deadline)
        if sharpen:
            params.mode = mm.GeneralOffsetParametersMode.Sharpening
        operation = mm.generalOffsetMesh if sharpen else mm.mcOffsetMesh if streaming else mm.offsetMesh
        backend = 'sharpening' if sharpen else 'marching_cubes_by_parts' if streaming else 'smooth_openvdb'
        slabbed = params.fwn is not None and (streaming or sharpen)
        device = ('cuda_winding_with_cpu_marching_cubes' if slabbed else 'cuda_winding_with_cpu_openvdb') if params.fwn is not None else 'cpu_native_parallel'
        report['compute_device'] = device
        item = {'probe': probe, 'sharpening': sharpen, 'voxel_size_mm': step, 'compute_device': device,
                'timing': timing, 'backend': backend, 'streaming': streaming,
                'grid_cells': grid_cells, 'cpu_closed_surface_band': narrow_band, 'winding_beta': winding_beta}
        if slabbed:
            item['field_evaluation'] = 'bounded_cuda_slabs'
        elif narrow_band:
            item['field_evaluation'] = 'closed_cpu_surface_band'
        report['voxel_passes'].append(item)
        tracker.checkpoint({'voxel_estimate': report['voxel_estimate'], 'voxel_pass': dict(item)})
        started = time.monotonic()
        emit(f'{stage}: шаг {step:g} мм; backend={backend}; compute_device={device}.')
        try:
            try:
                if slabbed:
                    from .winding import winding_slabs

                    def completed(block):
                        item['slabs'] = block
                        tracker.checkpoint({'voxel_pass': dict(item)})
                        callback(.90*block['completed_layers']/block['total_layers'])

                    return winding_slabs(mesh, mm, winding, params, completed, sharpen=sharpen, deadline=deadline)
                result = operation(mm.MeshPart(mesh), 0, params)
                if narrow_band and not result.topology.isClosed():
                    item['closed_surface_band_fallback'] = True
                    params.closeHolesInHoleWindingNumber = True
                    del result
                    item['field_evaluation'] = 'full_cpu_field'
                    tracker.checkpoint({'voxel_pass': dict(item)})
                    emit('Узкая область расчёта оставила отверстия; повтор с полным полем и тем же шагом.')
                    result = operation(mm.MeshPart(mesh), 0, params)
                return result
            except (RuntimeError, MemoryError) as exc:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Уточнение winding остановлено по лимиту времени. '
                                       'Подробности неудачного ремонта сохранены в JSON-отчёте.') from exc
                if params.fwn is None:
                    raise
                report['warnings'].append(f'Ошибка CUDA; повтор на CPU с тем же шагом: {exc}')
                cuda = winding = params.fwn = None
                report['compute_device'] = item['fallback_device'] = 'cpu_native_parallel'
                item.update(cuda_error=str(exc), field_evaluation='full_cpu_field')
                tracker.checkpoint({'voxel_pass': dict(item)})
                emit('Повтор после ошибки CUDA: compute_device=cpu_native_parallel.')
                return operation(mm.MeshPart(mesh), 0, params)
        finally:
            item['elapsed_seconds'] = time.monotonic()-started
            emit(f"{stage}: {item['elapsed_seconds']:.2f} с; compute_device={report['compute_device']}.")

    # A timing-only coarse pass still rebuilds structures for every input
    # triangle. It cannot change the ordinary (non-sharpening) result.
    skip_probe = (large_source or folded_rims) and not settings.restore_sharp_edges
    if cells >= 30e6 and skip_probe:
        report['voxel_estimate'].update(method='bbox_and_surface_area',
                                        probe_skipped='closed_fixed_rims' if folded_rims else 'dense_input')
    if cells >= 30e6 and not skip_probe:
        # Bound the probe itself for very fine grids. Total cells estimate work,
        # not peak memory of the layer-wise CPU/CUDA implementation.
        probe_factor = max(4, math.ceil((cells/4_000_000)**(1/3)))
        probe_step = settings.quality_mm*probe_factor
        probe_grid = grid_for(probe_step)
        probe_cells = float(probe_grid[0])*probe_grid[1]*probe_grid[2]
        started_probe = time.monotonic()
        probe_mesh = reconstruct(False, probe_step, probe=True,
                                 stage=f'Оценка времени (шаг ×{probe_factor})', span=(41, 42))
        probe_seconds = time.monotonic()-started_probe
        output_triangles = probe_mesh.topology.numValidFaces()*probe_factor**2
        del probe_mesh
        eta = estimated_seconds = probe_seconds*cells/probe_cells
        report['voxel_estimate'].update(estimated_seconds=eta, probe_seconds=probe_seconds, probe_cells=probe_cells,
                                       probe_factor=probe_factor, estimated_output_triangles=output_triangles)
        emit(f'Оценка только воксельного расчёта по пробе (шаг ×{probe_factor}): ≈ {eta:,.0f} с; '
             f'промежуточная поверхность ≈ {output_triangles:,} треугольников. '
             'Проверка и упрощение требуют дополнительного времени.')
        if eta > settings.voxel_warn_seconds:
            message = (f'Оценка воксельного расчёта: около {eta/60:.0f} мин при допуске '
                       f'{settings.quality_mm:g} мм, без проверки и упрощения. '
                       'Оценка приблизительная; расчёт продолжается с выбранным допуском.')
            report['warnings'].append(message)
            emit(message)

    use_sharp = False
    if settings.restore_sharp_edges:
        try:
            probe = reconstruct(True, settings.quality_mm*4, probe=True)
            check = diagnostics(probe, mm)
            sharp['probe_initial'] = check
            # A few coarse-grid seam intersections need not disable a useful
            # sharp reconstruction. Repair only the small, closed probe and
            # validate it; noisy probes with many intersections are rejected.
            threshold = max(settings.local_fix_max_faces, int(check['triangles']*settings.local_fix_fraction))
            if check['closed'] and check['topology_valid'] and 0 < check['self_intersecting_faces'] <= threshold:
                params = mm.FixSelfIntersectionSettings()
                params.method = mm.FixSelfIntersectionMethod.CutAndFill
                mm.SelfIntersections.fix(probe, params)
                probe.pack()
                check = diagnostics(probe, mm)
                sharp['probe_local_fix'] = True
            sharp['probe'] = check
            if not valid(check):
                raise RuntimeError(f"дыр {check['boundary_loops']}, пересекающихся граней {check['self_intersecting_faces']}")
            use_sharp = True
            emit('Грубая проба Sharpening прошла проверку.')
        except (RuntimeError, AttributeError, TypeError) as exc:
            sharp['reason'] = str(exc)
            warning = f'Sharpening отклонён на грубой пробе: {exc}. Полный расчёт — обычные воксели.'
            report['warnings'].append(warning)
            emit(warning)
    def full_result(sharpen, winding_beta=2.0, deadline=None):
        result = reconstruct(sharpen, settings.quality_mm, winding_beta=winding_beta,
                             stage='Уточнение воксельной поверхности' if winding_beta > 2 else None, deadline=deadline)
        if translation is not None:
            result.transform(mm.AffineXf3f.translation(translation))
        # Validate in output coordinates, once per full candidate. Never try
        # CutAndFill on a failed full-resolution Sharpening reconstruction.
        if not sharpen and result.topology.numValidFaces() >= DENSE_MESH_FACES:
            tracker.update(65, 'Очистка оболочек воксельной поверхности')
            result, check = clean_reconstruction_artifacts(result, topology_info(result, mm), mm, settings, report, emit)
            tracker.update(65, 'Проверка пересечений воксельной поверхности')
            collisions = mm.findSelfCollidingTrianglesBS(mm.MeshPart(result))
            check['self_intersecting_faces'] = collisions.count()
        else:
            check = diagnostics(result, mm)
            if not sharpen and not valid(check):
                result, check = clean_reconstruction_artifacts(result, check, mm, settings, report, emit)
            collisions = None
        if not sharpen and check['topology_valid'] and not check['closed']:
            from .seams import fill_voxel_seams
            tracker.update(65, 'Закрытие микротрещин воксельной поверхности')
            candidate, seam_report = fill_voxel_seams(result, mm, settings.quality_mm*.25)
            report['reconstruction_seams'] = seam_report
            if seam_report['accepted']:
                candidate_info = diagnostics(candidate, mm)
                if candidate_info['components'] == check['components']:
                    result, check = candidate, candidate_info
                    collisions = None  # The seam patch changed face ids.
                else:
                    seam_report.update(accepted=False, reason='seam_shell_count_changed')
            emit('Закрытие микротрещин: ' + ('принято.' if seam_report['accepted']
                 else f"отклонено [{seam_report.get('reason')}]."))
            del candidate
        if not sharpen and not valid(check):
            if collisions is None:
                result, check = repair_reconstruction_intersections(result, check, mm, settings, tracker, report, emit)
            else:
                result, check = repair_reconstruction_intersections(result, check, mm, settings, tracker, report, emit,
                                                                      initial_collisions=collisions)
        return result, check

    result, check = full_result(use_sharp)
    if use_sharp:
        sharp['full_validation'] = check
    if use_sharp and not valid(check):
        reason = (f"Полный результат Sharpening невалиден: дыр {check['boundary_loops']}, "
                  f"пересекающихся граней {check['self_intersecting_faces']}, "
                  f"топология {'корректна' if check['topology_valid'] else 'повреждена'}.")
        fallback = sharp['fallback'] = {'reason': 'full_sharpening_invalid', 'reason_ru': reason,
            'estimated_seconds': estimated_seconds, 'limit_seconds': settings.sharpening_fallback_max_seconds,
            'automatic': estimated_seconds <= settings.sharpening_fallback_max_seconds,
            'attempted': False, 'elapsed_seconds': 0.0}
        sharp['reason'] = reason
        report['warnings'].append(reason)
        emit(reason)
        if not fallback['automatic']:
            report['retry_without_sharpening'] = True
            raise SharpeningRetryRequired(report)
        emit(f'Автоматический повтор без Sharpening; оценка {estimated_seconds:.1f} с.')
        started = time.monotonic()
        fallback['attempted'] = True
        # Drop the rejected dense mesh before allocating its replacement.
        del result
        try:
            result, check = full_result(False)
            fallback.update(accepted=valid(check), after=check)
        finally:
            fallback['elapsed_seconds'] = time.monotonic()-started
            emit(f"Повтор без Sharpening: {fallback['elapsed_seconds']:.2f} с.")
        use_sharp = False
    if not use_sharp and not valid(check) and check['topology_valid']:
        threshold = max(settings.local_fix_max_faces, int(check['triangles']*settings.local_fix_fraction))
        if (check['boundary_loops'] <= 200 and check['self_intersecting_faces'] <= threshold
                and settings.accurate_winding_max_seconds > 0):
            # Near touching surfaces the approximate winding field can change
            # sign unreliably. Re-evaluate it more accurately, once, on the SAME
            # grid; do not relax the tolerance or accept a defective surface.
            retry = report['accurate_winding_retry'] = {
                'before': check, 'voxel_size_mm': settings.quality_mm, 'winding_beta': 4.0,
                'attempted': True, 'accepted': False,
                'limit_seconds': settings.accurate_winding_max_seconds,
                'previous_seams': report.pop('reconstruction_seams', None),
                'previous_local_repair': report.pop('reconstruction_local_repair', None)}
            report['reconstruction_validation'] = check
            emit('Остались дефекты поверхности: один повтор с более точным winding и тем же шагом; '
                 f'лимит вычисления поля {settings.accurate_winding_max_seconds:g} с.')
            started = time.monotonic()
            del result
            try:
                result, check = full_result(False, winding_beta=4.0,
                                            deadline=started+settings.accurate_winding_max_seconds)
                retry.update(after=check, accepted=valid(check))
            except TimeoutError:
                retry['reason'] = 'winding_retry_time_limit'
                raise
            finally:
                retry['elapsed_seconds'] = time.monotonic()-started
    sharp['accepted'] = use_sharp
    report['reconstruction_validation'] = check
    return result


def repair_reconstruction_intersections(mesh, info, mm, settings, tracker, report, emit, *, initial_collisions=None):
    """Keep bounded partial improvements and work on remaining collisions only."""
    from . import reduction as rd
    threshold = max(settings.local_fix_max_faces, int(info['triangles']*settings.local_fix_fraction))
    item = report['reconstruction_local_repair'] = {
        'attempted': False, 'accepted': False, 'before': info, 'threshold_faces': threshold,
        'tolerance_mm': settings.quality_mm*.25, 'maximum_tolerance_mm': settings.quality_mm*.5}
    if not (info['closed'] and info['topology_valid'] and 0 < info['self_intersecting_faces'] <= threshold):
        item['reason'] = 'outside_local_repair_limits'
        return mesh, info
    item['attempted'] = True
    started = time.monotonic()
    tracker.update(65, 'Локальное исправление воксельной поверхности')
    emit(f"Воксельная поверхность: локальное исправление {info['self_intersecting_faces']} пересекающихся граней.")
    item['attempts'] = []
    best, best_info = mesh, info
    best_collisions = (mm.findSelfCollidingTrianglesBS(mm.MeshPart(mesh))
                       if initial_collisions is None else initial_collisions)

    def assess(candidate, attempt):
        nonlocal best, best_info, best_collisions
        candidate.invalidateCaches()
        collisions = mm.findSelfCollidingTrianglesBS(mm.MeshPart(candidate))
        after = attempt['after'] = dict(topology_info(candidate, mm), self_intersecting_faces=collisions.count())
        if not after['closed'] or not after['topology_valid'] or not after['triangles']:
            attempt['reason'] = 'local_fix_invalid'
        elif after['components'] != info['components']:
            attempt['reason'] = 'local_fix_shell_count_changed'
        elif after['self_intersecting_faces'] >= best_info['self_intersecting_faces']:
            attempt['reason'] = 'no_local_improvement'
        else:
            if mesh.topology == candidate.topology:
                proof = rd.vertex_displacement_bound(mesh, candidate, attempt['tolerance_mm'])
            else:
                proof = rd.verify_local_surface_bound(mesh, candidate, attempt['tolerance_mm'],
                    tracker.callback(65, 66, 'Проверка локального исправления'),
                    acceleration=report.get('acceleration'), emit=emit)
            attempt['verification'] = proof
            if proof.get('verified'):
                best, best_info, best_collisions = candidate, after, collisions
                attempt['retained'] = True
                item['best_remaining_faces'] = after['self_intersecting_faces']
                emit(f"Локальное исправление: осталось пересекающихся граней {collisions.count()}.")
                if valid(after):
                    item.update(attempt, accepted=True)
                    return True
            else:
                attempt['reason'] = proof.get('reason', 'local_fix_exceeds_tolerance')
        return False

    try:
        # No pack() between local edits: stable face ids permit a patch proof.
        # Every attempt is bounded against the original reconstruction, so the
        # per-pass error budget cannot accumulate along retained candidates.
        methods = [('bounded_relax', 2, 20, .25), ('bounded_relax', 3, 40, .25),
                   ('local_degeneracies', 1, 0, .25), ('bounded_relax', 1, 80, .25),
                   ('native_relax', 0, 0, .25), ('cut_and_fill', 0, 0, .25),
                   ('bounded_relax', 2, 80, .5), ('cut_and_fill', 0, 0, .5)]
        for method, hops, iterations, fraction in methods:
            # Only a few residual faces may use the larger, still sub-voxel,
            # correction budget. It is always checked against the same origin.
            if fraction > .25 and best_info['self_intersecting_faces'] > 64:
                continue
            budget = settings.quality_mm*fraction
            attempt = {'method': method, 'input_intersecting_faces': best_info['self_intersecting_faces'],
                       'tolerance_mm': budget}
            item['attempts'].append(attempt)
            candidate = None
            try:
                candidate = mm.Mesh(best)
                if method == 'bounded_relax':
                    attempt.update(region_hops=hops, iterations=iterations)
                    region = mm.getIncidentVerts(candidate.topology, best_collisions)
                    mm.expand(candidate.topology, region, hops)
                    params = mm.MeshRelaxParams()
                    params.region, params.iterations, params.force = region, iterations, .25
                    params.limitNearInitial = True
                    params.maxInitialDist = budget*.98
                    mm.relax(candidate, params)
                    if mesh.topology == candidate.topology:
                        rd.clamp_displacements(mesh, candidate, budget*.98, mm)
                elif method == 'local_degeneracies':
                    region = mm.FaceBitSet(best_collisions)
                    mm.expand(candidate.topology, region, hops)
                    params = mm.FixMeshDegeneraciesParams()
                    params.region = region
                    params.mode = mm.FixMeshDegeneraciesParams.Mode.Decimate
                    params.maxDeviation = budget*.1
                    params.tinyEdgeLength = settings.quality_mm*.001
                    params.maxAngleChange = -1
                    mm.fixMeshDegeneracies(candidate, params)
                else:
                    params = mm.FixSelfIntersectionSettings()
                    params.method = (mm.FixSelfIntersectionMethod.Relax if method == 'native_relax'
                                     else mm.FixSelfIntersectionMethod.CutAndFill)
                    params.mimicPatch = True
                    params.touchIsIntersection = False
                    mm.SelfIntersections.fix(candidate, params)
                if assess(candidate, attempt):
                    return best, best_info
            except (RuntimeError, MemoryError) as exc:
                attempt.update(reason='local_fix_native_error', detail=str(exc))
            del candidate
        item.update(after=best_info, partial_improvement=best is not mesh, reason='remaining_local_intersections')
    finally:
        item['elapsed_seconds'] = time.monotonic()-started
        emit(f"Локальное исправление воксельной поверхности: {'принято' if item['accepted'] else 'отклонено'}; "
             f"{item['elapsed_seconds']:.2f} с" + (f" [{item['reason']}]" if item.get('reason') else '.'))
    return best, best_info


def local_repair(mesh, info, mm, settings, tracker, report, emit):
    threshold = max(settings.local_fix_max_faces, int(info['triangles']*settings.local_fix_fraction))
    item = report['local_repair'] = {'threshold_faces': threshold, 'attempted': False, 'accepted': False}
    if not 0 < info['self_intersecting_faces'] <= threshold:
        return None, None
    tracker.update(31, 'Локальное устранение самопересечений')
    started = time.monotonic()
    item.update(attempted=True, before=info)
    try:
        candidate = mm.Mesh(mesh)
        params = mm.FixSelfIntersectionSettings()
        params.method = mm.FixSelfIntersectionMethod.CutAndFill
        mm.SelfIntersections.fix(candidate, params)
        candidate.pack()
        after = diagnostics(candidate, mm)
        item['after'] = after
        if valid(after):
            item['accepted'] = True
            from .inspection import shell_summary
            _, item['shells'] = shell_summary(candidate, mm)
            return candidate, after
        item['reason'] = 'local_fix_invalid'
    except (RuntimeError, MemoryError) as exc:
        item.update(reason='local_fix_failed', detail=str(exc))
    finally:
        item['elapsed_seconds'] = time.monotonic()-started
        report['steps'].append({'name': 'Локальное устранение самопересечений', **item})
        emit(f"CutAndFill: {'принят' if item['accepted'] else 'отклонён'}; "
             f"{info['self_intersecting_faces']} → {item.get('after', {}).get('self_intersecting_faces', '—')}; "
             f"{item['elapsed_seconds']:.2f} с; compute_device=cpu_native_parallel.")
    return None, None



def clean_json(value):
    """Reports remain serializable even when an SDK diagnostic is non-finite."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def publish_stl(source, target):
    # Windows rename is no-replace and works on filesystems without hard links.
    if os.name == 'nt':
        os.rename(source, target)
    else:
        os.link(source, target)


def cleanup_exported(exported, path, mm, settings, tracker, report, folder, emit, shape_reference=None):
    """Compatibility name; cleanup and validation now stay entirely in memory."""
    started = time.monotonic()
    item = {'accepted': False, 'before': report['after']['triangles']}
    try:
        candidate = mm.Mesh(exported)
        params = mm.FixMeshDegeneraciesParams()
        params.mode = mm.FixMeshDegeneraciesParams.Mode.Decimate
        params.maxDeviation = settings.quality_mm*.01
        params.tinyEdgeLength = settings.quality_mm*.001
        params.cb = tracker.callback(78, 79, 'Очистка вырожденных граней')
        mm.fixMeshDegeneracies(candidate, params)
        candidate.invalidateCaches()
        candidate.pack()
        info = diagnostics(candidate, mm)
        if not valid(info):
            item['reason'] = 'cleanup_topology_changed'
        elif info['components'] != report['after']['components']:
            item['reason'] = 'cleanup_shell_count_changed'
        elif not same_closed_surface_topology(report['after'], info):
            item['reason'] = 'cleanup_topology_changed'
        elif shape_reference is not None and not level_set_distances(
                shape_reference, candidate, mm, settings.quality_mm)['within_step']:
            item['reason'] = 'level_set_distance_exceeds_step'
        else:
            exported = candidate
            report['after'] = info
            item.update(accepted=True, after=info['triangles'])
    except (RuntimeError, MemoryError) as exc:
        item.update(reason='cleanup_native_error', detail=str(exc))
    item['elapsed_seconds'] = time.monotonic()-started
    report['degenerate_cleanup'] = item
    report['steps'].append({'name': 'Очистка вырожденных граней', **item})
    emit(f"Очистка: {item['elapsed_seconds']:.2f} с; compute_device=cpu_native_parallel.")
    if not item['accepted']:
        message = 'Очистка вырожденных граней отклонена. Сохранена сетка до очистки. ['+item['reason']+']'
        emit(message)
        report['warnings'].append(message)
    return exported


def simplify_attempts(exported, tmp, mm, settings, tracker, report, emit, workdir, shape_reference=None):
    """Search measured error against the unchanged repaired reference; prove once."""
    from . import reduction as rd
    original = report['after']
    mode = 'proven' if settings.strict_tolerance else 'measured_not_proven'
    target = .95 * settings.quality_mm
    summary = {'requested': True, 'accepted': False, 'before': original['triangles'],
               'after': original['triangles'], 'tolerance_mm': settings.quality_mm,
               'measured_target_mm': target, 'scale_range': [0., .8],
               'reference_volume_mm3': original['signed_volume_mm3'],
               'max_relative_volume_change': MAX_SIMPLIFICATION_VOLUME_CHANGE,
               'strict_tolerance': settings.strict_tolerance, 'mode': mode,
               'input_triangles': report.get('before', {}).get('triangles'), 'attempts': []}
    report['simplification'] = summary
    emit('Режим допуска: '+('доказано (строгая проверка).' if settings.strict_tolerance else 'измерено (вершины в обе стороны).'))
    started = time.monotonic()
    shellwise = False
    prepared = None

    tracker.update(76, 'Подготовка к упрощению')

    dense = settings.simplify_curved and original['triangles'] >= DENSE_MESH_FACES

    def propose(factor, phase, base=None):
        nonlocal shellwise, prepared
        working = exported if base is None else base
        item = {'attempt': len(summary['attempts'])+1, 'phase': phase, 'shellwise': shellwise,
                'input_triangles': working.topology.numValidFaces(),
                'error_scale': factor, 'max_error_mm': settings.quality_mm*factor, 'checkpoints': []}
        summary['attempts'].append(item)
        t = time.monotonic()
        last_valid = None

        def checkpoint(candidate, stages):
            nonlocal last_valid
            check = {'pass': len(stages), 'stages': stages}
            item['checkpoints'].append(check)
            if candidate.topology.numValidFaces() >= working.topology.numValidFaces():
                check['reason'] = 'no_reducible_edges'
                return False
            callback = tracker.callback(86, 88, 'Проверка упрощённой сетки')
            if dense:
                collisions = mm.findSelfCollidingTrianglesBS(mm.MeshPart(candidate), callback)
                info = dict(topology_info(candidate, mm), self_intersecting_faces=collisions.count())
                if (0 < collisions.count() <= 64 and info['triangles'] <= original['triangles']//2
                        and info['closed'] and info['topology_valid']):
                    # One local movement of a private candidate. Its result must
                    # pass the same whole-surface checks against the original.
                    region = mm.getIncidentVerts(candidate.topology, collisions)
                    mm.expand(candidate.topology, region, 2)
                    params = mm.MeshRelaxParams()
                    params.region, params.iterations, params.force = region, 15, .25
                    params.limitNearInitial, params.maxInitialDist = True, settings.quality_mm*.15
                    mm.relax(candidate, params)
                    check['local_relax'] = {'initial_intersecting_faces': collisions.count(),
                                            'limit_mm': params.maxInitialDist}
                    info = diagnostics(candidate, mm)
            else:
                info = diagnostics(candidate, mm, callback)
            if not valid(info) and mm.findDegenerateFaces(mm.MeshPart(candidate)).count():
                cleanup_report = {'after': info, 'steps': [], 'warnings': []}
                candidate = cleanup_exported(candidate, None, mm, settings, tracker, cleanup_report, None, emit)
                info = cleanup_report['after']
                check['cleanup'] = cleanup_report['degenerate_cleanup']
            check['candidate'] = info
            if info['self_intersecting_faces'] and info['closed'] and info['topology_valid']:
                check['reason'] = 'self_intersections'
            elif not valid(info):
                check['reason'] = 'topology_changed'
            elif info['components'] != original['components']:
                check['reason'] = 'kept_shell_count_changed'
            elif not same_closed_surface_topology(original, info):
                check['reason'] = 'genus_changed'
            elif abs(info['signed_volume_mm3'] / original['signed_volume_mm3'] - 1) > MAX_SIMPLIFICATION_VOLUME_CHANGE:
                check['reason'] = 'volume_changed'
            else:
                tracker.update(88, f"Подбор допуска: {item['attempt']}")
                deviation = rd.measured_distance(exported, candidate, mm)
                check['verification'] = {'verified': False, 'mode': 'measured_not_proven',
                    'max_deviation_mm': deviation, 'reference': 'repaired_mesh_before_simplification',
                    'method': 'bidirectional_vertices_to_surface'}
                if math.isfinite(deviation) and deviation <= target:
                    if shape_reference is not None:
                        source_check = check['level_set_source_check'] = level_set_distances(
                            shape_reference, candidate, mm, settings.quality_mm)
                        if not source_check['within_step']:
                            check['reason'] = 'level_set_distance_exceeds_step'
                            return False
                    if last_valid is None or info['triangles'] < last_valid[1]['triangles']:
                        last_valid = candidate, info, check
                    return True
                check['reason'] = 'sample_exceeds_tolerance'
            emit(f"Попытка {item['attempt']}, проход {len(stages)} отклонён: {rd.reason_text(check['reason'])}")
            return False

        try:
            reduce = rd.propose_reduction_by_shell if shellwise else rd.propose_reduction
            # Keep the whole-mesh -> per-shell fallback. A single body also
            # retains safe intermediate QEM passes against the same reference.
            per_pass = original['components'] == 1 and not shellwise and not dense
            options = {'checkpoint': checkpoint} if per_pass else {}
            if dense:
                options['single_pass'] = True
            if phase in {'endpoint_retry', 'dense_seed', 'conservative_seed', 'compact_endpoints'}:
                options['endpoint_only'] = True
            if not shellwise:
                if base is None and prepared is None:
                    prepared = rd.prepare_reduction(exported, mm, components=original['components'])
                    summary['preparation_seconds'] = time.monotonic()-t
                options['prepared'] = prepared if base is None else rd.prepare_reduction(
                    base, mm, components=original['components'])
            candidate, stages = reduce(working, mm, settings, tracker, error_scale=factor,
                preserve_planar_corners=report.get('algorithm') in {
                    'surface_repair', 'solid_union', 'grouped_solid_union', 'closed_components_preserved'}, **options)
            if not item['checkpoints']:
                checkpoint(candidate, stages)
            if not item['checkpoints']:
                item.update(reason='no_reducible_edges', stages=stages)
            elif last_valid is None:
                item.update(item['checkpoints'][-1])
        except (RuntimeError, MemoryError) as exc:
            item.update(reason='native_error', detail=str(exc))
        finally:
            item['elapsed_seconds'] = time.monotonic()-t
            emit(f"Упрощение, попытка {item['attempt']}: {item['elapsed_seconds']:.2f} с; compute_device=cpu_native_parallel.")
            if item.get('reason') and last_valid is None:
                emit(f"Попытка {item['attempt']} отклонена: {rd.reason_text(item['reason'])}")
        if last_valid is not None:
            candidate, info, selected = last_valid
            if item['checkpoints'][-1].get('reason') or item.get('reason'):
                item['stopped_reason'] = item.get('reason') or item['checkpoints'][-1]['reason']
                emit(f"Сохранён проверенный проход {selected['pass']} попытки {item['attempt']}.")
            item.pop('reason', None)
            item.update(selected)
            return candidate, info, item
        if not dense and item.get('reason') in {'topology_changed', 'kept_shell_count_changed', 'self_intersections'} and not shellwise \
                and original['components'] > 1:
            shellwise = True
            summary['shellwise'] = True
            emit('Общее упрощение меняет оболочки; упрощение каждой оболочки отдельно.')
            return propose(factor, phase)
        return None, None, item

    best = seed = None
    if dense:
        summary['search_strategy'] = 'progressive_dense'
        # Start with existing vertices. Typically a voxel surface loses most
        # faces in this step. Only a rejected seed permits one dense retry.
        for factor, phase in ((.15625, 'dense_seed'), (.04, 'conservative_seed')):
            candidate, info, item = propose(factor, phase)
            if candidate is not None:
                seed = best = candidate, info, item
                summary['search_stop'] = 'accepted_seed'
                if info['triangles'] <= original['triangles']//2:
                    for _ in range(2):
                        previous = best[1]['triangles']
                        candidate, info, item = propose(.475, 'compact_refinement', base=best[0])
                        if candidate is None:
                            if item.get('reason') in {'self_intersections', 'level_set_distance_exceeds_step',
                                                      'sample_exceeds_tolerance', 'volume_changed'}:
                                candidate, info, item = propose(.15625, 'compact_endpoints', base=best[0])
                                if candidate is not None:
                                    best = candidate, info, item
                                    summary['search_stop'] = 'accepted_endpoint_refinement'
                            break
                        best = candidate, info, item
                        summary['search_stop'] = 'accepted_refinement'
                        if info['triangles'] > previous*.97:
                            break
                break
            if item.get('reason') == 'no_reducible_edges':
                break
        if best is None:
            summary['search_stop'] = 'dense_seed_unavailable'
    low, high = 0., .8
    for iteration in range(0 if dense else 6):
        # Preserve the first two historical probes for stable cases. If both
        # fail, approach zero instead of hitting the old hard floor of 0.15.
        factor = .475 if iteration == 0 else .3125 if iteration == 1 and best is None else (low+high)/2
        # Similar to disabling OptimalPlacement in MeshLab. Reuse one probe
        # from the existing budget only after three failed free placements.
        # Retain the original repaired mesh as the distance reference.
        if (iteration == 3 and best is None and settings.simplify_curved and original['components'] == 1 and
                all(a.get('reason') in {'self_intersections', 'sample_exceeds_tolerance', 'volume_changed'}
                    for a in summary['attempts'])):
            previous_scale = summary['attempts'][-1]['error_scale']
            candidate, info, item = propose(previous_scale, 'endpoint_retry')
            if candidate is not None:
                best = candidate, info, item
                summary['search_stop'] = 'valid_endpoint_retry'
                break
            continue
        candidate, info, item = propose(factor, 'search')
        if candidate is None:
            high = factor
            if item.get('reason') == 'no_reducible_edges':
                break
        else:
            low = factor
            # QEM is not strictly monotone in scale. Retain the smallest valid
            # candidate, even if a later, larger scale gives a worse mesh.
            if best is None or info['triangles'] < best[1]['triangles']:
                best = candidate, info, item
            if item['verification']['max_deviation_mm'] < settings.quality_mm*1e-4:
                # Exact planar collapse has no unused curved error budget.
                break
            if item['verification']['max_deviation_mm'] >= target*.99:
                summary['search_stop'] = 'within_one_percent_of_measured_target'
                break

    # Strict proof is intentionally outside the binary search. Each proof and
    # every retry uses the original repaired surface, never the previous pass.
    if best is not None and settings.strict_tolerance:
        for proof_attempt in range(3):
            candidate, info, item = best
            t = time.monotonic()
            try:
                verification = rd.verify_surface_bound(exported, candidate, settings.quality_mm,
                    tracker.callback(88, 96, f'Строгая проверка допуска: {proof_attempt+1}/3'),
                    acceleration=report['acceleration'], emit=emit)
            except (RuntimeError, MemoryError) as exc:
                verification = {'verified': False, 'reason': 'native_error', 'detail': str(exc)}
            verification['mode'] = mode
            verification['reference'] = 'repaired_mesh_before_simplification'
            item['measured_verification'] = item['verification']
            item['verification'] = verification
            item['proof_seconds'] = time.monotonic()-t
            if verification.get('verified'):
                verification['max_deviation_mm'] = verification['upper_bound_mm']
                break
            item['reason'] = verification.get('reason', 'distance_not_confirmed')
            best = None
            if dense:
                if seed is not None and seed[2] is not item:
                    best = seed
                    continue
                break
            if proof_attempt < 2:
                candidate, info, item = propose(item['error_scale']*.8, 'proof_retry')
                if candidate is not None:
                    best = candidate, info, item
                else:
                    break

    accepted = exported
    if best is not None:
        accepted, info, item = best
        report['after'] = info
        summary.update(accepted=True, after=info['triangles'], accepted_attempt=item['attempt'],
                       error_scale=item['error_scale'], stages=item['stages'], verification=item['verification'])
    else:
        item = summary['attempts'][-1]
        summary['reason'] = item.get('reason', 'native_error')
        summary['reason_ru'] = rd.reason_text(summary['reason'])
        summary['verification'] = item.get('verification', {})
        emit('Сохранена сетка без упрощения. ' + summary['reason_ru'])
        if summary['reason'] != 'no_reducible_edges':
            report['warnings'].append(summary['reason_ru'])
    summary['elapsed_seconds'] = time.monotonic()-started
    emit(f"Упрощение: {summary['before']:,} → {summary['after']:,} треугольников.")
    report['steps'].append({'name': 'Упрощение', 'elapsed_seconds': summary['elapsed_seconds']})
    return accepted


def finish_planar_faces(reference, current, mm, settings, tracker, report, emit, shape_reference=None):
    """Remove planar partition seams, without renewing the total error budget."""
    from . import reduction as rd
    before = report['after']
    item = report['planar_finish'] = {'requested': True, 'accepted': False,
                                     'before': before['triangles'], 'after': before['triangles']}
    started = time.monotonic()
    tracker.update(94, 'Финальное схлопывание плоских участков')
    try:
        # Existing vertices and ridge corners stay fixed. A single unpartitioned
        # pass removes the artificial seams left by parallel general decimation.
        planar = replace(settings, simplify_curved=False, collapse_coplanar=True)
        # A simplified whole body can have <50 faces. The general reducer's
        # small-shell exclusion must not disable its planar finishing pass.
        prepared = {'candidate': current, 'protected': None, 'protected_count': 0}
        candidate, stages = rd.propose_reduction(current, mm, planar, tracker, error_scale=.001, prepared=prepared)
        item['stages'] = stages
        if candidate.topology.numValidFaces() >= before['triangles']:
            item['reason'] = 'no_reducible_edges'
            return current
        after = item['candidate'] = diagnostics(candidate, mm)
        if not valid(after) or not same_closed_surface_topology(before, after):
            item['reason'] = 'planar_finish_topology_changed'
            return current
        if abs(after['signed_volume_mm3'] / reference.volume() - 1) > MAX_SIMPLIFICATION_VOLUME_CHANGE:
            item['reason'] = 'volume_changed'
            return current
        # Always compare to the original repaired surface, not the preceding
        # simplified mesh. No independent second tolerance can accumulate.
        measured = rd.measured_distance(reference, candidate, mm)
        proof = item['verification'] = {'verified': False, 'mode': 'measured_not_proven',
            'max_deviation_mm': measured, 'reference': 'repaired_mesh_before_simplification',
            'method': 'bidirectional_vertices_to_surface'}
        if not math.isfinite(measured) or measured > settings.quality_mm*.95:
            item['reason'] = 'planar_finish_exceeds_total_tolerance'
            return current
        if shape_reference is not None:
            source_check = item['level_set_source_check'] = level_set_distances(
                shape_reference, candidate, mm, settings.quality_mm)
            if not source_check['within_step']:
                item['reason'] = 'level_set_distance_exceeds_step'
                return current
        if settings.strict_tolerance:
            proof = item['verification'] = rd.verify_surface_bound(reference, candidate, settings.quality_mm,
                tracker.callback(94, 96, 'Строгая проверка финальных плоскостей'),
                acceleration=report.get('acceleration'), emit=emit)
            if not proof.get('verified'):
                item['reason'] = proof.get('reason', 'planar_finish_bound_unconfirmed')
                return current
        item.update(accepted=True, after=after['triangles'])
        report['after'] = after
        return candidate
    except (RuntimeError, MemoryError) as exc:
        item.update(reason='planar_finish_native_error', detail=str(exc))
        return current
    finally:
        item['elapsed_seconds'] = time.monotonic()-started
        report['steps'].append({'name': 'Финальное схлопывание плоских участков',
                               'elapsed_seconds': item['elapsed_seconds'], 'accepted': item['accepted']})
        emit(f"Финальные плоскости: {item['before']:,} → {item['after']:,} треугольников; "
             f"{item['elapsed_seconds']:.2f} с." + (f" [{item['reason']}]" if item.get('reason') else ''))



def run(source, destination=None, settings=Settings(), emit=lambda message: None,
        progress=lambda event: None, work_dir=None, *, array_output=None, checkpoint=None):
    from .storage import create_job, remove_job
    settings.validate()
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError('Входной файл не найден.')
    if source.suffix.lower() not in {'.stl', '.obj', '.ply'}:
        raise ValueError('Поддерживаются STL, OBJ и PLY.')
    target = Path(destination).resolve() if destination else None
    if array_output is not None:
        array_output = tuple(Path(path).resolve() for path in array_output)
        if target or len(array_output) != 2 or len(set(array_output)) != 2 or source in array_output:
            raise ValueError('Для индексированной сетки нужны два разных выходных файла, отдельно от STL.')
        if any(path.exists() for path in array_output):
            raise ValueError('Выходные файлы сетки уже существуют.')
    if target:
        if target == source or any(Path(str(target)+suffix).exists() for suffix in ('', '.json', '.deviation.npz')):
            raise ValueError('Выходной файл уже существует. Исходники и результаты не перезаписываются.')
        if not target.parent.is_dir() or target.suffix.lower() != '.stl':
            raise ValueError('Нужна существующая папка и новое имя STL.')
    owned = work_dir is None
    folder = create_job(settings.temp_directory) if owned else Path(work_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    report = {}
    started = time.monotonic()
    try:
        report = _run(source, settings, emit, progress, folder, report, array_output=array_output, checkpoint=checkpoint)
        if target:
            _publish_cli(report, target)
        return clean_json(report)
    except BaseException as exc:
        if isinstance(exc, Exception) and not hasattr(exc, 'report'):
            report.update(status='error', error=str(exc), elapsed_seconds=time.monotonic()-started)
            exc.report = clean_json(report)
        if owned:
            remove_job(folder)
        raise
    finally:
        if owned and target:
            remove_job(folder)


def _publish_cli(report, target):
    """Explicit --output retains STL + JSON + deviation and no-overwrite semantics."""
    from .storage import save_artifacts
    # Sidecar errors must not hide a successfully published STL (legacy contract).
    saved, _ = save_artifacts(report, target, {'stl': True}, no_replace=True)
    report.update(output=str(target), saved_at=saved['saved_at'])
    published = {'stl': str(target)}
    try:
        if report['source_deviation'].get('heatmap_error'):
            raise OSError(report['source_deviation']['heatmap_error'])
        save_artifacts(report, target, {'deviation': True}, no_replace=True)
        report['source_deviation']['heatmap_file'] = str(target)+'.deviation.npz'
        published['deviation'] = str(target)+'.deviation.npz'
    except OSError as exc:
        report['source_deviation']['heatmap_file'] = None
        report['warnings'].append('STL сохранён, но карту отклонения записать не удалось: '+str(exc))
    report['artifacts'] = published
    report.pop('work_directory', None)
    try:
        atomic_json(str(target)+'.json', report)
    except (OSError, ValueError, TypeError) as exc:
        report['warnings'].append('STL сохранён, но не удалось записать JSON-отчёт: '+str(exc))


def _export_stl(final, tmp, mm, tracker, report, emit, fallbacks):
    """Validate the serialized CLI output and retry earlier accepted stages if needed."""
    roundtrips = report['export_attempts'] = []
    while True:
        save_settings = mm.SaveSettings()
        save_settings.progress = tracker.callback(96, 97, 'Запись итогового STL')
        mm.saveMesh(final, tmp, save_settings)
        exported = mm.loadMesh(tmp)
        final_info = diagnostics(exported, mm, tracker.callback(97, 98, 'Контроль итогового STL'))
        roundtrips.append(final_info)
        if valid(final_info):
            shells_changed = final_info['components'] != report['after']['components']
            if not shells_changed or not fallbacks:
                if shells_changed:
                    # STL stores triangle soup. Its loader can split coincident
                    # voxel seams even when both meshes are closed and valid.
                    # Optional changes are rolled back first; a valid base is
                    # still usable, with the actual exported topology reported.
                    report['export_topology'] = {'before': report['after'], 'after': final_info}
                    report['warnings'].append(f"STL изменил связность при загрузке: {report['after']['components']} → "
                        f"{final_info['components']} оболочек. Замкнутость, топология и отсутствие пересечений проверены; "
                        'проверьте раздельные фрагменты. [stl_shell_count_changed]')
                break
        if not fallbacks:
            raise RuntimeError('Итоговый STL не прошёл проверку; проверенные стадии исчерпаны.')
        final, restored, stage = fallbacks.pop()
        report['after'] = restored
        report['warnings'].append(f'Итоговый STL не прошёл проверку; откат стадии [{stage}] и повторная запись.')
        if stage == 'simplification':
            from .reduction import reason_text
            reason_code = 'export_validation_failed'
            report['simplification'].update(accepted=False, after=restored['triangles'], reason=reason_code,
                                             reason_ru=reason_text(reason_code))
        elif stage == 'small_shells':
            report['removed_shells'] = [part for part in report.get('reconstruction_artifacts', {}).get('removed', [])
                                       if part['reason'] == 'small_shell']
        elif stage == 'cavities':
            report['removed_cavities'] = []
        elif stage == 'cleanup':
            report['degenerate_cleanup'].update(accepted=False, reason='export_validation_failed')
        elif stage == 'planar_finish':
            report['planar_finish'].update(accepted=False, after=restored['triangles'],
                                          reason='export_validation_failed')
        elif stage == 'exterior_recovery':
            report['exterior_recovery'].update(accepted=False, reason='export_validation_failed')
        elif stage == 'voxel_handle':
            report['voxel_handle'].update(accepted=False, reason='export_validation_failed')
    report['final_roundtrips'] = len(roundtrips)
    return exported, final_info


def _run(source, settings, emit, progress, folder, report=None, *, array_output=None, checkpoint=None):
    from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
    from .orientation import prepare_orientation, finite_vertices, is_flat_input
    from .inspection import describe_mesh
    started = time.monotonic()
    if report is None:
        report = {}
    report.update({'version': '1.0.0', 'engine_revision': 'blender-1.4.2',
              'settings': asdict(settings), 'source': str(source),
              'units': 'mm', 'tolerance_certified': False, 'warnings': [], 'steps': [],
              'compute_device': 'cpu_native_parallel', 'acceleration': capabilities(settings.disable_cuda)})
    report['stage_timings'] = []
    tracker = Progress(progress, emit, report['stage_timings'],
        lambda stage: report['compute_device'] if stage.startswith(
            ('Проба острых рёбер', 'Оценка времени', 'Восстановление поверхности', 'Уточнение воксельной поверхности'))
        else ','.join(report['acceleration'].get('projection_backends', ['cpu_native_parallel']))
        if stage == 'Карта отклонения от исходника' or stage.startswith('Строгая проверка') else 'cpu_native_parallel',
        checkpoint)
    tracker.checkpoint({'engine_revision': report['engine_revision'], 'acceleration': report['acceleration']})
    emit('compute_device=cpu_native_parallel; CUDA '+('выключена настройкой.' if settings.disable_cuda else 'авто.'))
    load_settings = mm.MeshLoadSettings()
    skipped, duplicated = mm.int_output(), mm.int_output()
    load_settings.skippedFaceCount, load_settings.duplicatedVertexCount = skipped, duplicated
    load_settings.callback = tracker.callback(0, 5, 'Загрузка сетки')
    mesh = mm.loadMesh(source, load_settings)
    finite_vertices(mn.getNumpyVerts(mesh))
    if not mesh.topology.numValidFaces():
        raise ValueError('Файл не содержит пригодных треугольников.')
    mesh.pack()
    if is_flat_input(mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology)):
        report['failure_reason'] = 'planar_input_without_thickness'
        raise ValueError('Все грани лежат в одной плоскости. Для объёмного ремонта нужна толщина. '
                         'Задайте её в Blender модификатором Solidify.')
    from .storage import check_space
    report['temporary_storage'] = check_space(folder, mesh.topology.numValidFaces())
    report['loader'] = {'skipped_faces': skipped.value, 'duplicated_vertices': duplicated.value,
                        'non_finite_vertices': 0}
    if skipped.value:
        report['warnings'].append(f'Загрузчик пропустил непригодные грани: {skipped.value}. Проверьте форму результата.')
    # Keep the loaded source for final one-way deviation, not just the preconditioned mesh.
    reference = mm.Mesh(mesh)
    mesh, orientation = prepare_orientation(mesh, mm,
        tracker.callback(5, 12, 'Согласование ориентации'),
        settings.quality_mm*.1 if settings.weld_close_vertices else 0)
    report['orientation'] = orientation
    report['steps'].append({'name': 'Согласование ориентации', **orientation})
    emit(f"Ориентация: несогласованных рёбер {orientation['disoriented_edges']}; фрагментов {orientation['components_before']} → {orientation['components_after']}; {orientation['elapsed_seconds']:.2f} с.")
    if orientation.get('conflicting_faces'):
        report['warnings'].append(f"Неориентируемых компонентов: {orientation['unoriented_components']}, граней: {orientation['conflicting_faces']}. Их ориентация пропущена; остальные компоненты обработаны.")
    if not orientation['accepted']:
        report['warnings'].append('Согласование ориентации не принято: ' + orientation.get('reason', 'unknown'))
    dense_input = reference.topology.numValidFaces() >= DENSE_MESH_FACES
    if dense_input:
        # Counting every collision and bounding every source shell is useful
        # for Check. During repair it cannot alter the decision on an open rim.
        tracker.update(12, 'Проверка границ крупной сетки')
        before = topology_info(reference, mm)
        before.update(area_mm2=reference.area(), dimensions_mm=list(reference.computeBoundingBox().size()),
                      self_intersecting_faces=None, collision_check='not_enumerated')
        report['source_shells'] = []
        report['dense_routing'] = {'threshold_triangles': DENSE_MESH_FACES,
                                   'source_triangles': before['triangles'], 'source_shell_details': 'not_enumerated'}
    else:
        before, report['source_shells'] = describe_mesh(reference, mm, tracker.callback(12, 15, 'Диагностика исходника'))
    report['before'] = before
    before['disoriented_edges'] = orientation['disoriented_edges']
    before['skipped_faces'], before['non_finite_vertices'] = skipped.value, 0
    info = topology_info(mesh, mm)
    report['prepared'] = info
    tracker.checkpoint({'before': before, 'prepared': info})
    emit(f"После согласования: треугольников {info['triangles']:,}; оболочек {info['components']}; дыр {info['boundary_loops']}.")
    from .holes import fill_simple_holes
    before_holes = mesh
    mesh, hole_report = fill_simple_holes(mesh, mm, settings.quality_mm,
        tracker.callback(18, 30, 'Заполнение простых дыр'), emit, defer_intersections=not dense_input,
        time_budget_seconds=8.0 if dense_input else 60.0)
    report['hole_filling'] = hole_report
    tracker.update(30, 'Проверка после заполнения дыр')
    if dense_input:
        # A closed valid surface can avoid voxels altogether. The null output
        # tells MeshLib to stop at the first collision instead of listing pairs.
        info = topology_info(mesh, mm)
        has_collision = None
        if info['closed'] and info['topology_valid'] and info['signed_volume_mm3'] > 0:
            has_collision = mm.findSelfCollidingTriangles(mm.MeshPart(mesh), outCollidingPairs=None)
        info.update(self_intersecting_faces=0 if has_collision is False else None,
                    collision_check='first_hit' if has_collision is not None else 'open_or_invalid_surface',
                    has_self_intersection=has_collision)
        needs_voxels = not valid(info)
        report['dense_routing'].update(needs_voxels=needs_voxels, had_intersection=has_collision)
        report['local_repair'] = {'attempted': False, 'accepted': False,
                                  'reason': 'dense_mesh_to_voxels' if needs_voxels else 'valid_surface'}
        repaired = after = None
    else:
        info = diagnostics(mesh, mm)
        repaired, after = local_repair(mesh, info, mm, settings, tracker, report, emit)
    report['after_hole_filling'] = info
    pending = hole_report['pending_local_fix']
    if pending:
        hole_report['intersections_before_local_fix'] = info['self_intersecting_faces']
        if repaired is not None:
            hole_report.update(accepted_with_local_fix=pending, pending_local_fix=0,
                               intersections_after_local_fix=after['self_intersecting_faces'])
            for item in hole_report['details']:
                if item.pop('pending_local_fix', False):
                    item['accepted_with_local_fix'] = True
            emit(f'Заплаток принято после локального исправления: {pending}.')
        else:
            # Patches can move old rim vertices, so deleting added faces is not
            # a rollback. Replay only safe patches from the untouched input.
            attempt_report = dict(hole_report)
            started_rollback = time.monotonic()
            mesh, hole_report = fill_simple_holes(before_holes, mm, settings.quality_mm, emit=emit)
            hole_report.update(rolled_back_local_fix=pending, conditional_attempt=attempt_report,
                rollback_reason=report['local_repair'].get('reason', 'local_fix_threshold_exceeded'),
                rollback_seconds=time.monotonic()-started_rollback)
            report['hole_filling'] = hole_report
            info = diagnostics(mesh, mm)
            report['after_hole_filling'] = info
            emit(f'Локальное исправление заплаток не принято; отменено заплаток: {pending}.')
    report['steps'].append({'name': 'Заполнение простых дыр', **hole_report})
    tracker.checkpoint({'after_hole_filling': info, 'hole_filling': {key: hole_report[key]
                        for key in ('holes_before', 'holes_after', 'filled', 'rejected', 'skipped')}})
    del before_holes
    if hole_report['holes_after']:
        report['warnings'].append(f"После безопасного заполнения осталось дыр: {hole_report['holes_after']}. Они переданы объёмному восстановлению.")
    t = time.monotonic()
    if repaired is not None:
        report['algorithm'] = 'local_cut_and_fill'
        from .deviation import source_deviation
        report['local_repair']['source_deviation'] = source_deviation(reference, repaired, mm,
            settings.quality_mm, None, report['acceleration'], emit=emit)
    elif valid(info):
        if info['components'] == 1:
            repaired, after = mesh, info
            report['algorithm'] = 'surface_repair'
        elif info['components'] <= 32:
            try:
                repaired = union_small(mesh, mm, tracker)
                after = diagnostics(repaired, mm)
                if not valid(after):
                    repaired = None
                report['algorithm'] = 'solid_union'
            except RuntimeError as exc:
                report['surface_attempt_error'] = str(exc)
                repaired = None
        else:
            repaired, after = mesh, info
            report['algorithm'] = 'closed_components_preserved'
    if not dense_input and repaired is None and not valid(info) and info['closed'] and info['topology_valid'] \
            and info['boundary_loops'] == 0 and info['components'] > 1 and info['self_intersecting_faces'] > 0:
        if info['components'] <= 32:
            repaired, after = union_between_shells(mesh, info, mm, settings, tracker, report, emit)
        else:
            from .solid_union import many_shell_union
            repaired, after = many_shell_union(mesh, info, mm, settings, tracker, report, emit)
    if repaired is None:
        center = mesh.computeBoundingBox().center()
        centered = mm.Mesh(mesh)
        centered.transform(mm.AffineXf3f.translation(-center))
        repaired = sparse_repair(centered, mm, settings, tracker, report, emit, translation=center)
        del centered
        after = report['reconstruction_validation']
        reason = ('Крупная повреждённая сетка после ориентации и заполнения дыр передана напрямую воксельному ремонту.'
                  if dense_input else 'После локального ремонта остались повреждения; выполнено воксельное восстановление.')
    elif report['algorithm'] == 'local_cut_and_fill':
        reason = 'Самопересечения устранены локально методом CutAndFill, без вокселизации модели.'
    elif report['algorithm'] == 'grouped_solid_union':
        reason = 'Замкнутые оболочки сборки объединены с проверкой объёма и поверхности.'
    elif report['algorithm'] == 'solid_union' and report.get('union_attempt', {}).get('accepted'):
        reason = 'Оболочки пересекались только друг с другом; объединены булевой операцией без вокселей.'
    else:
        reason = 'Ориентация и заполнение простых дыр позволили сохранить поверхность без вокселей.'
    report['algorithm_reason'] = reason
    shape_reference = mesh if report['algorithm'] == 'closed_openvdb_level_set' else None
    del mesh
    report['steps'].append({'name': 'Восстановление поверхности', 'elapsed_seconds': time.monotonic()-t,
                            'algorithm': report['algorithm'], 'reason': reason})
    emit(reason)
    if repaired is None or not repaired.topology.numValidFaces():
        raise RuntimeError('Не удалось восстановить непустую поверхность при выбранном качестве.')
    tracker.update(66, 'Проверка восстановленной сетки')
    if after is None:
        after = diagnostics(repaired, mm, tracker.callback(66, 70, 'Проверка восстановленной сетки'))
    if not valid(after):
        raise RuntimeError(f"Результат не прошёл проверку: дыр {after['boundary_loops']}; пересечений {after['self_intersecting_faces']}; "
                           f"объём {after['signed_volume_mm3']:.6g} мм³. "
                           'Оставшиеся дефекты не удалось безопасно исправить. Файл не сохранён.')
    report['after'] = after
    # Retain valid native stages for a failed final STL round-trip, never disk copies.
    fallbacks = []
    if report['algorithm'] == 'sparse_openvdb_winding':
        repaired, cleaned_after, before_cavities = drop_cavities(repaired, after, mm, report, emit, settings.quality_mm)
        if before_cavities is not None:
            if array_output is None:
                fallbacks.append((before_cavities, after, 'cavities'))
            del before_cavities
            after = cleaned_after
            report['after'] = after
    if settings.remove_small_shells:
        report['removed_shells'] = [part for part in report.get('reconstruction_artifacts', {}).get('removed', [])
                                    if part['reason'] == 'small_shell']
    if settings.remove_small_shells and after['components'] > 1:
        tracker.update(71, 'Удаление мелких оболочек')
        from .inspection import shell_summary
        regions = mm.MeshComponents.getAllComponents(mm.MeshPart(repaired))
        small = mm.FaceBitSet()
        for region in regions:
            if region.count() < 50:
                small |= region
        if small.count() and small.count() < repaired.topology.numValidFaces():
            _, removed = shell_summary(repaired.cloneRegion(small), mm)
            report['removed_shells'].extend(removed)
            if array_output is None:
                fallbacks.append((repaired, after, 'small_shells'))
            repaired = repaired.cloneRegion(repaired.topology.getValidFaces()-small)
            after = dict(topology_info(repaired, mm), self_intersecting_faces=0)
            report['after'] = after
            emit(f'Удалено мелких оболочек: {len(removed)}.')
        elif small.count():
            report['warnings'].append('Все оболочки мелкие; удаление отменено, чтобы не получить пустой результат.')
    emit(f"После ремонта: {info['triangles']:,} → {after['triangles']:,} треугольников.")
    tmp = folder/'result.stl'
    final = repaired
    if settings.collapse_coplanar or settings.simplify_curved:
        final = simplify_attempts(repaired, None, mm, settings, tracker, report, emit, folder, shape_reference)
        if final is not repaired and array_output is None:
            fallbacks.append((repaired, after, 'simplification'))
    else:
        count = mm.findDegenerateFaces(mm.MeshPart(final)).count()
        report['degenerate_faces_before_cleanup'] = count
        if count:
            final = cleanup_exported(repaired, None, mm, settings, tracker, report, folder, emit, shape_reference)
            if final is not repaired and array_output is None:
                fallbacks.append((repaired, after, 'cleanup'))
    if settings.collapse_coplanar:
        before_planar, before_planar_info = final, report['after']
        final = finish_planar_faces(repaired, final, mm, settings, tracker, report, emit, shape_reference)
        if final is not before_planar and array_output is None:
            fallbacks.append((before_planar, before_planar_info, 'planar_finish'))
    if (report['algorithm'] == 'sparse_openvdb_winding' and not settings.strict_tolerance
            and final is not repaired and final.topology.numValidFaces() < DENSE_MESH_FACES
            and reference.topology.numValidFaces() < DENSE_MESH_FACES):
        from .deviation import recover_exterior_details
        from .seams import close_voxel_handle
        tracker.update(95, 'Проверка мелких наружных деталей')
        try:
            capped, checked = close_voxel_handle(final, info, report['after'], repaired, mm, settings.quality_mm, report)
            if capped is not final:
                if array_output is None:
                    fallbacks.append((final, report['after'], 'voxel_handle'))
                final, report['after'] = capped, checked
            del capped
        except (RuntimeError, MemoryError) as exc:
            report['voxel_handle'] = {'accepted': False, 'error': str(exc)}
        detail = report['exterior_recovery'] = {}
        try:
            fitted, verified = recover_exterior_details(reference, final, repaired, mm, settings.quality_mm,
                                                        report['acceleration'], detail)
            if verified is not None:
                if array_output is None:
                    fallbacks.append((final, report['after'], 'exterior_recovery'))
                final, report['after'] = fitted, verified
            del fitted
        except (RuntimeError, MemoryError) as exc:
            detail.update(accepted=False, reason='detail_check_failed', error=str(exc))
    t = time.monotonic()
    if array_output is None:
        exported, final_info = _export_stl(final, tmp, mm, tracker, report, emit, fallbacks)
        artifacts = {'stl': str(tmp)}
        report['output'] = str(tmp)
        stage = 'Запись и повторная проверка итогового STL'
    else:
        from .transfer import write_indexed_mesh
        stage = 'Передача проверенной сетки в Blender'
        tracker.update(96, stage)
        exported, final_info = final, report['after']
        report['serialization'] = write_indexed_mesh(exported, array_output)
        artifacts = dict(zip(('vertices', 'faces'), map(str, array_output)))
        report['final_roundtrips'] = 0  # STL round-trips only.
    report['after'] = final_info
    fallbacks.clear()
    del repaired, final, shape_reference
    if settings.collapse_coplanar:
        del before_planar
    report['steps'].append({'name': stage, 'elapsed_seconds': time.monotonic()-t})
    from .deviation import source_deviation
    # Blender's on-demand colour map reads the source and indexed vertices.
    # Only a standalone export needs the separate source-distance sidecar.
    map_tmp = folder/'deviation.npz' if array_output is None else None
    t = time.monotonic()
    try:
        report['source_deviation'] = source_deviation(reference, exported, mm, settings.quality_mm,
            map_tmp, report['acceleration'], tracker.callback(98, 99, 'Карта отклонения от исходника'), emit)
    except (RuntimeError, MemoryError) as exc:
        if array_output is None:
            raise  # CLI explicitly requested a sidecar together with the STL.
        report['source_deviation'] = {'mode': 'unavailable', 'max_deviation_mm': None,
                                      'error': str(exc), 'elapsed_seconds': time.monotonic()-t}
        report['warnings'].append(f'Проверенная сетка сохранена. Не удалось измерить отклонение исходника: {exc}')
    if map_tmp is not None and report['source_deviation'].get('heatmap_error') is None and map_tmp.is_file():
        artifacts['deviation'] = str(map_tmp)
        report['source_deviation']['heatmap_file'] = str(map_tmp)
    from .planes import triangle_distribution
    tracker.update(99, 'Где остались треугольники')
    try:
        report['triangle_distribution'] = triangle_distribution(reference, exported, mm, settings.quality_mm)
    except (RuntimeError, MemoryError) as exc:
        report['warnings'].append(f'Не удалось рассчитать статистику плоскостей: {exc}')
    deviation = report['source_deviation']['max_deviation_mm']
    if deviation is not None and (not math.isfinite(deviation) or deviation > settings.quality_mm):
        report['warnings'].append(f'Отклонение наружных вершин исходника {deviation:.6g} мм '
                                  f'превышает выбранный шаг {settings.quality_mm:g} мм. Проверьте места ремонта.')
    from .inspection import shell_summary
    summary, report['shells'] = shell_summary(exported, mm, component_count=final_info['components'])
    report['after'].update(summary)
    count = report['after']['components']
    if count > 1:
        report['warnings'].append(f'В файле {count} замкнутых оболочек. Раздельные детали сохранены; перемычки не добавлялись.')
    report.update(status='repaired' if count == 1 else 'repaired_multiple_shells',
                  artifacts=artifacts, work_directory=str(folder),
                  elapsed_seconds=time.monotonic()-started)
    emit(f"Карта отклонения: {report['source_deviation']['elapsed_seconds']:.2f} с; "
         f"проекции={','.join(report['acceleration'].get('projection_backends', []))}.")
    tracker.update(100, 'Готово')
    return report
