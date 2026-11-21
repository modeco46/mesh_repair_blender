"""Try exact union for assemblies before reconstructing them on a voxel grid."""
import math
import time
import numpy as np


def many_shell_union(mesh, info, mm, settings, tracker, report, emit):
    from meshlib import mrmeshnumpy as mn
    from .engine import diagnostics, valid

    started = time.monotonic()
    item = report['assembly_union'] = {'accepted': False, 'components': info['components']}
    budget = min(30.0, settings.union_max_seconds)
    deadline = started + budget
    # getAllComponents returns a face bitset per operand. Bound both its
    # allocation and the amount of speculative work before a voxel fallback.
    if (info['triangles'] > 1_000_000 or info['components'] > 128 or not budget):
        item['reason'] = 'assembly_work_limit'
        return None, None
    tracker.update(32, 'Проверка оболочек для Union')
    parts, volumes = [], []
    try:
        for region in mm.MeshComponents.getAllComponents(mm.MeshPart(mesh)):
            if time.monotonic() >= deadline:
                item['reason'] = 'assembly_time_budget'
                return None, None
            part = mesh.cloneRegion(region)
            volume = part.volume()
            if (not math.isfinite(volume) or volume == 0 or not part.topology.isClosed()
                    or mm.findSelfCollidingTrianglesBS(mm.MeshPart(part)).count()):
                item['reason'] = 'invalid_operand'
                return None, None
            if volume < 0:
                part.topology.flipOrientation()
            parts.append(part)
            volumes.append(abs(volume))

        params = mm.UniteManyMeshesParams()
        # Use the SDK's grouping of non-intersecting solids. Never hide a
        # Boolean failure by concatenating crossing shells or jittering them.
        params.mergeOnFail = params.forceCut = params.useRandomShifts = False
        cb = tracker.callback(32, 41, 'Объединение замкнутых оболочек')
        params.progressCb = lambda progress: cb(progress) and time.monotonic() < deadline
        candidate = mm.uniteManyMeshes(mm.std_vector_Mesh_const_ptr(parts), params)
        after = item['candidate'] = diagnostics(candidate, mm)
        if not valid(after):
            item['reason'] = 'invalid_union'
            return None, None
        slack = 5e-5 * sum(volumes)
        if not max(volumes)-slack <= after['signed_volume_mm3'] <= sum(volumes)+slack:
            item['reason'] = 'invalid_union_volume'
            return None, None
        # All output vertices must remain on an operand. Original exterior
        # vertices must remain on or inside the union, including small details.
        distance = math.sqrt(mm.findMaxDistanceSqOneWay(mm.MeshPart(mesh), mm.MeshPart(candidate)))
        limit = settings.quality_mm * .1
        item.update(max_result_to_operands_mm=distance, tolerance_mm=limit)
        if not math.isfinite(distance) or distance > limit:
            item['reason'] = 'union_drift'
            return None, None
        vertices = mn.getNumpyVerts(mesh)
        exterior = 0.
        for start in range(0, len(vertices), 65536):
            block = vertices[start:start+65536].astype(np.float32)
            points = mm.VertCoords(mn.fromNumpyArray(block))
            signed = mm.findSignedDistances(candidate, points)
            values = np.fromiter(signed.vec_, dtype=np.float32, count=len(block))
            if not np.isfinite(values).all():
                raise RuntimeError('Не удалось проверить положение исходных вершин относительно Union.')
            exterior = max(exterior, float(values.max()))
            item['max_operand_outside_mm'] = exterior
            if exterior > limit:
                item['reason'] = 'lost_exterior'
                return None, None
        item['accepted'] = True
        report['algorithm'] = 'grouped_solid_union'
        return candidate, after
    except (RuntimeError, MemoryError) as exc:
        item.update(reason='assembly_union_failed', detail=str(exc))
        return None, None
    finally:
        item['elapsed_seconds'] = time.monotonic()-started
        emit(f"Union сборки: {info['components']} оболочек, {item['elapsed_seconds']:.2f} с, "
             + ('принято.' if item['accepted'] else f"отклонено [{item.get('reason')}]."))
