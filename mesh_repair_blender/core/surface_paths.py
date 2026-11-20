"""Bounded surface repairs before falling back to whole-volume reconstruction."""
from dataclasses import replace
import time


def repair_shells_before_union(mesh, info, damaged_regions, internal_faces,
                               mm, settings, tracker, report, emit):
    from .engine import diagnostics, repair_reconstruction_intersections, valid
    from .components import component_labels
    from meshlib import mrmeshnumpy as mn
    threshold = max(settings.local_fix_max_faces, int(info['triangles']*settings.local_fix_fraction))
    item = report['shell_local_repair'] = {'attempted': False, 'accepted': False,
        'internal_intersecting_faces': internal_faces, 'threshold_faces': threshold,
        'time_budget_seconds': 15., 'shells': []}
    if internal_faces > threshold:
        item['reason'] = 'internal_face_threshold_exceeded'
        return None
    started = time.monotonic()
    item['attempted'] = True
    labels, count = component_labels(mesh, mm)
    output = mm.Mesh()
    bounded = replace(settings, local_fix_max_faces=threshold, local_fix_fraction=0)
    try:
        for index in range(count):
            if time.monotonic()-started >= 15.:
                item['reason'] = 'shell_repair_time_budget'
                return None
            part = mesh.cloneRegion(mn.faceBitSetFromBools(labels == index))
            if index in damaged_regions:
                detail = {'acceleration': report.get('acceleration')}
                part, after = repair_reconstruction_intersections(part, diagnostics(part, mm), mm,
                    bounded, tracker, detail, emit, report_key='repair', proof_reference='shell_before_local_fix',
                    stage='Локальный ремонт оболочки перед Union', span=(31, 36), deadline=started+15.)
                item['shells'].append({'component': index+1, **detail['repair']})
                if not valid(after) or after['components'] != 1:
                    item['reason'] = 'shell_repair_invalid'
                    return None
            output.addMesh(part)
        item['accepted'] = True
        return output
    except (RuntimeError, MemoryError) as exc:
        item.update(reason='shell_repair_native_error', detail=str(exc))
        return None
    finally:
        item['elapsed_seconds'] = time.monotonic()-started


def rescue_decimated_surface(reference, candidate, info, mm, settings, tracker, report, emit):
    """Repair a few residual collisions, then check the TOTAL simplification error."""
    from .engine import repair_reconstruction_intersections, valid
    from .reduction import measured_distance
    bounded = replace(settings, local_fix_max_faces=64, local_fix_fraction=0)
    repaired, after = repair_reconstruction_intersections(candidate, info, mm, bounded, tracker, report, emit,
        report_key='repair', proof_reference='decimated_candidate_before_local_fix',
        stage='Исправление пересечений после упрощения', span=(86, 88), deadline=time.monotonic()+5.)
    if not valid(after):
        return candidate, info
    distance = measured_distance(reference, repaired, mm)
    report['total_deviation_mm'] = distance
    if not distance <= settings.quality_mm*.95:
        report['repair'].update(accepted=False, reason='repair_exceeds_total_simplification_tolerance')
        return candidate, info
    # Strict whole-surface proof, if requested, remains at final acceptance.
    return repaired, after
