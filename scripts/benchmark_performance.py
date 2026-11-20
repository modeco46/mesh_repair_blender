"""Reproducible CPU comparison; pass a baseline checkout with --package-root.

No user's geometry or CUDA speed is inferred from these synthetic fixtures.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_root))
    from mesh_repair_blender.core import engine, inspection, reduction
    mesh = mm.makeUVSphere(1, 1000, 500)
    timings = {}
    started = time.perf_counter()
    prepared = reduction.prepare_reduction(mesh, mm)
    timings['prepare_reduction_seconds'] = time.perf_counter()-started
    assert prepared['protected_count'] == 0
    started = time.perf_counter()
    summary, shells = inspection.shell_summary(mesh, mm)
    timings['shell_summary_seconds'] = time.perf_counter()-started
    assert summary['closed_components'] == len(shells) == 1
    curved = mm.makeUVSphere(1, 320, 320)
    report = {'after': engine.diagnostics(curved, mm), 'steps': [], 'warnings': []}
    engine.finish_planar_faces(curved, curved, mm, engine.Settings(.018), engine.Progress(lambda _: None),
                              report, lambda _: None)
    timings['curved_planar_finish_seconds'] = report['planar_finish']['elapsed_seconds']
    timings['curved_planar_finish'] = report['planar_finish']
    damaged = mm.makeUVSphere(.1, 24, 24)
    vertices = mn.getNumpyVerts(damaged)
    vertices[np.argmax(vertices[:, 0])] += [0, .06, 0]
    damaged = mn.meshFromFacesVerts(mn.getNumpyFaces(damaged.topology), vertices)
    other = mm.makeUVSphere(.1, 24, 24)
    other.transform(mm.AffineXf3f.translation(mm.Vector3f(.04, 0, 0)))
    damaged.addMesh(other)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'source.stl'
        mm.saveMesh(damaged, path)
        report = engine.run(path, settings=engine.Settings(.02, disable_cuda=True, restore_sharp_edges=False),
                            work_dir=Path(folder)/'job')
        timings.update(shell_union_seconds=report['elapsed_seconds'], shell_union_algorithm=report['algorithm'],
                       revision=report['engine_revision'], shell_union_after=report['after'])
    result = {'device': 'CPU synthetic fixtures; not a CUDA benchmark',
              'component_fixture_triangles': mesh.topology.numValidFaces(), **timings}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
