import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from meshlib import mrmeshpy as mm

from mesh_repair_blender.heatmap import colors


def test_native_map_measures_result_vertices_to_source_triangles(tmp_path):
    source = tmp_path / "source.stl"
    mm.saveMesh(mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(0, 0, 0)), source)
    vertices = tmp_path / "vertices.npy"
    np.save(vertices, np.array([[.5, .5, 1.25], [.5, .5, 1], [.5, .5, -.1]], dtype=np.float32))
    settings = tmp_path / "settings.json"
    settings.write_text('{"disable_cuda": true}', encoding="utf-8")
    result, distances = tmp_path / "result.json", tmp_path / "distances.npy"
    worker = Path(__file__).resolve().parents[1] / "mesh_repair_blender" / "worker.py"
    process = subprocess.run([
        sys.executable, str(worker), "heatmap", "--source", str(source), "--settings", str(settings),
        "--job-dir", str(tmp_path / "job"), "--progress-file", str(tmp_path / "progress.jsonl"),
        "--result-file", str(result), "--vertices", str(vertices), "--distances", str(distances),
    ], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "ok"
    np.testing.assert_allclose(np.load(distances), [.25, 0, .1], atol=1e-6)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["acceleration"]["projection_backends"] == ["cpu_native_parallel"]


def test_palette_tolerance_and_saturation():
    palette = colors(np.array([0, .02, .04, .2]), .02)
    assert np.argmax(palette[0, :3]) == 2  # blue
    assert np.argmax(palette[1, :3]) == 1  # green
    assert np.argmax(palette[2, :3]) == 0  # red
    np.testing.assert_array_equal(palette[2], palette[3])
