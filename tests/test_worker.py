import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from meshlib import mrmeshpy as mm


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "mesh_repair_blender" / "worker.py"


def test_worker_returns_blender_arrays(tmp_path):
    source = tmp_path / "cube.stl"
    mm.saveMesh(mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(0, 0, 0)), source)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"quality_mm": .05, "disable_cuda": True}), encoding="utf-8")
    result, progress = tmp_path / "result.json", tmp_path / "progress.jsonl"
    vertices, faces = tmp_path / "vertices.npy", tmp_path / "faces.npy"
    process = subprocess.run([
        sys.executable, str(WORKER), "repair", "--source", str(source), "--settings", str(settings),
        "--job-dir", str(tmp_path / "job"), "--progress-file", str(progress),
        "--result-file", str(result), "--vertices", str(vertices), "--faces", str(faces),
    ], capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "ok"
    assert np.load(vertices).shape[1] == 3
    assert np.load(faces).shape == (12, 3)
    events = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "progress" and event["value"]["percent"] == 100 for event in events)
