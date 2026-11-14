"""Exercise the extracted release archive with its bundled native libraries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mesh-repair-release-smoke-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(args.archive) as archive:
            archive.extractall(root)
        package = root / "mesh_repair_blender"
        worker = package / "worker.py"
        sys.path.insert(0, str(package / "vendor"))
        from meshlib import mrmeshpy as mm

        source = root / "source.stl"
        mesh = mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(0, 0, 0))
        mm.saveMesh(mesh, source)
        settings = root / "settings.json"
        settings.write_text('{"quality_mm": 0.05, "disable_cuda": true}', encoding="utf-8")
        result = root / "result.json"
        vertices, faces = root / "vertices.npy", root / "faces.npy"
        command = [
            sys.executable, str(worker), "repair", "--source", str(source), "--settings", str(settings),
            "--job-dir", str(root / "job"), "--progress-file", str(root / "progress.jsonl"),
            "--result-file", str(result), "--vertices", str(vertices), "--faces", str(faces),
        ]
        subprocess.run(command, check=True)
        import numpy as np

        assert json.loads(result.read_text(encoding="utf-8"))["status"] == "ok"
        assert np.load(vertices).shape == (8, 3)
        assert np.load(faces).shape == (12, 3)
        print("MESH_REPAIR_RELEASE_SMOKE_OK")


if __name__ == "__main__":
    main()
