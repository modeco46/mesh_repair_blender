"""Exercise the extracted release archive with its bundled native libraries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zipfile


def write_cube(source: Path) -> None:
    vertices = (
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    )
    faces = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
        (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
    )
    with source.open("wb") as stream:
        stream.write(b"MeshRepair smoke".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(faces)))
        for face in faces:
            points = tuple(float(value) for index in face for value in vertices[index])
            stream.write(struct.pack("<12fH", 0, 0, 0, *points, 0))


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
        source = root / "source.stl"
        write_cube(source)
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
        assert json.loads(result.read_text(encoding="utf-8"))["status"] == "ok"
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        assert report["after"]["triangles"] == 12
        assert vertices.stat().st_size > 128 and faces.stat().st_size > 128
        subprocess.run([
            sys.executable, str(worker), "heatmap", "--source", str(source), "--settings", str(settings),
            "--job-dir", str(root / "map-job"), "--progress-file", str(root / "map-progress.jsonl"),
            "--result-file", str(root / "map-result.json"), "--vertices", str(vertices),
            "--distances", str(root / "distances.npy"),
        ], check=True)
        assert json.loads((root / "map-result.json").read_text(encoding="utf-8"))["status"] == "ok"
        assert (root / "distances.npy").is_file()
        print("MESH_REPAIR_RELEASE_SMOKE_OK")


if __name__ == "__main__":
    main()
