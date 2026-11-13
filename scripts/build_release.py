"""Build a self-contained Blender add-on ZIP for one target platform."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "mesh_repair_blender"
DEPENDENCIES = ("numpy==2.2.4", "scipy==1.16.3", "meshlib==3.1.4.297")
PLATFORMS = {
    "windows-x64": ["win_amd64"],
    "linux-x64": ["manylinux_2_28_x86_64", "manylinux2014_x86_64"],
}


def download_wheels(destination: Path, target: str) -> list[Path]:
    command = [
        sys.executable, "-m", "pip", "download", "--dest", str(destination),
        "--only-binary=:all:", "--python-version", "3.13", "--implementation", "cp", "--abi", "cp313",
    ]
    for platform in PLATFORMS[target]:
        command += ["--platform", platform]
    command += list(DEPENDENCIES)
    subprocess.run(command, check=True)
    return sorted(destination.glob("*.whl"))


def extract_wheels(wheels: list[Path], vendor: Path) -> None:
    vendor.mkdir(parents=True)
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                if member.is_dir() or ".data/scripts/" in member.filename:
                    continue
                archive.extract(member, vendor)


def prune_non_runtime_assets(vendor: Path) -> None:
    """Remove MeshLib viewer/type assets; geometry and CUDA binaries stay intact."""
    meshlib = vendor / "meshlib"
    patterns = ("mrviewerpy.*", "*.pyi", "*.ttf", "*.ttc", "*Theme.json")
    for pattern in patterns:
        for path in meshlib.glob(pattern):
            path.unlink()
    # The wheel supports many Python minors through one distribution.  Blender
    # 5.3 embeds 3.13, so bridges for the other interpreters are dead weight.
    for path in meshlib.glob("libpybind11nonlimitedapi_meshlib_*"):
        if "3.13" not in path.name:
            path.unlink()


def copy_package(destination: Path) -> Path:
    package = destination / PACKAGE.name
    shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "vendor"))
    shutil.copy2(ROOT / "README.md", package / "README.md")
    shutil.copy2(ROOT / "THIRD_PARTY.md", package / "THIRD_PARTY.md")
    return package


def make_zip(staging: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging).as_posix())
    output.chmod(0o644)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=sorted(PLATFORMS), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "dist" / f"mesh-repair-blender-1.0.0-{args.target}.zip"
    with tempfile.TemporaryDirectory(prefix="mesh-repair-build-") as temporary:
        temporary = Path(temporary)
        package = copy_package(temporary / "stage")
        wheels = download_wheels(temporary / "wheels", args.target)
        if not any(wheel.name.startswith("meshlib_core-") for wheel in wheels):
            raise RuntimeError("pip did not resolve meshlib-core")
        vendor = package / "vendor"
        extract_wheels(wheels, vendor)
        prune_non_runtime_assets(vendor)
        make_zip(temporary / "stage", output.resolve())
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
