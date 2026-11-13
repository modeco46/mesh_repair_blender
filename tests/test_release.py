from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_addon_has_only_required_application_layers():
    package = ROOT / "mesh_repair_blender"
    assert (package / "blender_manifest.toml").is_file()
    assert (package / "worker.py").is_file()
    assert not (package / "viewer.py").exists()
    assert not (package / "gui.py").exists()
    assert not (package / "widgets.py").exists()


def test_core_version_is_pinned():
    text = (ROOT / "UPSTREAM.md").read_text(encoding="utf-8")
    assert "6ca780f" in text
    assert "MeshRepair 1.0.0" in text
