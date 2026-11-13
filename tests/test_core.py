from pathlib import Path

from meshlib import mrmeshpy as mm

from mesh_repair_blender.core.engine import Settings, run


def test_embedded_core_repairs_hole(tmp_path):
    source = tmp_path / "open-cube.stl"
    mesh = mm.makeCube(mm.Vector3f(2, 2, 2), mm.Vector3f(0, 0, 0))
    mesh.topology.deleteFace(mm.FaceId(0))
    mm.saveMesh(mesh, source)
    report = run(source, settings=Settings(.05, disable_cuda=True), work_dir=tmp_path / "job")
    assert report["before"]["boundary_loops"] == 1
    assert report["after"]["closed"]
    assert report["after"]["self_intersecting_faces"] == 0
    assert Path(report["artifacts"]["stl"]).is_file()
