"""
The IQE exporter, read back line by line.

The convention comes from the format's own tools
(https://github.com/lsalzman/iqm), not from this project: its Blender exporter
writes Blender's right-handed coordinates unchanged and reverses every
triangle ("Quake winding is reversed"); its compiler loads an OBJ by rotating
it to Z up (``.zxy()``, not a mirror), reversing the triangles and flipping V.
So an IQE is right-handed, like NeoX data, with clockwise front faces.

This exporter used to mirror X. That showed the model as the mirror image of
the game, which is what a person comparing the two had seen for the GLB.
"""

import numpy as np
import pytest

from core.mesh_converter.formats import iqe
from tests.support.synthetic import box_figure, reversed_storage_mesh


def read_iqe(payload: bytes) -> dict:
    """The commands this exporter writes, parsed without its code."""
    result = {"joints": [], "poses": [], "vp": [], "vn": [], "vt": [], "fm": []}
    for line in payload.decode("utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        command, values = parts[0], parts[1:]
        if command == "joint":
            result["joints"].append((values[0].strip('"'), int(values[1])))
        elif command == "pq":
            result["poses"].append([float(value) for value in values])
        elif command in ("vp", "vn", "vt"):
            result[command].append([float(value) for value in values])
        elif command == "fm":
            result["fm"].append([int(value) for value in values])
    return {
        key: np.asarray(value) if key not in ("joints",) else value
        for key, value in result.items()
    }


def test_positions_and_normals_are_not_mirrored():
    figure = box_figure()
    document = read_iqe(iqe.convert(figure))

    assert np.allclose(document["vp"], figure.mesh.position)
    assert np.allclose(document["vn"], figure.mesh.normal)
    # The staff is in the figure's left hand, at +X.
    assert document["vp"][:, 0].max() == pytest.approx(1.5)


def test_front_faces_are_clockwise():
    """Every triangle's winding runs against its stored normal, as IQE wants."""
    figure = box_figure()
    document = read_iqe(iqe.convert(figure))
    positions, normals, faces = document["vp"], document["vn"], document["fm"]

    geometric = np.cross(
        positions[faces[:, 1]] - positions[faces[:, 0]],
        positions[faces[:, 2]] - positions[faces[:, 0]],
    )
    agreement = np.einsum("ij,ij->i", geometric, normals[faces[:, 0]])
    assert (agreement < 0.0).all()


def test_joint_poses_are_local_and_not_mirrored():
    """
    Parents stored after their children, and bones turned about Z.

    ``base`` at the origin, ``mid`` at (0, 2, 0) turned 15 degrees, ``tip`` at
    (0, 4, 0) turned -30: relative to ``mid``, ``tip`` is turned -45 degrees,
    and its offset of (0, 2, 0) in the model reads as that vector turned by
    -15 degrees in ``mid``'s frame.
    """
    mesh = reversed_storage_mesh()
    document = read_iqe(iqe.convert(mesh))

    names = [name for name, _ in document["joints"]]
    assert names == ["base", "mid", "tip"], "depth first from the root"
    assert [parent for _, parent in document["joints"]] == [-1, 0, 1]

    tip = document["poses"][names.index("tip")]
    translation, (qx, qy, qz, qw) = tip[:3], tip[3:7]

    turn = np.radians(-15.0)
    up_mid = np.array([-np.sin(turn), np.cos(turn), 0.0])
    assert np.allclose(translation, 2.0 * up_mid, atol=1e-6)

    half = np.radians(-45.0) / 2.0
    expected = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    got = np.array([qx, qy, qz, qw])
    assert np.allclose(got, expected, atol=1e-6) or np.allclose(got, -expected, atol=1e-6)
