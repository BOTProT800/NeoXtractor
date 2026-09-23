"""
End-to-end verification through Blender's glTF importer.

Blender cannot run inside the project environment: the ``bpy`` wheel is tied to
one CPython version and weighs hundreds of megabytes, so it is not a project
dependency. Instead these tests shell out to a separate interpreter that has
``bpy`` installed, named by the ``NEOX_BLENDER_PYTHON`` environment variable,
and skip when it is absent.

Setting one up, on Windows (``bpy`` 4.5 LTS needs CPython 3.11)::

    uv venv --python 3.11 C:/tmp/blv
    uv pip install --python C:/tmp/blv/Scripts/python.exe bpy==4.5.14
    set NEOX_BLENDER_PYTHON=C:/tmp/blv/Scripts/python.exe

Keep that path short. Blender's addon tree is deeply nested and a long prefix
pushes ``io_scene_gltf2`` past the Windows 260 character path limit, where
Python silently fails to import parts of the glTF addon.

The heavy lifting lives in :mod:`tests.support.blender_check`, which is also
usable directly on a real model::

    <blender-python> tests/support/blender_check.py model.glb
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.mesh_converter.formats import glb
from tests.support.synthetic import (
    asymmetric_character,
    make_mesh_data,
    row_vector_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "tests" / "support" / "blender_check.py"

blender_python = pytest.mark.skipif(
    not os.environ.get("NEOX_BLENDER_PYTHON"),
    reason="set NEOX_BLENDER_PYTHON to an interpreter with bpy installed",
)


def run_blender_check(payload: bytes, tmp_path: Path, name: str) -> str:
    """Write a GLB and run the Blender checker on it, returning its output."""
    interpreter = os.environ["NEOX_BLENDER_PYTHON"]
    target = tmp_path / f"{name}.glb"
    target.write_bytes(payload)

    process = subprocess.run(
        [interpreter, str(CHECKER), str(target)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=900,
    )
    output = process.stdout + process.stderr
    assert "RESULT ok" in output, output[-4000:]
    return output


def multi_root_mesh():
    return make_mesh_data(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (5.5, 0.5, 0.0)],
        faces=[(0, 1, 2), (1, 2, 3)],
        bone_parents=[-1, -1, 0],
        bone_names=["root_a", "root_b", "child_a"],
        bone_matrices=[
            row_vector_matrix((0.0, 0.0, 0.0)),
            row_vector_matrix((5.0, 0.0, 0.0)),
            row_vector_matrix((0.0, 3.0, 0.0), 25.0),
        ],
        joints=[(0, 0, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0)],
        weights=[(1.0, 0.0, 0.0, 0.0)] * 4,
    )


def reversed_storage_mesh():
    """Parents stored strictly after their children."""
    return make_mesh_data(
        positions=[(0.5, 4.0, 0.0), (0.5, 2.0, 0.0), (0.5, 0.0, 0.0)],
        faces=[(0, 1, 2)],
        bone_parents=[1, 2, -1],
        bone_names=["tip", "mid", "base"],
        bone_matrices=[
            row_vector_matrix((0.0, 4.0, 0.0), -30.0),
            row_vector_matrix((0.0, 2.0, 0.0), 15.0),
            row_vector_matrix((0.0, 0.0, 0.0)),
        ],
        joints=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)],
        weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
    )


def bone_255_mesh():
    bone_count = 257
    return make_mesh_data(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
        bone_parents=[-1] + list(range(bone_count - 1)),
        bone_names=[f"bone_{index}" for index in range(bone_count)],
        bone_matrices=[
            row_vector_matrix((index * 0.05, index * 0.02, 0.0))
            for index in range(bone_count)
        ],
        joints=[
            (255, 65535, 65535, 65535),
            (256, 65535, 65535, 65535),
            (0, 65535, 65535, 65535),
        ],
        weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
        joint_index_bits=16,
        mesh_type=5,
    )


@blender_python
@pytest.mark.parametrize(
    "name, factory",
    [
        ("character", asymmetric_character),
        ("character16", lambda: asymmetric_character(joint_index_bits=16)),
        ("multiroot", multi_root_mesh),
        ("reversed", reversed_storage_mesh),
        ("bone255", bone_255_mesh),
    ],
)
def test_blender_imports_and_deforms_the_rig(name, factory, tmp_path):
    """
    Blender resolves the skin and poses it the way the maths predicts.

    The checker verifies, inside Blender: rest geometry, that every bone head
    sits at its exported joint origin, that the bones are *not* all collapsed
    onto the origin, the parent hierarchy, the vertex groups and their weights,
    and that an armature-space delta applied to a bone moves exactly the
    vertices of its subtree and nothing else.
    """
    run_blender_check(glb.convert(factory()), tmp_path, name)
