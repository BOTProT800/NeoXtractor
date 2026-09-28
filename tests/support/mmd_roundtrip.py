"""
Read a PMX with mmd_tools, the reference PMX reader, and write what it built as GLB.

Run with the Blender-as-a-module interpreter::

    <blender-python> tests/support/mmd_roundtrip.py <mmd_tools checkout> model.pmx out.glb

mmd_tools (https://github.com/MMD-Blender/blender_mmd_tools) is how most
people get a PMX into Blender, and it encodes MMD's conventions: left-handed
with Y up, faces reversed. Round-tripping through it and Blender's own glTF
exporter turns "is this PMX a mirror image?" into a question about positions
that can be compared with the source. It needs the
``opencc-python-reimplemented`` package that mmd_tools ships as a wheel.
"""

import os
import sys


def main(checkout: str, source: str, destination: str) -> None:
    sys.path.insert(0, checkout)
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import mmd_tools

    mmd_tools.register()
    bpy.ops.mmd_tools.import_model(filepath=source, scale=1.0, types={"MESH", "ARMATURE"})
    bpy.ops.export_scene.gltf(filepath=destination, export_format="GLB")
    print("RESULT ok", flush=True)


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    main(*arguments)
    # Blender's leak report at interpreter exit can hang the module build.
    os._exit(0)
