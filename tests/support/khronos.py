"""
Python side of the official Khronos glTF-Validator.

The validator is published by Khronos on npm as ``gltf-validator`` (built from
https://github.com/KhronosGroup/glTF-Validator). It is not on PyPI and not a
project dependency, so it is found through an environment variable, the same
way the Blender checks find ``bpy``::

    npm install --prefix C:/tmp/kv gltf-validator
    set NEOX_GLTF_VALIDATOR=C:/tmp/kv/node_modules/gltf-validator

and ``node`` must be on ``PATH``. :func:`available` says whether both are.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("khronos_validate.js")

#: Messages below warning level that are expected, with the reason. Anything
#: else the validator says is treated as news.
EXPECTED_INFOS = {
    (
        "UNUSED_OBJECT",
        "/meshes/0/primitives/0/attributes/TEXCOORD_0",
    ): "UVs are exported but no material samples a texture yet",
}

ERROR, WARNING, INFO, HINT = 0, 1, 2, 3


def available() -> bool:
    """True when both node and the validator package can be found."""
    location = os.environ.get("NEOX_GLTF_VALIDATOR")
    return bool(location) and Path(location).is_dir() and shutil.which("node") is not None


def validate(*paths) -> list[dict]:
    """Validate files; return one report per file, in order."""
    process = subprocess.run(
        ["node", str(SCRIPT), *(str(path) for path in paths)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"the validator did not run:\n{process.stderr[-2000:]}")
    return [json.loads(line) for line in process.stdout.splitlines() if line.strip()]


def unexpected(report: dict) -> list[dict]:
    """Every message that is an error, a warning, or an info not on the list."""
    return [
        message
        for message in report["messages"]
        if message["severity"] <= WARNING
        or (message["code"], message.get("pointer")) not in EXPECTED_INFOS
    ]


def summary(report: dict) -> str:
    """One line per report, then one per message worth reading."""
    lines = [
        f"Khronos glTF-Validator {report['validatorVersion']}: "
        f"{report['numErrors']} errors, {report['numWarnings']} warnings, "
        f"{report['numInfos']} infos, {report['numHints']} hints"
    ]
    names = {ERROR: "error", WARNING: "warning", INFO: "info", HINT: "hint"}
    for message in report["messages"]:
        key = (message["code"], message.get("pointer"))
        note = f"  (expected: {EXPECTED_INFOS[key]})" if key in EXPECTED_INFOS else ""
        lines.append(
            f"  {names.get(message['severity'], message['severity'])} "
            f"{message['code']} at {message.get('pointer', '-')}: "
            f"{message['message']}{note}"
        )
    return "\n".join(lines)
