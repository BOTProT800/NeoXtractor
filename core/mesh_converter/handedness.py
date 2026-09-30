"""
Tell from a rig's own bone names whether an export shows it as modelled.

Every other check in this project compares an export with what the code
believes the source means, so a wrong belief passes all of them. That is how a
mirrored export survived 188 tests, and how the fix for it went the wrong way.

A rig can settle handedness without that belief. Bones named for a side
(``biped_l_hand``, ``Bip01 L Hand``, ``hand_r``) sit on the character's own
left and right, and toe bones sit ahead of the feet. Seen right-handed with Y
up, a character facing ``F`` has its left side along ``Y x F``. If the bones
named left point the other way, the export is the mirror image of the model.

This is what showed NeoX to be left-handed: a real player rig, exported
without a mirror, had its left bones on its right (agreement -0.97).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

#: Below this agreement either way, the rig does not say.
DECISIVE = 0.5

_SIDES = {"l": "l", "left": "l", "r": "r", "right": "r"}


@dataclass
class HandednessReport:
    """What the side-named bones say about an export."""

    #: Cosine between the left-minus-right direction and ``up x forward``:
    #: +1 as modelled, -1 the mirror image.
    agreement: float
    #: Left/right bone pairs that were compared.
    pairs: int
    #: Where the facing direction came from.
    forward_from: str

    @property
    def verdict(self) -> str:
        """``as modelled``, ``mirror image`` or ``undetermined``."""
        if self.agreement >= DECISIVE:
            return "as modelled"
        if self.agreement <= -DECISIVE:
            return "mirror image"
        return "undetermined"

    def describe(self) -> str:
        """One line for a log or a report."""
        return (
            f"handedness from bone names: {self.verdict} "
            f"(agreement {self.agreement:+.2f} over {self.pairs} left/right pairs, "
            f"facing from {self.forward_from})"
        )


def side_of(name: str) -> tuple[str, str] | None:
    """
    Which side a bone is named for, and its name with the side taken out.

    ``biped_l_hand`` gives ``("l", "biped_hand")``; ``Bip01 R Toe0`` gives
    ``("r", "bip01_toe0")``. Two bones pair up when the rest matches.
    """
    tokens = [token for token in re.split(r"[\s_.\-]+", name.lower()) if token]
    sides = [index for index, token in enumerate(tokens) if token in _SIDES]
    if len(sides) != 1:
        return None
    index = sides[0]
    rest = tokens[:index] + tokens[index + 1 :]
    if not rest:
        return None
    return _SIDES[tokens[index]], "_".join(rest)


def check_handedness(names, positions, up=(0.0, 1.0, 0.0)) -> HandednessReport | None:
    """
    Measure an exported rig against its own left/right names.

    Parameters:
    - names: bone names.
    - positions: each bone's rest origin **in the exported, right-handed
      space**, one row per name.
    - up: the up axis of that space.

    Returns:
    - A :class:`HandednessReport`, or None when the rig has no side-named
      pairs or nothing to tell which way it faces.
    """
    positions = np.asarray(positions, dtype=np.float64)
    up_axis = np.asarray(up, dtype=np.float64)
    up_axis = up_axis / np.linalg.norm(up_axis)

    by_key: dict[str, dict[str, int]] = {}
    for index, name in enumerate(names):
        found = side_of(name)
        if found is not None:
            side, key = found
            by_key.setdefault(key, {})[side] = index
    pairs = {key: sides for key, sides in by_key.items() if set(sides) == {"l", "r"}}
    if not pairs:
        return None

    def horizontal(vector):
        return vector - up_axis * float(np.dot(vector, up_axis))

    # Facing: toes ahead of the feet, both sides averaged.
    toes = [key for key in pairs if "toe" in key]
    feet = [key for key in pairs if "foot" in key]
    forward = None
    forward_from = ""
    if toes and feet:
        toe = np.mean([positions[pairs[key][side]] for key in toes for side in "lr"], axis=0)
        foot = np.mean([positions[pairs[key][side]] for key in feet for side in "lr"], axis=0)
        forward = horizontal(toe - foot)
        forward_from = "toes ahead of feet"
    if forward is None or np.linalg.norm(forward) < 1e-9:
        return None

    left = np.mean(
        [positions[sides["l"]] - positions[sides["r"]] for sides in pairs.values()], axis=0
    )
    left = horizontal(left)
    expected = np.cross(up_axis, forward)
    scale = np.linalg.norm(left) * np.linalg.norm(expected)
    if scale < 1e-12:
        return None
    return HandednessReport(float(np.dot(left, expected) / scale), len(pairs), forward_from)


def check_skeleton(skeleton) -> HandednessReport | None:
    """:func:`check_handedness` on a converted :class:`~.skeleton.Skeleton`."""
    names = [bone.name for bone in skeleton.bones]
    positions = [bone.global_rest[:3, 3] for bone in skeleton.bones]
    return check_handedness(names, positions)
