"""
Animation clips, expressed against the skeleton contract.

This is the format-neutral side of animation support. A clip holds, per bone,
the local transform over time: exactly the quantity
:class:`~core.mesh_converter.skeleton.SkeletonBone` already exposes at rest as
``translation`` / ``rotation`` / ``scale``.

Two things are deliberate:

* **Tracks address bones by their source index**, the same identity the rest of
  the pipeline uses. A NeoX clip reader only has to resolve its own bone naming
  onto that index; it never has to know about glTF nodes or joint slots.
* **Values are absolute local transforms, not offsets.** glTF animation
  channels replace a node's TRS rather than adding to it, so a channel that
  only animates rotation leaves the rest translation in place. A clip that
  wants to express "rest pose plus a delta" has to bake that in; helpers here
  do it explicitly so the choice is visible.

Time is in seconds throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.mesh_converter.skeleton import Skeleton, compose_trs

#: Interpolation modes glTF understands.
INTERPOLATIONS = ("LINEAR", "STEP", "CUBICSPLINE")

#: Two quaternions closer than this are treated as the same orientation.
QUATERNION_EPSILON = 1e-12


class AnimationError(ValueError):
    """A clip cannot be expressed as a glTF animation."""


@dataclass
class BoneTrack:
    """
    The local transform of one bone over time.

    Each channel is optional; a bone that only rotates needs only ``rotation``.
    Channels are independent and may have different sample times, which is what
    glTF allows and what keeps a sparse clip sparse.

    Times must be non-negative and strictly increasing. Rotations are
    ``(x, y, z, w)``.
    """

    bone: int

    translation_times: np.ndarray | None = None
    translation_values: np.ndarray | None = None

    rotation_times: np.ndarray | None = None
    rotation_values: np.ndarray | None = None

    scale_times: np.ndarray | None = None
    scale_values: np.ndarray | None = None

    interpolation: str = "LINEAR"

    def channels(self):
        """Yield ``(path, times, values, components)`` for each present channel."""
        if self.translation_times is not None:
            yield "translation", self.translation_times, self.translation_values, 3
        if self.rotation_times is not None:
            yield "rotation", self.rotation_times, self.rotation_values, 4
        if self.scale_times is not None:
            yield "scale", self.scale_times, self.scale_values, 3

    @property
    def duration(self) -> float:
        """Latest sample time across this track's channels."""
        ends = [float(times[-1]) for _, times, _, _ in self.channels() if len(times)]
        return max(ends) if ends else 0.0

    def validate(self) -> list[str]:
        """Return a list of problems, empty when the track is usable."""
        problems: list[str] = []
        if self.interpolation not in INTERPOLATIONS:
            problems.append(
                f"bone {self.bone}: interpolation {self.interpolation!r} is not one of "
                f"{', '.join(INTERPOLATIONS)}"
            )

        for path, times, values, components in self.channels():
            times = np.asarray(times, dtype=np.float64)
            values = np.asarray(values, dtype=np.float64)
            where = f"bone {self.bone} {path}"

            if times.ndim != 1 or not len(times):
                problems.append(f"{where}: needs at least one sample time")
                continue
            if not np.all(np.isfinite(times)):
                problems.append(f"{where}: sample times must be finite")
            if float(times[0]) < 0.0:
                problems.append(f"{where}: sample times must not be negative")
            if len(times) > 1 and not np.all(np.diff(times) > 0.0):
                problems.append(f"{where}: sample times must strictly increase")

            expected = len(times)
            if self.interpolation == "CUBICSPLINE":
                # Cubic samples carry in-tangent, value and out-tangent.
                expected *= 3
            if values.shape != (expected, components):
                problems.append(
                    f"{where}: expected values of shape {(expected, components)}, "
                    f"got {values.shape}"
                )
                continue
            if not np.all(np.isfinite(values)):
                problems.append(f"{where}: values must be finite")

            if path == "rotation" and self.interpolation != "CUBICSPLINE":
                lengths = np.linalg.norm(values, axis=1)
                if not np.allclose(lengths, 1.0, atol=1e-4):
                    problems.append(
                        f"{where}: rotations must be unit quaternions "
                        f"(worst length {float(np.abs(lengths - 1.0).max()) + 1.0:.6f})"
                    )
        return problems


@dataclass
class AnimationClip:
    """One named animation over a set of bones."""

    name: str
    tracks: list[BoneTrack] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Length of the clip in seconds."""
        return max((track.duration for track in self.tracks), default=0.0)

    def validate(self) -> list[str]:
        """Return a list of problems, empty when the clip is usable."""
        problems: list[str] = []
        if not self.name:
            problems.append("clip has no name")
        if not self.tracks:
            problems.append(f"clip {self.name!r} has no tracks")
        seen: set[int] = set()
        for track in self.tracks:
            if track.bone in seen:
                problems.append(
                    f"clip {self.name!r} has more than one track for bone {track.bone}"
                )
            seen.add(track.bone)
            problems.extend(track.validate())
        return problems


def make_quaternions_continuous(values) -> np.ndarray:
    """
    Flip quaternions so consecutive samples take the short way round.

    ``q`` and ``-q`` are the same orientation, but a renderer interpolating
    between them travels the long way and the bone visibly spins. Sign is
    chosen so every step has a non-negative dot product with the previous one.

    Parameters:
    - values: ``(n, 4)`` array of ``(x, y, z, w)`` quaternions.

    Returns:
    - A new array with the same orientations and consistent signs.
    """
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 2 or array.shape[1] != 4:
        raise AnimationError(
            f"quaternion track must have shape (n, 4), got {array.shape}"
        )
    for index in range(1, len(array)):
        if float(np.dot(array[index - 1], array[index])) < 0.0:
            array[index] = -array[index]
    return array


def normalize_quaternions(values) -> np.ndarray:
    """Scale each quaternion to unit length, leaving orientation unchanged."""
    array = np.asarray(values, dtype=np.float64)
    lengths = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(lengths < QUATERNION_EPSILON):
        raise AnimationError("a quaternion sample has zero length")
    return array / lengths


def sample_track_at(track: BoneTrack, time: float, rest) -> np.ndarray:
    """
    Evaluate a track's local transform at one instant.

    Channels the track does not animate fall back to the bone's rest values,
    which is what glTF does. Only LINEAR and STEP are evaluated here;
    CUBICSPLINE is accepted by the writer but not by this helper.

    Parameters:
    - track: the track to evaluate.
    - time: instant in seconds.
    - rest: ``(translation, rotation, scale)`` to fall back to.

    Returns:
    - The 4x4 local transform at that instant.
    """
    if track.interpolation == "CUBICSPLINE":
        raise AnimationError("sample_track_at does not evaluate CUBICSPLINE tracks")

    rest_translation, rest_rotation, rest_scale = rest
    result = {
        "translation": np.asarray(rest_translation, dtype=np.float64),
        "rotation": np.asarray(rest_rotation, dtype=np.float64),
        "scale": np.asarray(rest_scale, dtype=np.float64),
    }

    for path, times, values, _ in track.channels():
        times = np.asarray(times, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)

        # glTF clamps outside the sampled range rather than extrapolating.
        if time <= times[0]:
            result[path] = values[0]
            continue
        if time >= times[-1]:
            result[path] = values[-1]
            continue

        upper = int(np.searchsorted(times, time))
        lower = upper - 1
        if track.interpolation == "STEP":
            result[path] = values[lower]
            continue

        span = times[upper] - times[lower]
        ratio = 0.0 if span <= 0.0 else (time - times[lower]) / span
        if path == "rotation":
            result[path] = _slerp(values[lower], values[upper], ratio)
        else:
            result[path] = values[lower] * (1.0 - ratio) + values[upper] * ratio

    return compose_trs(result["translation"], result["rotation"], result["scale"])


def _slerp(start, end, ratio: float) -> np.ndarray:
    """Spherical interpolation between two quaternions, taking the short way."""
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 1.0 - 1e-9:
        # Nearly identical; linear interpolation is both accurate and stable.
        result = a + (b - a) * ratio
        return result / np.linalg.norm(result)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (
        a * float(np.sin((1.0 - ratio) * theta) / sin_theta)
        + b * float(np.sin(ratio * theta) / sin_theta)
    )


def clips_from_rgis(
    rgis,
    skeleton: Skeleton,
    *,
    only: list[str] | None = None,
) -> tuple[list[AnimationClip], list[str]]:
    """
    Convert parsed RGIS clips onto a skeleton.

    RGIS stores parent-relative TRS per bone, which is the same quantity a glTF
    node animation channel carries, so no re-parenting is needed. Two things
    are handled here:

    * **Bones are matched by name**, never by position. A clip drives a subset
      of the rig and stores it in its own order, so index-based matching would
      silently animate the wrong limb.
    * **The skeleton's coordinate conversion is applied** to every keyframe, as
      ``C @ L @ inverse(C)``, so the animation lands in the same space as the
      geometry and the inverse bind matrices.

    Parameters:
    - rgis: a :class:`core.anim_loader.RGISFile`.
    - skeleton: the rig the clips apply to, built from the matching ``.mesh``.
    - only: clip names to convert; ``None`` converts all of them.

    Returns:
    - ``(clips, notes)`` where ``notes`` records bones a clip drives that the
      mesh does not have, and any clip that ended up with nothing to animate.
    """
    from core.mesh_converter.skeleton import decompose_trs

    index_of = {bone.name: bone.source_index for bone in skeleton.bones}
    conversion = skeleton.conversion
    inverse_conversion = conversion.inverse
    notes: list[str] = []
    clips: list[AnimationClip] = []

    for source in rgis.clips:
        if only is not None and source.name not in only:
            continue

        tracks: list[BoneTrack] = []
        missing: list[str] = []
        for track in source.tracks:
            if track.name not in index_of:
                if track.animated:
                    missing.append(track.name)
                continue
            if not track.animated:
                # A constant channel equals the node's own rest transform, so
                # writing it would only add a channel that changes nothing.
                continue

            key_count = source.key_count
            translations = np.zeros((key_count, 3))
            rotations = np.zeros((key_count, 4))
            scales = np.zeros((key_count, 3))
            for frame in range(key_count):
                t = track.translation[frame if track.translation_animated else 0]
                r = track.rotation[frame if track.rotation_animated else 0]
                s = track.scale[frame if track.scale_animated else 0]
                local = compose_trs(t, r, s)
                converted = conversion.matrix @ local @ inverse_conversion
                out_t, out_r, out_s, _ = decompose_trs(converted)
                translations[frame] = out_t
                rotations[frame] = out_r
                scales[frame] = out_s

            times = np.asarray(source.times, dtype=np.float64)
            bone_track = BoneTrack(bone=index_of[track.name])
            if track.translation_animated:
                bone_track.translation_times = times
                bone_track.translation_values = translations
            if track.rotation_animated:
                bone_track.rotation_times = times
                bone_track.rotation_values = make_quaternions_continuous(
                    normalize_quaternions(rotations)
                )
            if track.scale_animated:
                bone_track.scale_times = times
                bone_track.scale_values = scales
            tracks.append(bone_track)

        if missing:
            notes.append(
                f"clip {source.name!r} drives {len(missing)} bone(s) the mesh does "
                f"not have: {', '.join(sorted(set(missing))[:6])}"
            )
        if not tracks:
            notes.append(f"clip {source.name!r} has no channel this rig can use")
            continue
        clips.append(AnimationClip(name=source.name, tracks=tracks))

    return clips, notes


def rotation_clip(
    skeleton: Skeleton,
    bone: int,
    axis: str,
    degrees_over_time,
    *,
    name: str = "rotation",
    interpolation: str = "LINEAR",
) -> AnimationClip:
    """
    Build a clip that turns one bone about an axis of its own local frame.

    Intended for exercising the animation writer without a NeoX clip reader,
    which is the step the plan puts before reading real clips.

    Parameters:
    - skeleton: the rig the clip applies to, for the bone's rest transform.
    - bone: source index of the bone to turn.
    - axis: ``"x"``, ``"y"`` or ``"z"``.
    - degrees_over_time: iterable of ``(seconds, degrees)`` pairs.
    - name: clip name.
    - interpolation: LINEAR or STEP.

    Returns:
    - The clip. Rotation values are absolute local orientations, the rest
      orientation composed with the requested turn, because glTF channels
      replace rather than add.
    """
    from core.mesh_converter.skeleton import (  # local import avoids a cycle
        matrix_from_quaternion,
        quaternion_from_matrix,
    )

    if bone < 0 or bone >= len(skeleton.bones):
        raise AnimationError(f"bone {bone} is outside of [0, {len(skeleton.bones)})")

    rest_rotation = matrix_from_quaternion(skeleton.bones[bone].rotation)
    times, quaternions = [], []
    for seconds, degrees in degrees_over_time:
        angle = np.radians(float(degrees))
        cos, sin = np.cos(angle), np.sin(angle)
        if axis == "x":
            turn = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float64)
        elif axis == "y":
            turn = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float64)
        elif axis == "z":
            turn = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)
        else:
            raise AnimationError(f"axis must be x, y or z, got {axis!r}")
        times.append(float(seconds))
        quaternions.append(quaternion_from_matrix(rest_rotation @ turn))

    return AnimationClip(
        name=name,
        tracks=[
            BoneTrack(
                bone=bone,
                rotation_times=np.asarray(times, dtype=np.float64),
                rotation_values=make_quaternions_continuous(quaternions),
                interpolation=interpolation,
            )
        ],
    )
