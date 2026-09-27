"""
Reader for the NeoX ``RGIS`` animation container (``.gis``).

Layout, worked out from Cyber Hunter samples and verified by walking whole
files byte for byte::

    char[4]   "RGIS"
    uint16    version                     (2 in every sample)
    uint16    unknown                     (774 in every sample)
    uint16    clip_count
    uint32    reference_bone_count
    char[32]  × reference_bone_count      bone names
    float[10] × reference_bone_count      reference pose: T(3) Q(4) S(3)
    uint32    0

    per clip:
      char[32]  clip name
      char[32]  unused, empty in every sample
      char[32]  root bone name
      uint16    bone_count
      char[32]  × bone_count              bones this clip drives
      uint16[8] header; [0] = fps, [6] = layout flag, [7] = key count
      float     × key_count               sample times, in MILLISECONDS
      per bone:
        uint8   translation_animated
        uint8   rotation_animated
        uint8   scale_animated
        uint8   padding
        float32 vec3 × (key_count if animated else 1)
        float32 quat × (key_count if animated else 1)   as (x, y, z, w)
        float16 vec3 × (key_count if animated else 1)   scale, half precision
      uint8     0

Two things were settled against the matching ``.mesh`` rather than assumed,
because getting either wrong produces a rig that looks plausible and animates
wrongly:

* **Transforms are parent-relative**, not absolute. Comparing the reference
  pose against ``jianzao_dunpai.mesh`` gave a total translation error of 2.68
  read as local versus 87.28 read as global, and six of its ten bones matched
  the mesh's local rest transform exactly.
* **Quaternions are stored ``(x, y, z, w)``.** Under that order six bones
  reproduced the mesh's local rotation exactly; under ``(w, x, y, z)`` none did.

The reference pose is *not* guaranteed to equal the mesh bind pose -- four of
the ten bones differ, all of them halves of left/right pairs -- so it is kept
for diagnostics and never used to rebuild the bind pose. The skin comes from
the ``.mesh``.

Known gap: a clip whose header flag (``header[6]``) is not 4 stores a separate
time array per bone. One clip of the 35 sampled does this. Its layout is not
worked out, so those clips are reported and skipped rather than guessed at.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"RGIS"

#: Size of a name field, NUL padded.
NAME_SIZE = 32

#: ``header[6]`` value for the layout this reader understands.
LAYOUT_SHARED_TIMES = 4

#: Sanity ceilings, so a misread length cannot ask for gigabytes.
MAX_CLIPS = 4096
MAX_BONES = 4096
MAX_KEYS = 100000


class RGISReadError(ValueError):
    """The data is not a readable RGIS file."""


class RGISUnsupportedLayout(RGISReadError):
    """A clip uses the per-bone timeline layout, which is not decoded yet."""


@dataclass
class RGISTrack:
    """One bone's channels within a clip."""

    name: str

    #: ``(n, 3)`` translations. ``n`` is the clip's key count when animated,
    #: otherwise 1.
    translation: np.ndarray
    #: ``(n, 4)`` rotations as ``(x, y, z, w)``.
    rotation: np.ndarray
    #: ``(n, 3)`` scales, read from half precision.
    scale: np.ndarray

    translation_animated: bool
    rotation_animated: bool
    scale_animated: bool
    padding: int = 0

    @property
    def animated(self) -> bool:
        """True when any channel varies over time."""
        return (
            self.translation_animated
            or self.rotation_animated
            or self.scale_animated
        )


@dataclass
class RGISClip:
    """One named animation."""

    name: str
    root_name: str
    fps: int
    #: Sample times in **seconds**, converted on read.
    times: np.ndarray
    tracks: list[RGISTrack] = field(default_factory=list)
    layout_flag: int = LAYOUT_SHARED_TIMES

    @property
    def duration(self) -> float:
        """Clip length in seconds."""
        return float(self.times[-1]) if len(self.times) else 0.0

    @property
    def key_count(self) -> int:
        """Number of sample times."""
        return len(self.times)


@dataclass
class RGISFile:
    """A parsed ``.gis`` file."""

    version: int
    unknown: int
    reference_names: list[str]
    #: ``(n, 10)`` reference pose: translation, ``(x, y, z, w)`` rotation, scale.
    reference_pose: np.ndarray
    clips: list[RGISClip] = field(default_factory=list)
    #: Clips that could not be read, as ``(name, reason)``.
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def clip(self, name: str) -> RGISClip | None:
        """Find a clip by name."""
        for item in self.clips:
            if item.name == name:
                return item
        return None

    @property
    def clip_names(self) -> list[str]:
        """Names of the clips that were read."""
        return [clip.name for clip in self.clips]


class _Cursor:
    """A bounds-checked read cursor."""

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def take(self, fmt: str):
        size = struct.calcsize(fmt)
        self._require(size)
        values = struct.unpack_from(fmt, self.data, self.offset)
        self.offset += size
        return values

    def name(self) -> str:
        self._require(NAME_SIZE)
        raw = self.data[self.offset : self.offset + NAME_SIZE]
        self.offset += NAME_SIZE
        return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")

    def array(self, count: int, components: int, dtype: str) -> np.ndarray:
        item = np.dtype(dtype).itemsize
        size = item * components * count
        self._require(size)
        values = np.frombuffer(
            self.data, dtype=dtype, count=count * components, offset=self.offset
        )
        self.offset += size
        return values.reshape(count, components).astype(np.float64)

    def _require(self, size: int) -> None:
        if self.offset + size > len(self.data):
            raise RGISReadError(
                f"needed {size} bytes at offset {self.offset} but the file holds "
                f"only {len(self.data)}"
            )

    @property
    def remaining(self) -> int:
        return len(self.data) - self.offset


def is_rgis(data: bytes) -> bool:
    """True when the data starts with the RGIS magic."""
    return data[:4] == MAGIC


def read_rgis(data: bytes, *, strict: bool = False) -> RGISFile:
    """
    Parse an RGIS animation container.

    Parameters:
    - data: the file contents.
    - strict: raise when a clip cannot be read. By default such clips are
      recorded in :attr:`RGISFile.skipped` and the rest are returned, because
      one undecoded clip should not cost the caller the other twenty-one.

    Returns:
    - The parsed file.

    Raises:
    - RGISReadError: the header is not RGIS, or a length is implausible, or
      (with ``strict``) a clip uses a layout this reader does not decode.
    """
    if not is_rgis(data):
        raise RGISReadError(f"expected magic {MAGIC!r}, found {data[:4]!r}")

    cursor = _Cursor(data)
    cursor.offset = 4
    version, unknown, clip_count, reference_count = cursor.take("<HHHI")

    if clip_count > MAX_CLIPS:
        raise RGISReadError(f"clip count {clip_count} is implausible")
    if reference_count > MAX_BONES:
        raise RGISReadError(f"reference bone count {reference_count} is implausible")

    reference_names = [cursor.name() for _ in range(reference_count)]
    reference_pose = cursor.array(reference_count, 10, "<f4")
    cursor.take("<I")  # always zero in every sample

    result = RGISFile(
        version=version,
        unknown=unknown,
        reference_names=reference_names,
        reference_pose=reference_pose,
    )

    for index in range(clip_count):
        start = cursor.offset
        try:
            result.clips.append(_read_clip(cursor))
        except RGISReadError as error:
            # The cursor is now somewhere unknown inside the clip, and clips are
            # not individually length-prefixed, so there is no way to resume.
            # Report what was read and stop rather than emit garbage.
            name = _peek_name(data, start)
            result.skipped.append((name, str(error)))
            if strict:
                raise
            break

    return result


def _peek_name(data: bytes, offset: int) -> str:
    """Read a clip name without moving a cursor, for error messages."""
    raw = data[offset : offset + NAME_SIZE]
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace") or f"@{offset}"


def _read_clip(cursor: _Cursor) -> RGISClip:
    """Read one clip record."""
    name = cursor.name()
    unused = cursor.name()
    root_name = cursor.name()
    if unused:
        # Empty in every sample; worth knowing about if it ever appears.
        raise RGISReadError(
            f"clip {name!r} has an unexpected value {unused!r} in its second "
            "name field"
        )
    (bone_count,) = cursor.take("<H")
    if bone_count > MAX_BONES:
        raise RGISReadError(f"clip {name!r} declares {bone_count} bones")
    bone_names = [cursor.name() for _ in range(bone_count)]

    header = cursor.take("<8H")
    fps, layout_flag, key_count = header[0], header[6], header[7]
    if key_count > MAX_KEYS:
        raise RGISReadError(f"clip {name!r} declares {key_count} keys")
    if layout_flag != LAYOUT_SHARED_TIMES:
        raise RGISUnsupportedLayout(
            f"clip {name!r} uses layout flag {layout_flag} (0x{layout_flag:04x}); "
            "that variant stores a separate time array per bone and is not "
            "decoded yet"
        )

    # Times are stored in milliseconds.
    times = cursor.array(key_count, 1, "<f4").reshape(-1) / 1000.0

    tracks = []
    for bone_name in bone_names:
        t_anim, r_anim, s_anim, padding = cursor.take("<4B")
        translation = cursor.array(key_count if t_anim else 1, 3, "<f4")
        rotation = cursor.array(key_count if r_anim else 1, 4, "<f4")
        scale = cursor.array(key_count if s_anim else 1, 3, "<f2")
        tracks.append(
            RGISTrack(
                name=bone_name,
                translation=translation,
                rotation=rotation,
                scale=scale,
                translation_animated=bool(t_anim),
                rotation_animated=bool(r_anim),
                scale_animated=bool(s_anim),
                padding=padding,
            )
        )

    (terminator,) = cursor.take("<B")
    if terminator != 0:
        raise RGISReadError(
            f"clip {name!r} ends with terminator {terminator}, expected 0"
        )

    return RGISClip(
        name=name,
        root_name=root_name,
        fps=fps,
        times=times,
        tracks=tracks,
        layout_flag=layout_flag,
    )
