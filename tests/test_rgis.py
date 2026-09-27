"""
Tests for the RGIS animation reader and its bridge onto the clip model.

Fixtures are byte-exact RGIS blobs built by ``tests.support.synthetic``, so the
reader is exercised over the real binary layout rather than a mock.
"""

import numpy as np
import pytest

from core.anim_loader import (
    RGISReadError,
    is_rgis,
    read_rgis,
)
from core.anim_loader.rgis import RGISUnsupportedLayout
from core.mesh_converter.animation import clips_from_rgis
from core.mesh_converter.formats import glb
from core.mesh_converter.gltf_scene import build_scene
from core.mesh_converter.skeleton import IDENTITY_CONVERSION, matrix_from_quaternion
from tests.support import gltf_spec_check as spec
from tests.support.gltf_reader import read_any
from tests.support.skinning import animation_overrides, skin_vertices
from tests.support.synthetic import asymmetric_character, build_rgis_file

TURN_Z_45 = (0.0, 0.0, np.sin(np.radians(22.5)), np.cos(np.radians(22.5)))
IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)


def simple_file(half_rotation=False, **overrides):
    """One clip turning `arm_l` about Z, on the asymmetric rig's bone names."""
    clip = {
        "name": "wave",
        "fps": 30,
        "times": [0.0, 0.5, 1.0],
        "tracks": [
            ("arm_l", (1.0, 2.0, 0.0), [IDENTITY_Q, TURN_Z_45, IDENTITY_Q], (1.0, 1.0, 1.0)),
            ("root", (0.0, 0.0, 0.0), IDENTITY_Q, (1.0, 1.0, 1.0)),
        ],
    }
    clip.update(overrides)
    return build_rgis_file(
        reference=[
            ("root", (0.0, 0.0, 0.0), IDENTITY_Q, (1.0, 1.0, 1.0)),
            ("arm_l", (1.0, 2.0, 0.0), IDENTITY_Q, (1.0, 1.0, 1.0)),
        ],
        clips=[clip],
        half_rotation=half_rotation,
    )


class TestReader:
    def test_magic_is_recognised(self):
        assert is_rgis(simple_file()) is True
        assert is_rgis(b"NOPE" + b"\x00" * 32) is False

    def test_a_wrong_magic_is_refused(self):
        with pytest.raises(RGISReadError, match="expected magic"):
            read_rgis(b"NOPE" + b"\x00" * 64)

    def test_header_and_reference_pose_round_trip(self):
        parsed = read_rgis(simple_file())

        assert parsed.version == 2
        assert parsed.reference_names == ["root", "arm_l"]
        assert parsed.reference_pose.shape == (2, 10)
        assert parsed.reference_pose[1, :3] == pytest.approx([1.0, 2.0, 0.0])

    def test_times_are_converted_to_seconds(self):
        """The file stores milliseconds; the reader hands back seconds."""
        parsed = read_rgis(simple_file())
        clip = parsed.clips[0]

        assert clip.times == pytest.approx([0.0, 0.5, 1.0])
        assert clip.duration == pytest.approx(1.0)
        assert clip.key_count == 3

    def test_animated_and_constant_channels_are_distinguished(self):
        parsed = read_rgis(simple_file())
        track = parsed.clips[0].tracks[0]

        assert track.rotation_animated is True
        assert track.translation_animated is False
        assert track.scale_animated is False
        assert track.rotation.shape == (3, 4)
        assert track.translation.shape == (1, 3)
        assert track.animated is True

        still = parsed.clips[0].tracks[1]
        assert still.animated is False

    def test_scale_survives_half_precision(self):
        data = build_rgis_file(
            reference=[("root", (0.0, 0.0, 0.0), IDENTITY_Q, (1.0, 1.0, 1.0))],
            clips=[
                {
                    "name": "grow",
                    "times": [0.0, 1.0],
                    "tracks": [
                        (
                            "root",
                            (0.0, 0.0, 0.0),
                            IDENTITY_Q,
                            [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
                        )
                    ],
                }
            ],
        )

        track = read_rgis(data).clips[0].tracks[0]

        assert track.scale_animated is True
        # float16 holds these exactly.
        assert track.scale[1] == pytest.approx([2.0, 2.0, 2.0])

    def test_quaternions_are_read_as_xyzw(self):
        """
        Order settled against the matching mesh, not assumed.

        A 45 degree turn about Z is ``(0, 0, sin22.5, cos22.5)``; read as
        ``(w, x, y, z)`` it would come out as a turn about Y instead.
        """
        parsed = read_rgis(simple_file())
        rotation = parsed.clips[0].tracks[0].rotation[1]

        assert rotation == pytest.approx(TURN_Z_45)
        basis = matrix_from_quaternion(rotation)
        assert basis[2, 2] == pytest.approx(1.0)  # Z axis is the one held fixed

    def test_the_per_bone_timeline_variant_is_reported_not_guessed(self):
        """
        Bit 0x0100 gives each bone its own time array.

        That layout is not worked out, so it is skipped with a reason rather
        than parsed into plausible-looking nonsense.
        """
        parsed = read_rgis(simple_file(layout_flag=0x0104))

        assert parsed.clips == []
        assert len(parsed.skipped) == 1
        name, reason = parsed.skipped[0]
        assert name == "wave"
        assert "0x0104" in reason

    def test_strict_mode_raises_on_the_per_bone_timeline_variant(self):
        with pytest.raises(RGISUnsupportedLayout, match="0x0104"):
            read_rgis(simple_file(layout_flag=0x0104), strict=True)

    def test_an_unrecognised_layout_bit_is_refused(self):
        """A bit nobody has explained must not be silently ignored."""
        with pytest.raises(RGISUnsupportedLayout, match="unrecognised bits"):
            read_rgis(simple_file(layout_flag=0x0044), strict=True)

    def test_half_precision_rotations_are_read_when_the_flag_says_so(self):
        """
        Bit 0x0002 halves the rotation payload.

        Sizing it wrong shifts every following bone, so it is read from the
        flag rather than sniffed. This is what the samples of the user's own
        model use.
        """
        full = read_rgis(simple_file())
        half = read_rgis(simple_file(layout_flag=0x0006, half_rotation=True))

        a = full.clips[0].tracks[0].rotation
        b = half.clips[0].tracks[0].rotation
        assert a.shape == b.shape
        # float16 keeps these to about three decimals.
        assert b[1] == pytest.approx(a[1], abs=2e-3)
        assert np.allclose(np.linalg.norm(b, axis=1), 1.0, atol=2e-3)

    def test_the_frame_rate_comes_from_the_times_not_the_header(self):
        """``header[0]`` is 30 in some files and 0xFFFF in others."""
        parsed = read_rgis(simple_file(times=[0.0, 1 / 60, 2 / 60]))

        assert parsed.clips[0].fps == 60

    def test_a_truncated_file_is_refused(self):
        data = simple_file()

        with pytest.raises(RGISReadError, match="needed"):
            read_rgis(data[: len(data) // 2], strict=True)

    def test_an_implausible_clip_count_is_refused(self):
        data = bytearray(simple_file())
        data[8:10] = (9999).to_bytes(2, "little")

        with pytest.raises(RGISReadError, match="implausible"):
            read_rgis(bytes(data))


class TestBridge:
    def test_bones_are_matched_by_name_not_position(self):
        """
        A clip drives a subset of the rig, in its own order.

        `arm_l` is source bone 1 of the fixture but track 0 of the clip, so an
        index-based bridge would animate the wrong limb.
        """
        mesh = asymmetric_character()
        skeleton = build_scene(mesh, conversion=IDENTITY_CONVERSION).skeleton
        parsed = read_rgis(simple_file())

        clips, notes = clips_from_rgis(parsed, skeleton)

        assert notes == []
        assert len(clips) == 1
        assert [track.bone for track in clips[0].tracks] == [
            mesh.bones.names.index("arm_l")
        ]

    def test_constant_channels_are_not_written(self):
        """They equal the node's rest transform, so a channel would add nothing."""
        mesh = asymmetric_character()
        skeleton = build_scene(mesh, conversion=IDENTITY_CONVERSION).skeleton

        clips, _ = clips_from_rgis(read_rgis(simple_file()), skeleton)

        track = clips[0].tracks[0]
        assert track.rotation_times is not None
        assert track.translation_times is None
        assert track.scale_times is None

    def test_a_bone_the_mesh_lacks_is_reported(self):
        mesh = asymmetric_character()
        skeleton = build_scene(mesh, conversion=IDENTITY_CONVERSION).skeleton
        data = build_rgis_file(
            reference=[("ghost", (0.0, 0.0, 0.0), IDENTITY_Q, (1.0, 1.0, 1.0))],
            clips=[
                {
                    "name": "ghostly",
                    "times": [0.0, 1.0],
                    "tracks": [
                        ("ghost", (0.0, 0.0, 0.0), [IDENTITY_Q, TURN_Z_45], (1.0, 1.0, 1.0))
                    ],
                }
            ],
        )

        clips, notes = clips_from_rgis(read_rgis(data), skeleton)

        assert clips == []
        assert any("does not have" in note for note in notes)
        assert any("no channel this rig can use" in note for note in notes)

    def test_only_selects_the_named_clips(self):
        mesh = asymmetric_character()
        skeleton = build_scene(mesh, conversion=IDENTITY_CONVERSION).skeleton

        clips, _ = clips_from_rgis(read_rgis(simple_file()), skeleton, only=["nope"])

        assert clips == []

    def test_the_coordinate_conversion_reaches_the_animation(self):
        """
        Geometry and animation must land in the same space.

        The default export mirrors X. A turn about Z seen through that mirror
        is the same angle the other way, so the exported quaternion's Z
        component flips sign relative to the unconverted one.
        """
        mesh = asymmetric_character()
        parsed = read_rgis(simple_file())

        plain = build_scene(mesh, conversion=IDENTITY_CONVERSION).skeleton
        mirrored = build_scene(mesh).skeleton

        unconverted, _ = clips_from_rgis(parsed, plain)
        converted, _ = clips_from_rgis(parsed, mirrored)

        a = unconverted[0].tracks[0].rotation_values[1]
        b = converted[0].tracks[0].rotation_values[1]
        assert a[2] == pytest.approx(-b[2], abs=1e-6)
        assert abs(a[3]) == pytest.approx(abs(b[3]), abs=1e-6)


class TestEndToEnd:
    def test_a_read_clip_exports_and_deforms(self):
        mesh = asymmetric_character()
        skeleton = build_scene(mesh).skeleton
        clips, _ = clips_from_rgis(read_rgis(simple_file()), skeleton)

        document = read_any(glb.convert(mesh, animations=clips))
        assert spec.check_gltf(document) == []

        rest = skin_vertices(document)
        middle = skin_vertices(document, animation_overrides(document, 0, 0.5))
        end = skin_vertices(document, animation_overrides(document, 0, 1.0))

        vertex_of = {name: i for i, name in enumerate(mesh.bones.names)}
        assert not np.allclose(
            middle[vertex_of["arm_l"]], rest[vertex_of["arm_l"]], atol=1e-4
        )
        # The clip returns to identity, so the last key restores the rest pose.
        assert np.allclose(end, rest, atol=1e-5)
        # The other arm is not in the clip and must not move.
        assert np.allclose(
            middle[vertex_of["arm_r"]], rest[vertex_of["arm_r"]], atol=1e-6
        )

    def test_the_animation_is_named_after_the_clip(self):
        mesh = asymmetric_character()
        skeleton = build_scene(mesh).skeleton
        clips, _ = clips_from_rgis(read_rgis(simple_file()), skeleton)

        document = read_any(glb.convert(mesh, animations=clips))

        assert [a["name"] for a in document.json_data["animations"]] == ["wave"]
