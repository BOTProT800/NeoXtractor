"""
Tests for the glTF animation writer.

Clips are synthetic on purpose: this exercises the writer independently of any
NeoX clip reader, which is the step the plan puts before reading real clips.
Every assertion evaluates the *exported file*, sampling its channels the way
the specification says an importer would.
"""

import numpy as np
import pytest

from core.mesh_converter.animation import (
    AnimationClip,
    AnimationError,
    BoneTrack,
    make_quaternions_continuous,
    rotation_clip,
    sample_track_at,
)
from core.mesh_converter.formats import glb
from core.mesh_converter.gltf_scene import MeshExportError, build_scene
from tests.support import gltf_spec_check as spec
from tests.support.gltf_reader import read_any
from tests.support.skinning import animation_overrides, skin_vertices
from tests.support.synthetic import asymmetric_character, make_mesh_data, row_vector_matrix


@pytest.fixture
def rig():
    mesh = asymmetric_character()
    return mesh, build_scene(mesh).skeleton


def export(mesh, clips):
    document = read_any(glb.convert(mesh, animations=clips))
    assert spec.check_gltf(document) == []
    return document


def node_index_of(document):
    gltf = document.json_data
    return {
        gltf["nodes"][node]["name"]: node for node in gltf["skins"][0]["joints"]
    }


class TestWriterStructure:
    def test_a_clip_becomes_samplers_and_channels(self, rig):
        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton, bone=1, axis="z", degrees_over_time=[(0.0, 0.0), (1.0, 90.0)]
        )

        document = export(mesh, [clip])
        animation = document.json_data["animations"][0]

        assert animation["name"] == "rotation"
        assert len(animation["samplers"]) == 1
        assert len(animation["channels"]) == 1
        assert animation["channels"][0]["target"]["path"] == "rotation"

    def test_the_channel_targets_the_right_bone_node(self, rig):
        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton, bone=1, axis="z", degrees_over_time=[(0.0, 0.0), (1.0, 90.0)]
        )

        document = export(mesh, [clip])
        target = document.json_data["animations"][0]["channels"][0]["target"]["node"]

        assert document.json_data["nodes"][target]["name"] == mesh.bones.names[1]

    def test_sample_times_round_trip(self, rig):
        mesh, skeleton = rig
        times = [0.0, 0.25, 0.5, 1.5]
        clip = rotation_clip(
            skeleton,
            bone=1,
            axis="z",
            degrees_over_time=[(t, 10.0 * i) for i, t in enumerate(times)],
        )

        document = export(mesh, [clip])
        sampler = document.json_data["animations"][0]["samplers"][0]
        written = document.accessor(sampler["input"])

        assert written == pytest.approx(times)
        assert clip.duration == pytest.approx(1.5)

    def test_input_accessors_declare_min_and_max(self, rig):
        """The specification requires it of animation sampler inputs."""
        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton, bone=1, axis="z", degrees_over_time=[(0.0, 0.0), (2.0, 90.0)]
        )

        document = export(mesh, [clip])
        sampler = document.json_data["animations"][0]["samplers"][0]
        accessor = document.json_data["accessors"][sampler["input"]]

        assert accessor["min"] == pytest.approx([0.0])
        assert accessor["max"] == pytest.approx([2.0])

    def test_rotation_samples_stay_unit_length(self, rig):
        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton,
            bone=3,
            axis="y",
            degrees_over_time=[(0.0, 0.0), (0.5, 137.0), (1.0, -211.0)],
        )

        document = export(mesh, [clip])
        sampler = document.json_data["animations"][0]["samplers"][0]
        values = document.accessor(sampler["output"])

        assert np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-6)

    def test_several_clips_and_tracks_are_written(self, rig):
        mesh, skeleton = rig
        first = rotation_clip(
            skeleton, bone=1, axis="z", degrees_over_time=[(0.0, 0.0), (1.0, 40.0)],
            name="left",
        )
        second = AnimationClip(
            name="both",
            tracks=[
                rotation_clip(
                    skeleton, bone=1, axis="z",
                    degrees_over_time=[(0.0, 0.0), (1.0, 20.0)],
                ).tracks[0],
                rotation_clip(
                    skeleton, bone=4, axis="x",
                    degrees_over_time=[(0.0, 0.0), (1.0, -30.0)],
                ).tracks[0],
            ],
        )

        document = export(mesh, [first, second])
        animations = document.json_data["animations"]

        assert [a["name"] for a in animations] == ["left", "both"]
        assert len(animations[1]["channels"]) == 2

    def test_a_translation_and_scale_track_is_written(self, rig):
        mesh, skeleton = rig
        clip = AnimationClip(
            name="move",
            tracks=[
                BoneTrack(
                    bone=2,
                    translation_times=np.array([0.0, 1.0]),
                    translation_values=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
                    scale_times=np.array([0.0, 1.0]),
                    scale_values=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
                )
            ],
        )

        document = export(mesh, [clip])
        paths = {
            channel["target"]["path"]
            for channel in document.json_data["animations"][0]["channels"]
        }

        assert paths == {"translation", "scale"}


class TestDeformationOverTime:
    def test_the_first_keyframe_reproduces_the_rest_pose(self, rig):
        """
        A clip whose first sample is the rest orientation must not move anything.

        This catches the classic mistake of writing a delta where glTF expects
        an absolute local transform: a channel replaces the node's TRS, it does
        not add to it.
        """
        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton, bone=1, axis="z", degrees_over_time=[(0.0, 0.0), (1.0, 90.0)]
        )

        document = export(mesh, [clip])
        rest = skin_vertices(document)
        at_zero = skin_vertices(document, animation_overrides(document, 0, 0.0))

        assert np.allclose(at_zero, rest, atol=1e-5)

    def test_the_pose_at_a_keyframe_matches_a_direct_rotation(self, rig):
        """The animated pose must equal posing that bone by hand."""
        from tests.support.skinning import rotation_override

        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton, bone=1, axis="z", degrees_over_time=[(0.0, 0.0), (1.0, 90.0)]
        )

        document = export(mesh, [clip])
        node = node_index_of(document)[mesh.bones.names[1]]

        animated = skin_vertices(document, animation_overrides(document, 0, 1.0))
        by_hand = skin_vertices(
            document, {node: rotation_override(document.json_data["nodes"][node], 90.0)}
        )

        assert np.allclose(animated, by_hand, atol=1e-5)

    def test_only_the_animated_subtree_moves(self, rig):
        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton, bone=1, axis="z", degrees_over_time=[(0.0, 0.0), (1.0, 70.0)]
        )

        document = export(mesh, [clip])
        rest = skin_vertices(document)
        posed = skin_vertices(document, animation_overrides(document, 0, 1.0))

        vertex_of = {name: index for index, name in enumerate(mesh.bones.names)}
        # Bone 1 is arm_l; its subtree is arm_l and forearm_l.
        assert not np.allclose(posed[vertex_of["arm_l"]], rest[vertex_of["arm_l"]], atol=1e-5)
        assert not np.allclose(
            posed[vertex_of["forearm_l"]], rest[vertex_of["forearm_l"]], atol=1e-5
        )
        for still in ("root", "arm_r", "forearm_r"):
            assert np.allclose(
                posed[vertex_of[still]], rest[vertex_of[still]], atol=1e-5
            ), still

    def test_returning_to_the_rest_keyframe_restores_the_geometry(self, rig):
        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton,
            bone=1,
            axis="z",
            degrees_over_time=[(0.0, 0.0), (0.5, 80.0), (1.0, 0.0)],
        )

        document = export(mesh, [clip])
        rest = skin_vertices(document)

        assert not np.allclose(
            skin_vertices(document, animation_overrides(document, 0, 0.5)), rest, atol=1e-4
        )
        assert np.allclose(
            skin_vertices(document, animation_overrides(document, 0, 1.0)), rest, atol=1e-5
        )

    def test_interpolation_is_monotonic_between_keyframes(self, rig):
        """Halfway through a 0 to 90 turn the bone must be near 45 degrees."""
        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton, bone=1, axis="z", degrees_over_time=[(0.0, 0.0), (1.0, 90.0)]
        )
        document = export(mesh, [clip])
        node = node_index_of(document)[mesh.bones.names[1]]

        from tests.support.skinning import rotation_override

        half = skin_vertices(document, animation_overrides(document, 0, 0.5))
        expected = skin_vertices(
            document, {node: rotation_override(document.json_data["nodes"][node], 45.0)}
        )

        assert np.allclose(half, expected, atol=1e-4)

    def test_values_are_clamped_outside_the_sampled_range(self, rig):
        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton, bone=1, axis="z", degrees_over_time=[(0.5, 0.0), (1.0, 60.0)]
        )
        document = export(mesh, [clip])

        before = skin_vertices(document, animation_overrides(document, 0, 0.0))
        at_start = skin_vertices(document, animation_overrides(document, 0, 0.5))
        after = skin_vertices(document, animation_overrides(document, 0, 99.0))
        at_end = skin_vertices(document, animation_overrides(document, 0, 1.0))

        assert np.allclose(before, at_start, atol=1e-6)
        assert np.allclose(after, at_end, atol=1e-6)

    def test_step_interpolation_holds_its_value(self, rig):
        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton,
            bone=1,
            axis="z",
            degrees_over_time=[(0.0, 0.0), (1.0, 90.0)],
            interpolation="STEP",
        )
        document = export(mesh, [clip])

        rest = skin_vertices(document)
        just_before = skin_vertices(document, animation_overrides(document, 0, 0.999))

        assert np.allclose(just_before, rest, atol=1e-5)

    def test_a_translation_track_moves_the_subtree_rigidly(self, rig):
        mesh, skeleton = rig
        bone = 2  # root
        rest_translation = np.asarray(skeleton.bones[bone].translation)
        offset = np.array([1.5, -2.0, 0.5])

        clip = AnimationClip(
            name="slide",
            tracks=[
                BoneTrack(
                    bone=bone,
                    translation_times=np.array([0.0, 1.0]),
                    translation_values=np.array(
                        [rest_translation, rest_translation + offset]
                    ),
                )
            ],
        )

        document = export(mesh, [clip])
        rest = skin_vertices(document)
        moved = skin_vertices(document, animation_overrides(document, 0, 1.0))

        assert np.allclose(moved, rest + offset, atol=1e-5)


class TestValidation:
    def test_a_clip_without_tracks_is_refused(self, rig):
        mesh, _ = rig

        with pytest.raises(MeshExportError, match="no tracks"):
            glb.convert(mesh, animations=[AnimationClip(name="empty")])

    def test_unsorted_sample_times_are_refused(self, rig):
        mesh, _ = rig
        clip = AnimationClip(
            name="bad",
            tracks=[
                BoneTrack(
                    bone=0,
                    translation_times=np.array([0.0, 1.0, 0.5]),
                    translation_values=np.zeros((3, 3)),
                )
            ],
        )

        with pytest.raises(MeshExportError, match="strictly increase"):
            glb.convert(mesh, animations=[clip])

    def test_mismatched_value_count_is_refused(self, rig):
        mesh, _ = rig
        clip = AnimationClip(
            name="bad",
            tracks=[
                BoneTrack(
                    bone=0,
                    translation_times=np.array([0.0, 1.0]),
                    translation_values=np.zeros((5, 3)),
                )
            ],
        )

        with pytest.raises(MeshExportError, match="expected values of shape"):
            glb.convert(mesh, animations=[clip])

    def test_a_non_unit_quaternion_is_refused(self, rig):
        mesh, _ = rig
        clip = AnimationClip(
            name="bad",
            tracks=[
                BoneTrack(
                    bone=0,
                    rotation_times=np.array([0.0, 1.0]),
                    rotation_values=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 3.0]]),
                )
            ],
        )

        with pytest.raises(MeshExportError, match="unit quaternions"):
            glb.convert(mesh, animations=[clip])

    def test_an_out_of_range_bone_is_refused(self, rig):
        mesh, _ = rig
        clip = AnimationClip(
            name="bad",
            tracks=[
                BoneTrack(
                    bone=99,
                    rotation_times=np.array([0.0]),
                    rotation_values=np.array([[0.0, 0.0, 0.0, 1.0]]),
                )
            ],
        )

        with pytest.raises(MeshExportError, match="outside of"):
            glb.convert(mesh, animations=[clip])

    def test_animating_a_baked_matrix_bone_is_refused(self):
        """
        glTF channels only address TRS.

        A bone whose rest transform carries shear is written as a matrix, and
        it cannot be animated; saying so beats writing a channel a viewer will
        ignore.
        """
        sheared = np.identity(4)
        sheared[0, 1] = 0.5
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
            bone_parents=[-1],
            bone_names=["sheared"],
            bone_matrices=[sheared],
            joints=[(0, 0, 0, 0)] * 3,
            weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
        )
        clip = AnimationClip(
            name="nope",
            tracks=[
                BoneTrack(
                    bone=0,
                    rotation_times=np.array([0.0]),
                    rotation_values=np.array([[0.0, 0.0, 0.0, 1.0]]),
                )
            ],
        )

        with pytest.raises(MeshExportError, match="baked into a matrix"):
            glb.convert(mesh, animations=[clip])

    def test_animations_without_a_skeleton_are_refused(self):
        mesh = make_mesh_data(
            positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            faces=[(0, 1, 2)],
        )
        clip = AnimationClip(
            name="nope",
            tracks=[
                BoneTrack(
                    bone=0,
                    rotation_times=np.array([0.0]),
                    rotation_values=np.array([[0.0, 0.0, 0.0, 1.0]]),
                )
            ],
        )

        with pytest.raises(MeshExportError, match="no usable skeleton"):
            glb.convert(mesh, animations=[clip])

    def test_two_tracks_for_one_bone_are_refused(self, rig):
        mesh, _ = rig
        track = BoneTrack(
            bone=0,
            rotation_times=np.array([0.0]),
            rotation_values=np.array([[0.0, 0.0, 0.0, 1.0]]),
        )

        with pytest.raises(MeshExportError, match="more than one track"):
            glb.convert(mesh, animations=[AnimationClip(name="dup", tracks=[track, track])])


class TestQuaternionContinuity:
    def test_opposite_signs_are_flipped_to_the_short_way(self):
        values = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]])

        fixed = make_quaternions_continuous(values)

        assert float(np.dot(fixed[0], fixed[1])) > 0.0
        # The orientation is unchanged: q and -q are the same rotation.
        assert np.allclose(np.abs(fixed[1]), np.abs(values[1]))

    def test_a_long_turn_stays_continuous_after_export(self, rig):
        """
        A turn past 180 degrees is where sign flips show up.

        Without the fix a viewer would interpolate the long way and the bone
        would visibly spin backwards partway through.
        """
        mesh, skeleton = rig
        clip = rotation_clip(
            skeleton,
            bone=1,
            axis="z",
            degrees_over_time=[(0.0, 0.0), (0.5, 170.0), (1.0, 340.0)],
        )

        document = export(mesh, [clip])
        sampler = document.json_data["animations"][0]["samplers"][0]
        values = document.accessor(sampler["output"])

        for index in range(1, len(values)):
            assert float(np.dot(values[index - 1], values[index])) >= 0.0

    def test_a_zero_length_quaternion_is_rejected(self):
        from core.mesh_converter.animation import normalize_quaternions

        with pytest.raises(AnimationError, match="zero length"):
            normalize_quaternions(np.array([[0.0, 0.0, 0.0, 0.0]]))


class TestSampleHelper:
    def test_sampling_falls_back_to_rest_for_untouched_channels(self):
        track = BoneTrack(
            bone=0,
            rotation_times=np.array([0.0, 1.0]),
            rotation_values=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        )

        matrix = sample_track_at(track, 0.5, ((3.0, 4.0, 5.0), (0.0, 0.0, 0.0, 1.0), (2.0, 2.0, 2.0)))

        assert matrix[:3, 3] == pytest.approx([3.0, 4.0, 5.0])
        assert np.diag(matrix)[:3] == pytest.approx([2.0, 2.0, 2.0])

    def test_cubicspline_is_not_evaluated_here(self):
        track = BoneTrack(
            bone=0,
            translation_times=np.array([0.0]),
            translation_values=np.zeros((3, 3)),
            interpolation="CUBICSPLINE",
        )

        with pytest.raises(AnimationError, match="CUBICSPLINE"):
            sample_track_at(track, 0.0, ((0, 0, 0), (0, 0, 0, 1), (1, 1, 1)))
