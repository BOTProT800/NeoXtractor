"""
Readers for NeoX skeletal animation resources.

Currently one format: ``RGIS`` (``.gis``), see :mod:`core.anim_loader.rgis`.
"""

from .rgis import (
    RGISClip,
    RGISFile,
    RGISReadError,
    RGISTrack,
    is_rgis,
    looks_like_bare_clip,
    read_gis,
    read_rgis,
)

__all__ = [
    "RGISClip",
    "RGISFile",
    "RGISTrack",
    "RGISReadError",
    "read_rgis",
    "read_gis",
    "is_rgis",
    "looks_like_bare_clip",
]
