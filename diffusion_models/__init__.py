"""Diffusion Models package.

This package implements various diffusion models for image generation.
"""

try:
    from ._version import version as __version__
except ImportError:
    # During development or when package is not installed,
    # the version file might not be available
    __version__ = "0.0.0+development" 