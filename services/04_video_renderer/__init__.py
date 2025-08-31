"""
Video renderer service for generating video clips from storyboard prompts.

Supports multiple video generation approaches:
- Text-to-video models (primary)
- Text-to-image + image-to-video models (alternative)
"""

from .service import VideoRendererService
from .config import VideoRendererConfig

__all__ = ["VideoRendererService", "VideoRendererConfig"]
