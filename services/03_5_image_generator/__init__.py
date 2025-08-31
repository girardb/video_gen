"""
Text-to-image generator service for creating reference images from storyboard prompts.

Generates reference images that will be used by the video renderer (Wan2.2-S2V).
"""

from .service import ImageGeneratorService
from .config import ImageGeneratorConfig

__all__ = ["ImageGeneratorService", "ImageGeneratorConfig"]
