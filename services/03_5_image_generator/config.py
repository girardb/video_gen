"""
Configuration models for the text-to-image generator service.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ImageServerConfig(BaseModel):
    """Configuration for image generation server."""
    url: str = Field("http://localhost:8005", description="Image server URL")
    timeout: int = Field(120, description="Request timeout in seconds")
    num_inference_steps: int = Field(20, description="Number of inference steps")
    guidance_scale: float = Field(7.5, description="Guidance scale")
    negative_prompt: Optional[str] = Field("blurry, low quality, distorted", description="Negative prompt")


class ImageModelConfig(BaseModel):
    """Configuration for image generation models."""
    type: Literal["Qwen", "SDXL", "SD15", "SD21"] = Field("Qwen", description="Image model type")
    model_id: str = Field("Qwen/Qwen-Image", description="Model ID")
    resolution: List[int] = Field([1664, 928], description="Image resolution [width, height]")
    aspect_ratio: str = Field("16:9", description="Aspect ratio (1:1, 16:9, 9:16, etc.)")


class ConsistencyConfig(BaseModel):
    """Consistency settings for image generation."""
    use_seed: bool = Field(True, description="Use seed from storyboard for consistency")
    seed_offset: int = Field(1000, description="Offset to add to storyboard seeds")
    character_consistency: bool = Field(True, description="Try to maintain character consistency")
    style_prompt_suffix: str = Field("high quality, detailed, cinematic lighting", description="Style suffix added to all prompts")


class ProcessingConfig(BaseModel):
    """Processing settings for storyboard enhancement."""
    enhance_prompts: bool = Field(True, description="Enhance prompts for better image generation")
    batch_size: int = Field(1, description="Batch size for generation")
    skip_existing: bool = Field(True, description="Skip generation if image already exists")


class IOConfig(BaseModel):
    """Input/Output configuration."""
    input: str = Field("data/storyboard.json", description="Input storyboard file")
    output: str = Field("data/storyboard_with_images.json", description="Output enhanced storyboard file")
    image_dir: str = Field("out/reference_images", description="Directory to save reference images")


class ImageGeneratorConfig(BaseModel):
    """Main configuration for the text-to-image generator service."""
    io: IOConfig = Field(default_factory=IOConfig)
    image_server: ImageServerConfig = Field(default_factory=ImageServerConfig)
    model: ImageModelConfig = Field(default_factory=ImageModelConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
