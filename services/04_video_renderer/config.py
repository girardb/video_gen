"""
Configuration models for the video renderer service.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class AnimateDiffConfig(BaseModel):
    """Configuration for AnimateDiff engine."""
    model: str = Field("guoyww/animatediff-sdxl", description="AnimateDiff model ID")
    pipeline_class: str = Field("AnimateDiffPipeline", description="Pipeline class name")
    num_inference_steps: int = Field(25, description="Number of inference steps")
    guidance_scale: float = Field(7.5, description="Guidance scale")


class SkyReelsConfig(BaseModel):
    """Configuration for SkyReels engine."""
    model: str = Field("skyreels/skyreels-v1", description="SkyReels model ID")
    num_inference_steps: int = Field(20, description="Number of inference steps")
    guidance_scale: float = Field(8.0, description="Guidance scale")


class M4VConfig(BaseModel):
    """Configuration for M4V engine."""
    model: str = Field("m4v/m4v-v1", description="M4V model ID")
    num_inference_steps: int = Field(30, description="Number of inference steps")
    guidance_scale: float = Field(7.0, description="Guidance scale")


class Wan2Config(BaseModel):
    """Configuration for WAN 2.1 engine."""
    model: str = Field("wan2/wan2.1", description="WAN 2.1 model ID")
    num_inference_steps: int = Field(28, description="Number of inference steps")
    guidance_scale: float = Field(7.5, description="Guidance scale")


class StableVideoDiffusionConfig(BaseModel):
    """Configuration for Stable Video Diffusion (image-to-video)."""
    model: str = Field("stabilityai/stable-video-diffusion-img2vid-xt", description="SVD model ID")
    num_inference_steps: int = Field(25, description="Number of inference steps")
    guidance_scale: float = Field(3.0, description="Guidance scale")
    motion_bucket_id: int = Field(127, description="Motion bucket ID")
    fps_id: int = Field(6, description="FPS ID")
    noise_aug_strength: float = Field(0.1, description="Noise augmentation strength")


class EngineConfig(BaseModel):
    """Configuration for video generation engines."""
    type: Literal["AnimateDiff", "SkyReelsV1", "M4V", "Wan2.1", "StableVideoDiffusion"] = Field(
        "AnimateDiff", description="Video generation engine type"
    )
    
    # Text-to-video engines
    animatediff: AnimateDiffConfig = Field(default_factory=AnimateDiffConfig)
    skyreels: SkyReelsConfig = Field(default_factory=SkyReelsConfig)
    m4v: M4VConfig = Field(default_factory=M4VConfig)
    wan2: Wan2Config = Field(default_factory=Wan2Config)
    
    # Image-to-video engines
    stable_video_diffusion: StableVideoDiffusionConfig = Field(default_factory=StableVideoDiffusionConfig)


class VideoConfig(BaseModel):
    """Video output settings."""
    fps: int = Field(12, description="Frames per second")
    resolution: List[int] = Field([512, 512], description="Video resolution [width, height]")
    format: str = Field("mp4", description="Video format")
    codec: str = Field("h264", description="Video codec")
    bitrate: str = Field("2M", description="Video bitrate")


class GenerationConfig(BaseModel):
    """Video generation parameters."""
    batch_size: int = Field(1, description="Batch size for generation")
    num_frames: Optional[int] = Field(None, description="Number of frames (auto-calculate if None)")
    motion_bucket_id: int = Field(127, description="Motion bucket ID for SVD")
    fps_id: int = Field(6, description="FPS ID for SVD")


class ConsistencyConfig(BaseModel):
    """Consistency settings for video generation."""
    use_seed: bool = Field(True, description="Use seed from storyboard for consistency")
    use_ref_image: bool = Field(True, description="Use reference image if available")
    seed_offset: int = Field(0, description="Offset to add to storyboard seeds")


class LoRAConfig(BaseModel):
    """LoRA configuration."""
    enabled: bool = Field(False, description="Enable LoRA support")
    path: str = Field("lora/", description="Path to LoRA weights")
    weight: float = Field(0.8, description="LoRA weight")


class VideoServerConfig(BaseModel):
    """Configuration for video generation server."""
    url: str = Field("http://localhost:8004", description="Video server URL")
    timeout: int = Field(300, description="Request timeout in seconds")
    num_inference_steps: int = Field(25, description="Number of inference steps")
    guidance_scale: float = Field(7.5, description="Guidance scale")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid certain content")


class PerformanceConfig(BaseModel):
    """Performance optimization settings."""
    device: str = Field("auto", description="Device to use: auto, cpu, cuda, mps")
    memory_efficient: bool = Field(True, description="Enable memory efficient attention")
    compile: bool = Field(False, description="Use torch.compile() for optimization")
    enable_xformers: bool = Field(True, description="Enable xformers memory efficient attention")


class IOConfig(BaseModel):
    """Input/Output configuration."""
    input: str = Field("data/storyboard_with_images.json", description="Input storyboard file with ref images")
    output_dir: str = Field("out/clips", description="Output directory for video clips")
    audio_file: str = Field("data/generated_song.mp3", description="Generated song file from service 01")
    ref_image: Optional[str] = Field(None, description="Optional reference image path")


class VideoRendererConfig(BaseModel):
    """Main configuration for the video renderer service."""
    io: IOConfig = Field(default_factory=IOConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    video_server: VideoServerConfig = Field(default_factory=VideoServerConfig)
