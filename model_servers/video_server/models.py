"""
Pydantic models for video generation server API.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


class GenerateVideoRequest(BaseModel):
    """Request model for video generation with Wan2.2-S2V."""
    prompt: str = Field(..., description="Text prompt for video generation")
    ref_image_path: str = Field(..., description="Path to reference image")
    audio_path: str = Field(..., description="Path to audio file")
    audio_start: float = Field(0.0, description="Start time in audio file (seconds)")
    audio_duration: float = Field(4.0, description="Duration of audio clip (seconds)")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid certain content")
    num_frames: int = Field(16, description="Number of frames to generate")
    width: int = Field(512, description="Video width in pixels")
    height: int = Field(512, description="Video height in pixels")
    num_inference_steps: int = Field(25, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, description="Guidance scale for generation")
    fps: int = Field(8, description="Frames per second for output video")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    motion_bucket_id: Optional[int] = Field(127, description="Motion bucket ID for motion control")
    noise_aug_strength: Optional[float] = Field(0.02, description="Noise augmentation strength")


class GenerateVideoResponse(BaseModel):
    """Response model for video generation."""
    video_path: str = Field(..., description="Path to the generated video file")
    frames_generated: int = Field(..., description="Number of frames generated")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    model_name: str = Field(..., description="Name of the model used")
    parameters: dict = Field(..., description="Parameters used for generation")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    device: Optional[str] = Field(None, description="Device the model is running on")
    memory_usage: Optional[dict] = Field(None, description="Memory usage statistics")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str = Field(..., description="Name of the loaded model")
    model_type: str = Field(..., description="Type of model (e.g., 'text-to-video')")
    max_frames: int = Field(..., description="Maximum number of frames supported")
    max_resolution: List[int] = Field(..., description="Maximum resolution [width, height]")
    supported_formats: List[str] = Field(..., description="Supported output video formats")
    parameters: Optional[int] = Field(None, description="Number of model parameters")
    device: str = Field(..., description="Device the model is running on")
