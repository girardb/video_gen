"""
Pydantic models for image generation server API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class GenerateImageRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid certain content")
    width: int = Field(1024, description="Image width in pixels")
    height: int = Field(1024, description="Image height in pixels")
    num_inference_steps: int = Field(20, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, description="Guidance scale for generation")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")


class GenerateImageResponse(BaseModel):
    """Response model for image generation."""
    image_path: str = Field(..., description="Path to the generated image file")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    model_name: str = Field(..., description="Name of the model used")
    parameters: dict = Field(..., description="Parameters used for generation")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    device: Optional[str] = Field(None, description="Device the model is running on")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str = Field(..., description="Name of the loaded model")
    model_type: str = Field(..., description="Type of model (e.g., 'text-to-image')")
    max_resolution: int = Field(..., description="Maximum resolution supported")
    supported_formats: list = Field(..., description="Supported output image formats")
    device: str = Field(..., description="Device the model is running on")
