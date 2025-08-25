"""
Pydantic models for the CLAP server API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request model for audio analysis."""
    audio_file: str = Field(..., description="Path to audio file")
    top_k: int = Field(3, description="Number of top predictions to return")
    threshold: float = Field(0.1, description="Confidence threshold")


class VibeTag(BaseModel):
    """Vibe tag with confidence score."""
    tag: str = Field(..., description="Vibe tag name")
    confidence: float = Field(..., description="Confidence score")


class AnalyzeResponse(BaseModel):
    """Response model for audio analysis."""
    vibe_tags: List[VibeTag] = Field(..., description="Top vibe tags")
    embeddings: Optional[List[float]] = Field(None, description="Audio embeddings")
    duration: float = Field(..., description="Audio duration in seconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Model type (CLAP)")
    supported_formats: List[str] = Field(..., description="Supported audio formats")
    max_audio_length: int = Field(..., description="Maximum audio length in seconds") 