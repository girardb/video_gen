"""
Pydantic models for the LLM server API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    max_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    text: str = Field(..., description="Generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    model_name: str = Field(..., description="Name of the model used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Name of the model")
    model_path: str = Field(..., description="Path to the model")
    max_context_length: int = Field(..., description="Maximum context length")
    parameters: int = Field(..., description="Number of parameters")
    quantization: Optional[str] = Field(None, description="Quantization type") 