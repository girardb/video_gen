"""
Configuration models for the storyboard generator service.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM server."""
    server_url: str = Field("http://localhost:8001", description="LLM server URL")
    max_tokens: int = Field(500, description="Maximum tokens to generate")
    temperature: float = Field(0.8, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")


class GenerationConfig(BaseModel):
    """Storyboard generation parameters."""
    shots_per_minute: int = Field(4, description="Average shots per minute of music")
    min_shot_duration: float = Field(2.0, description="Minimum shot duration in seconds")
    max_shot_duration: float = Field(8.0, description="Maximum shot duration in seconds")
    prompt_length: str = Field("20-40", description="Target word count for prompts")


class ConsistencyConfig(BaseModel):
    """Consistency settings for storyboard generation."""
    reuse_seeds: bool = Field(True, description="Reuse seeds for similar shots")
    subject_grouping: bool = Field(True, description="Group shots by subject for seed reuse")
    max_seeds: int = Field(10, description="Maximum unique seeds to use")


class ValidationConfig(BaseModel):
    """Validation rules for storyboard generation."""
    schema_file: str = Field("configs/storyboard_schema.json", description="JSON schema file")
    require_vibe_words: bool = Field(True, description="Require vibe words in prompts")
    min_prompt_words: int = Field(20, description="Minimum words in prompts")
    max_prompt_words: int = Field(40, description="Maximum words in prompts")


class IOConfig(BaseModel):
    """Input/Output configuration."""
    inputs: dict = Field(
        default={
            "lyrics": "data/lyrics.json",
            "vibe_tags": "data/vibe_tags.json"
        },
        description="Input files"
    )
    output: str = Field("out/storyboard.json", description="Output storyboard file")


class StoryboardGeneratorConfig(BaseModel):
    """Main configuration for the storyboard generator service."""
    llm: LLMConfig
    generation: GenerationConfig
    consistency: ConsistencyConfig
    validation: ValidationConfig
    io: IOConfig 