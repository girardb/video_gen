"""
Configuration models for the music generator service.
"""

from typing import Optional
from pydantic import BaseModel, Field


class SunoConfig(BaseModel):
    """Configuration for Suno API."""
    base_url: str = Field("https://api.sunoapi.org", description="Suno API base URL")
    api_key_env: str = Field("SUNO_API_KEY", description="Environment variable for API key")
    model: str = Field("V4_5", description="Suno model to use")
    timeout: int = Field(300, description="Request timeout in seconds")


class APIConfig(BaseModel):
    """API configuration with provider switching."""
    provider: str = Field("suno", description="API provider: 'suno' or 'offline'")
    suno: Optional[SunoConfig] = None


class IOConfig(BaseModel):
    """Input/Output configuration."""
    input_file: str = Field("data/suno_prompt.txt", description="Input prompt file")
    output_file: str = Field("data/song.mp3", description="Output music file")
    lyrics_file: str = Field("data/lyrics.json", description="Output lyrics file")


class GenerationConfig(BaseModel):
    """Music generation parameters."""
    duration: int = Field(30, description="Duration in seconds")
    format: str = Field("mp3", description="Output format")
    sample_rate: int = Field(48000, description="Sample rate in Hz")
    channels: int = Field(2, description="Number of audio channels")


class OfflineConfig(BaseModel):
    """Offline mode configuration."""
    enabled: bool = Field(False, description="Enable offline mode")
    generate_silent: bool = Field(True, description="Generate silent MP3 for testing")
    silent_duration: float = Field(1.0, description="Duration of silent file in seconds")


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration."""
    max_retries: int = Field(3, description="Maximum retry attempts")
    retry_delay: int = Field(10, description="Delay between retries in seconds")
    fail_if_no_output: bool = Field(True, description="Fail if no output file is created")


class MusicGeneratorConfig(BaseModel):
    """Main configuration for the music generator service."""
    api: APIConfig
    io: IOConfig
    generation: GenerationConfig
    offline: OfflineConfig
    error_handling: ErrorHandlingConfig 