"""
Configuration models for the audio analyser service.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CLAPConfig(BaseModel):
    """Configuration for CLAP server."""
    server_url: str = Field("http://localhost:8002", description="CLAP server URL")
    top_k: int = Field(3, description="Number of top vibe tags to return")
    threshold: float = Field(0.1, description="Confidence threshold")


class BeatDetectionConfig(BaseModel):
    """Configuration for beat detection."""
    method: str = Field("librosa", description="Beat detection method")
    hop_length: int = Field(512, description="Hop length for analysis")
    sr: int = Field(22050, description="Sample rate for analysis")
    onset_threshold: float = Field(0.5, description="Onset detection threshold")
    units: str = Field("time", description="Output units: 'time' or 'frames'")


class ProcessingConfig(BaseModel):
    """Audio processing configuration."""
    resample_rate: int = Field(22050, description="Resample rate for analysis")
    normalize: bool = Field(True, description="Normalize audio")
    trim_silence: bool = Field(True, description="Trim silence from audio")


class IOConfig(BaseModel):
    """Input/Output configuration."""
    inputs: dict = Field(
        default={
            "audio": "data/song.mp3",
            "lyrics": "data/lyrics.json"
        },
        description="Input files"
    )
    outputs: dict = Field(
        default={
            "lyrics": "data/lyrics.json",
            "vibe_tags": "data/vibe_tags.json",
            "beats": "data/beats.npy"
        },
        description="Output files"
    )


class OutputConfig(BaseModel):
    """Output format configuration."""
    lyrics_format: str = Field("timed", description="Lyrics format: 'timed' or 'raw'")
    vibe_format: str = Field("top3", description="Vibe format: 'top3', 'all', or 'threshold'")
    beats_format: str = Field("numpy", description="Beats format: 'numpy' or 'json'")


class AudioAnalyserConfig(BaseModel):
    """Main configuration for the audio analyser service."""
    clap: CLAPConfig
    beat_detection: BeatDetectionConfig
    processing: ProcessingConfig
    io: IOConfig
    output: OutputConfig 