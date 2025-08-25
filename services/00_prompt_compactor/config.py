"""
Configuration models for the prompt compactor service.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class LocalModelConfig(BaseModel):
    """Configuration for local model server."""
    server_url: str = Field("http://localhost:8001", description="LLM server URL")
    max_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI API."""
    model: str = Field("gpt-4o-mini", description="OpenAI model to use")
    max_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    api_key_env: str = Field("OPENAI_API_KEY", description="Environment variable for API key")


class ModelConfig(BaseModel):
    """Model configuration with type switching."""
    type: str = Field(..., description="Model type: 'local' or 'openai'")
    local: Optional[LocalModelConfig] = None
    openai: Optional[OpenAIConfig] = None


class IOConfig(BaseModel):
    """Input/Output configuration."""
    input_file: str = Field("data/song_brief.txt", description="Input song brief file")
    output_file: str = Field("data/suno_prompt.txt", description="Output Suno prompt file")
    max_chars: int = Field(200, description="Maximum characters in output")


class PromptConfig(BaseModel):
    """Prompt engineering configuration."""
    system_template: str = Field(..., description="System prompt template")
    user_template: str = Field(..., description="User prompt template")


class ValidationConfig(BaseModel):
    """Validation rules configuration."""
    banned_chars: List[str] = Field(default_factory=list, description="Characters not allowed in output")
    max_lines: int = Field(1, description="Maximum lines in output")
    min_chars: int = Field(10, description="Minimum characters in output")


class PromptCompactorConfig(BaseModel):
    """Main configuration for the prompt compactor service."""
    model: ModelConfig
    io: IOConfig
    prompt: PromptConfig
    validation: ValidationConfig 