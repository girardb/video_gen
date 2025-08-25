"""
Prompt compactor service implementation.

Converts a song brief into a concise Suno-compatible prompt for music generation.
"""

import os
import re
from pathlib import Path
from typing import Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from pydantic import ValidationError

from ..base import BaseService, run_service
from .config import PromptCompactorConfig


class PromptCompactorService(BaseService):
    """Service for compacting song briefs into Suno prompts."""
    
    def _create_config(self, config_data: dict) -> PromptCompactorConfig:
        """Create configuration object from data."""
        return PromptCompactorConfig(**config_data)
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        input_file = Path(self.config.io.input_file)
        if not input_file.exists():
            self.logger.error(f"Input file not found: {input_file}")
            return False
        
        # Check file size (≤ 5 KB)
        if input_file.stat().st_size > 5 * 1024:
            self.logger.error(f"Input file too large: {input_file.stat().st_size} bytes")
            return False
        
        return True
    
    def validate_outputs(self) -> bool:
        """Validate that outputs were created successfully."""
        output_file = Path(self.config.io.output_file)
        if not output_file.exists():
            self.logger.error(f"Output file not created: {output_file}")
            return False
        
        # Read and validate output
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Check character limit
        if len(content) > self.config.io.max_chars:
            self.logger.error(f"Output too long: {len(content)} chars > {self.config.io.max_chars}")
            return False
        
        # Check for banned characters
        for char in self.config.validation.banned_chars:
            if char in content:
                self.logger.error(f"Banned character found: '{char}'")
                return False
        
        # Check line count
        newline_count = content.count('\n')
        if newline_count >= self.config.validation.max_lines:
            self.logger.error(f"Too many lines: {newline_count + 1}")
            return False
        
        # Check minimum length
        if len(content) < self.config.validation.min_chars:
            self.logger.error(f"Output too short: {len(content)} chars < {self.config.validation.min_chars}")
            return False
        
        return True
    
    def _load_input(self) -> str:
        """Load the input song brief."""
        with open(self.config.io.input_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _save_output(self, prompt: str) -> None:
        """Save the generated prompt to output file."""
        # Ensure output directory exists
        output_path = Path(self.config.io.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
    
    def _generate_with_openai(self, input_text: str) -> str:
        """Generate prompt using OpenAI API."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        api_key = os.getenv(self.config.model.openai.api_key_env)
        if not api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {self.config.model.openai.api_key_env}")
        
        client = openai.OpenAI(api_key=api_key)
        
        system_prompt = self.config.prompt.system_template
        user_prompt = self.config.prompt.user_template.format(input_text=input_text)
        
        response = client.chat.completions.create(
            model=self.config.model.openai.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.config.model.openai.max_tokens,
            temperature=self.config.model.openai.temperature
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_with_local(self, input_text: str) -> str:
        """Generate prompt using local model server."""
        try:
            import requests
            
            # Format the user prompt with the template
            user_prompt = self.config.prompt.user_template.format(input_text=input_text)
            
            # Prepare request payload
            payload = {
                "prompt": user_prompt,
                "system_prompt": self.config.prompt.system_template,
                "max_tokens": self.config.model.local.max_tokens,
                "temperature": self.config.model.local.temperature,
                "top_p": self.config.model.local.top_p
            }
            
            self.logger.info(f"Sending request to LLM server: {self.config.model.local.server_url}/generate")
            self.logger.info(f"Request payload: {payload}")
            
            # Make request to LLM server
            response = requests.post(
                f"{self.config.model.local.server_url}/generate",
                json=payload,
                timeout=30
            )
            
            self.logger.info(f"LLM server response status: {response.status_code}")
            self.logger.info(f"LLM server response headers: {dict(response.headers)}")
            self.logger.info(f"LLM server response text: {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"Parsed response data: {data}")
                return data["text"]
            else:
                self.logger.error(f"LLM server request failed: {response.status_code} - {response.text}")
                raise Exception(f"LLM server returned status {response.status_code}: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Failed to call LLM server: {e}")
            raise e
    
    def _generate_fallback(self, input_text: str) -> str:
        """Fallback rule-based generation if LLM server is unavailable."""
        # Simple prompt generation based on keywords
        keywords = {
            'rock': 'energetic rock music with electric guitars and drums',
            'pop': 'catchy pop music with melodic vocals and upbeat rhythm',
            'jazz': 'smooth jazz with saxophone and piano',
            'electronic': 'electronic dance music with synthesizers and beats',
            'classical': 'orchestral classical music with strings and brass',
            'hip hop': 'hip hop music with rap vocals and heavy bass',
            'country': 'country music with acoustic guitar and twangy vocals',
            'blues': 'blues music with soulful vocals and guitar solos'
        }
        
        input_lower = input_text.lower()
        prompt_parts = []
        
        # Extract genre keywords
        for genre, description in keywords.items():
            if genre in input_lower:
                prompt_parts.append(description)
                break
        
        # Add mood keywords
        mood_keywords = {
            'happy': 'upbeat and cheerful',
            'sad': 'melancholic and emotional',
            'energetic': 'high energy and dynamic',
            'relaxed': 'calm and peaceful',
            'romantic': 'romantic and intimate',
            'epic': 'grand and cinematic'
        }
        
        for mood, description in mood_keywords.items():
            if mood in input_lower:
                prompt_parts.append(description)
                break
        
        # Add tempo hints
        if any(word in input_lower for word in ['fast', 'quick', 'upbeat']):
            prompt_parts.append('fast tempo')
        elif any(word in input_lower for word in ['slow', 'ballad', 'gentle']):
            prompt_parts.append('slow tempo')
        
        # Combine into final prompt
        if prompt_parts:
            prompt = ', '.join(prompt_parts)
        else:
            prompt = 'modern pop music with vocals and instruments'
        
        # Ensure it fits within character limit
        if len(prompt) > self.config.io.max_chars:
            prompt = prompt[:self.config.io.max_chars-3] + '...'
        
        return prompt
    
    def run(self) -> int:
        """Run the prompt compactor service."""
        try:
            self.logger.info("Starting prompt compactor service")
            
            # Validate inputs
            if not self.validate_inputs():
                return 1
            
            # Load input
            input_text = self._load_input()
            self.logger.info(f"Loaded input: {len(input_text)} characters")
            
            # Generate prompt based on model type
            if self.config.model.type == "openai":
                self.logger.info("Using OpenAI model")
                prompt = self._generate_with_openai(input_text)
            elif self.config.model.type == "local":
                self.logger.info("Using local model")
                prompt = self._generate_with_local(input_text)
            else:
                raise ValueError(f"Unknown model type: {self.config.model.type}")
            
            # Clean and validate prompt
            prompt = prompt.strip()
            prompt = re.sub(r'\s+', ' ', prompt)  # Normalize whitespace
            
            # Ensure single line
            prompt = prompt.replace('\n', ' ').replace('\r', ' ')
            
            # Remove banned characters
            for char in self.config.validation.banned_chars:
                prompt = prompt.replace(char, '')
            
            # Truncate if too long
            if len(prompt) > self.config.io.max_chars:
                prompt = prompt[:self.config.io.max_chars-3] + '...'
            
            # Save output
            self._save_output(prompt)
            self.logger.info(f"Generated prompt: {len(prompt)} characters")
            
            # Validate outputs
            if not self.validate_outputs():
                return 1
            
            self.logger.info("Prompt compactor service completed successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Prompt compactor service failed: {e}")
            return 1


def main():
    """Entry point for the prompt compactor service."""
    run_service(PromptCompactorService, "Prompt Compactor")


if __name__ == "__main__":
    main() 