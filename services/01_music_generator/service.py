"""
Music generator service implementation.

Generates music using Suno API or creates placeholder files for testing.
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import soundfile as sf

from ..base import BaseService, run_service
from .config import MusicGeneratorConfig


class MusicGeneratorService(BaseService):
    """Service for generating music using Suno API or offline mode."""
    
    def _create_config(self, config_data: dict) -> MusicGeneratorConfig:
        """Create configuration object from data."""
        return MusicGeneratorConfig(**config_data)
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        input_file = Path(self.config.io.input_file)
        if not input_file.exists():
            self.logger.error(f"Input file not found: {input_file}")
            return False
        
        return True
    
    def validate_outputs(self) -> bool:
        """Validate that outputs were created successfully."""
        output_file = Path(self.config.io.output_file)
        
        # Check for both MP3 and WAV files
        wav_file = output_file.with_suffix('.wav')
        
        if output_file.exists():
            # MP3 file exists
            if output_file.stat().st_size == 0:
                self.logger.error(f"Output file is empty: {output_file}")
                return False
            return True
        elif wav_file.exists():
            # WAV file exists (fallback)
            self.logger.info(f"WAV file created instead of MP3: {wav_file}")
            if wav_file.stat().st_size == 0:
                self.logger.error(f"Output file is empty: {wav_file}")
                return False
            return True
        else:
            self.logger.error(f"Output file not created: {output_file}")
            return False
    
    def _load_input(self) -> str:
        """Load the input prompt."""
        with open(self.config.io.input_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _generate_silent_audio(self, duration: float, sample_rate: int, channels: int) -> np.ndarray:
        """Generate silent audio for testing."""
        num_samples = int(duration * sample_rate)
        audio = np.zeros((num_samples, channels), dtype=np.float32)
        return audio
    
    def _save_audio(self, audio: np.ndarray, output_path: str, sample_rate: int) -> None:
        """Save audio to file."""
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(output_path, audio, sample_rate)
    
    def _save_lyrics(self, lyrics: str, title: str = "") -> None:
        """Save lyrics to file."""
        import json
        
        # Ensure output directory exists
        lyrics_path = Path(self.config.io.lyrics_file)
        lyrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse lyrics with verse/chorus structure if available
        lyrics_data = {
            "title": title,
            "text": lyrics,
            "structured_lyrics": self._parse_lyrics_structure(lyrics),
            "metadata": {
                "source": "suno_api",
                "format": "structured"
            }
        }
        
        with open(lyrics_path, 'w', encoding='utf-8') as f:
            json.dump(lyrics_data, f, indent=2)
    
    def _parse_lyrics_structure(self, lyrics: str) -> List[Dict]:
        """Parse Suno's structured lyrics format ([Verse], [Chorus], etc.)."""
        import re
        
        # Split lyrics into sections based on [Section] markers
        sections = []
        current_section = {"type": "intro", "text": ""}
        
        lines = lyrics.split('\n')
        for line in lines:
            line = line.strip()
            
            # Check for section markers like [Verse], [Chorus], [Bridge]
            section_match = re.match(r'\[([^\]]+)\]', line)
            if section_match:
                # Save previous section if it has content
                if current_section["text"].strip():
                    sections.append(current_section)
                
                # Start new section
                section_type = section_match.group(1).lower()
                current_section = {"type": section_type, "text": ""}
            else:
                # Add line to current section
                if line:  # Skip empty lines
                    current_section["text"] += line + "\n"
        
        # Add final section
        if current_section["text"].strip():
            sections.append(current_section)
        
        # Clean up text in each section
        for section in sections:
            section["text"] = section["text"].strip()
        
        return sections
    
    def _generate_with_suno(self, prompt: str) -> bool:
        """Generate music using Suno API."""
        api_key = os.getenv(self.config.api.suno.api_key_env)
        if not api_key:
            self.logger.warning(f"Suno API key not found in environment variable: {self.config.api.suno.api_key_env}")
            return False
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "customMode": False,
            "instrumental": False,
            "model": self.config.api.suno.model,
            "callBackUrl": "https://httpbin.org/post"  # Dummy callback URL for testing
        }
        
        try:
            self.logger.info(f"Calling Suno API with prompt: {prompt[:50]}...")
            
            response = requests.post(
                f"{self.config.api.suno.base_url}/api/v1/generate",
                headers=headers,
                json=payload,
                timeout=self.config.api.suno.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 200 and 'taskId' in data.get('data', {}):
                    task_id = data['data']['taskId']
                    self.logger.info(f"Got task ID: {task_id}, waiting for completion...")
                    
                    # Poll for completion
                    return self._wait_for_completion(task_id)
                else:
                    self.logger.error(f"Unexpected response format: {data}")
                    return False
            else:
                self.logger.error(f"Suno API request failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return False
    
    def _wait_for_completion(self, task_id: str) -> bool:
        """Wait for Suno task to complete and download the result."""
        import time
        
        api_key = os.getenv(self.config.api.suno.api_key_env)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        max_wait_time = 300  # 5 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Use the correct endpoint from documentation
                status_url = f"{self.config.api.suno.base_url}/api/v1/generate/record-info?taskId={task_id}"
                
                response = requests.get(status_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    self.logger.info(f"Status response: {data}")
                    
                    if data.get('code') == 200 and 'data' in data:
                        task_data = data['data']
                        status = task_data.get('status')
                        
                        self.logger.info(f"Task status: {status}")
                        
                        if status == 'SUCCESS':
                            # Task completed successfully
                            response_data = task_data.get('response', {})
                            suno_data = response_data.get('sunoData', [])
                            
                            if suno_data:
                                # Use the first song
                                song = suno_data[0]
                                self.logger.info(f"Song completed: {song.get('title', 'Untitled')}")
                                return self._download_audio_from_suno_format(song)
                            else:
                                self.logger.error("No song data in successful response")
                                return False
                                
                        elif status in ['PENDING', 'TEXT_SUCCESS', 'FIRST_SUCCESS']:
                            # Still processing
                            self.logger.info(f"Task still processing (status: {status}), waiting...")
                            time.sleep(10)
                            continue
                            
                        elif status in ['CREATE_TASK_FAILED', 'GENERATE_AUDIO_FAILED', 'CALLBACK_EXCEPTION', 'SENSITIVE_WORD_ERROR']:
                            # Failed
                            error_msg = task_data.get('errorMessage', 'Unknown error')
                            self.logger.error(f"Task failed with status {status}: {error_msg}")
                            return False
                        else:
                            self.logger.warning(f"Unknown status: {status}, continuing to wait...")
                            time.sleep(10)
                            continue
                    else:
                        self.logger.error(f"Unexpected status response: {data}")
                        return False
                else:
                    self.logger.error(f"Status check failed: {response.status_code} - {response.text}")
                    return False
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Status check request failed: {e}")
                time.sleep(10)
                continue
        
        self.logger.error("Task timeout - exceeded maximum wait time")
        return False
    
    def _download_audio_from_suno_format(self, song_data: dict) -> bool:
        """Download audio from Suno API response format."""
        try:
            # Suno API returns audioUrl (not audio_url)
            audio_url = song_data.get('audioUrl')
            if not audio_url:
                self.logger.error("No audioUrl in song data")
                return False
            
            self.logger.info(f"Downloading audio from: {audio_url}")
            
            # Download the audio file
            audio_response = requests.get(audio_url, timeout=60)
            if audio_response.status_code == 200:
                with open(self.config.io.output_file, 'wb') as f:
                    f.write(audio_response.content)
                
                # Save lyrics - Suno format includes lyrics object with text and title
                lyrics_obj = song_data.get('lyrics', {})
                lyrics_text = lyrics_obj.get('text', '')
                lyrics_title = lyrics_obj.get('title', '')
                if lyrics_text:
                    self._save_lyrics(lyrics_text, lyrics_title)
                
                self.logger.info("Successfully downloaded audio and lyrics from Suno API")
                return True
            else:
                self.logger.error(f"Failed to download audio: {audio_response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading audio: {e}")
            return False

    def _download_audio(self, song_data: dict) -> bool:
        """Download the completed audio file."""
        try:
            audio_url = song_data.get('audio_url')
            if not audio_url:
                self.logger.error("No audio URL in song data")
                return False
            
            self.logger.info(f"Downloading audio from: {audio_url}")
            
            # Download the audio file
            audio_response = requests.get(audio_url, timeout=60)
            if audio_response.status_code == 200:
                with open(self.config.io.output_file, 'wb') as f:
                    f.write(audio_response.content)
                
                # Save lyrics if available
                lyrics_obj = song_data.get('lyrics', {})
                lyrics_text = lyrics_obj.get('text', '')
                lyrics_title = lyrics_obj.get('title', '')
                if lyrics_text:
                    self._save_lyrics(lyrics_text, lyrics_title)
                
                self.logger.info("Successfully downloaded audio and lyrics from Suno API")
                return True
            else:
                self.logger.error(f"Failed to download audio: {audio_response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading audio: {e}")
            return False

    def run(self) -> int:
        """Run the music generator service."""
        try:
            self.logger.info("Starting music generator service")
            
            # Validate inputs
            if not self.validate_inputs():
                return 1
            
            # Load input
            prompt = self._load_input()
            self.logger.info(f"Loaded prompt: {len(prompt)} characters")
            
            # Generate music based on provider
            success = False
            
            if self.config.api.provider == "suno":
                self.logger.info("Using Suno API")
                success = self._generate_with_suno(prompt)
                
                if not success:
                    raise RuntimeError("Suno API failed to generate music. No offline fallback available.")
                    
            elif self.config.api.provider == "offline":
                raise RuntimeError("Offline mode has been disabled. Please configure Suno API access.")
                
            else:
                raise ValueError(f"Unknown API provider: {self.config.api.provider}")
            
            if not success:
                raise RuntimeError("Music generation failed")
            
            # Validate outputs
            if not self.validate_outputs():
                return 1
            
            self.logger.info("Music generator service completed successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Music generator service failed: {e}")
            return 1


def main():
    """Entry point for the music generator service."""
    run_service(MusicGeneratorService, "Music Generator")


if __name__ == "__main__":
    main() 