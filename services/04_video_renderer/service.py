"""
Video renderer service implementation.

Generates video clips from storyboard prompts using various video generation engines.
Supports both text-to-video and text-to-image + image-to-video approaches.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

import requests

from ..base import BaseService, run_service
from .config import VideoRendererConfig


class VideoRendererService(BaseService):
    """Service for rendering video clips from storyboard prompts."""
    
    def __init__(self, config_path: str, **kwargs):
        super().__init__(config_path, **kwargs)
        
    def _create_config(self, config_data: dict) -> VideoRendererConfig:
        """Create configuration object from data."""
        return VideoRendererConfig(**config_data)
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        input_file = Path(self.config.io.input)
        if not input_file.exists():
            self.logger.error(f"Input storyboard file not found: {input_file}")
            return False
        
        # Check if video server is available
        try:
            health_response = requests.get(f"{self.config.video_server.url}/health", timeout=5)
            if health_response.status_code != 200:
                self.logger.error(f"Video server not available at {self.config.video_server.url}")
                return False
            
            health_data = health_response.json()
            if not health_data.get("model_loaded", False):
                self.logger.error("Video server model not loaded")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to video server: {e}")
            return False
        
        return True
    
    def validate_outputs(self) -> bool:
        """Validate that outputs were created successfully."""
        output_dir = Path(self.config.io.output_dir)
        if not output_dir.exists():
            self.logger.error(f"Output directory not created: {output_dir}")
            return False
        
        # Check if any video files were created
        video_files = list(output_dir.glob("*.mp4"))
        if not video_files:
            self.logger.error("No video files were generated")
            return False
        
        self.logger.info(f"Generated {len(video_files)} video clips")
        return True
    
    def _check_video_server(self) -> bool:
        """Check if video server is healthy and ready."""
        try:
            response = requests.get(f"{self.config.video_server.url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("model_loaded", False):
                    self.logger.info(f"Video server ready: {health_data.get('model_name')}")
                    return True
                else:
                    self.logger.error("Video server model not loaded")
                    return False
            else:
                self.logger.error(f"Video server health check failed: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to connect to video server: {e}")
            return False
    
    def _load_storyboard(self) -> List[Dict]:
        """Load storyboard from JSON file."""
        with open(self.config.io.input, 'r', encoding='utf-8') as f:
            storyboard = json.load(f)
        
        # Validate storyboard format
        if not isinstance(storyboard, list):
            raise ValueError("Storyboard must be a list of shots")
        
        return storyboard
    
    def _calculate_num_frames(self, duration: float) -> int:
        """Calculate number of frames based on duration and FPS."""
        num_frames = int(duration * self.config.video.fps)
        # Ensure minimum frames for good quality, max 64 for Wan2.2
        return max(8, min(num_frames, 64))
    
    def _generate_video_via_server(self, shot: Dict) -> str:
        """Generate video by calling the video server API."""
        prompt = shot["prompt"]
        duration = shot["end"] - shot["start"]
        num_frames = self._calculate_num_frames(duration)
        
        # Override num_frames if specified in config
        if self.config.generation.num_frames:
            num_frames = self.config.generation.num_frames
        
        self.logger.info(f"Generating {num_frames} frames via server for: {prompt[:50]}...")
        
        # Prepare request payload for Wan2.2-S2V
        payload = {
            "prompt": prompt,
            "ref_image_path": shot["ref_image"],  # From service 03.5
            "audio_path": self.config.io.audio_file,  # Full song file
            "audio_start": shot["start"],  # Extract clip from this time
            "audio_duration": duration,  # Duration of the clip
            "num_frames": num_frames,
            "width": self.config.video.resolution[0],
            "height": self.config.video.resolution[1],
            "num_inference_steps": self.config.video_server.num_inference_steps,
            "guidance_scale": self.config.video_server.guidance_scale,
            "fps": self.config.video.fps
        }
        
        # Add seed if using consistency
        if self.config.consistency.use_seed and "seed" in shot:
            payload["seed"] = shot["seed"] + self.config.consistency.seed_offset
        
        # Add negative prompt if configured
        if hasattr(self.config.video_server, 'negative_prompt') and self.config.video_server.negative_prompt:
            payload["negative_prompt"] = self.config.video_server.negative_prompt
        
        # Make request to video server
        try:
            response = requests.post(
                f"{self.config.video_server.url}/generate",
                json=payload,
                timeout=self.config.video_server.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                video_path = result["video_path"]
                self.logger.info(f"Server generated video in {result['generation_time']:.1f}s: {video_path}")
                return video_path
            else:
                self.logger.error(f"Video server request failed: {response.status_code} - {response.text}")
                raise RuntimeError(f"Video server failed: {response.status_code}")
                
        except requests.exceptions.Timeout:
            self.logger.error("Video server request timed out")
            raise RuntimeError("Video server timeout")
        except Exception as e:
            self.logger.error(f"Failed to call video server: {e}")
            raise RuntimeError(f"Video server communication failed: {e}")
    
    def _generate_shot(self, shot: Dict, shot_index: int) -> None:
        """Generate a single video shot."""
        start_time = time.time()
        
        # Create output filename
        shot_id = f"shot_{shot_index:03d}_{shot['start']:.1f}s_{shot['end']:.1f}s"
        output_path = Path(self.config.io.output_dir) / f"{shot_id}.mp4"
        
        # Skip if already exists (for resuming interrupted runs)
        if output_path.exists():
            self.logger.info(f"Shot {shot_index} already exists, skipping")
            return
        
        try:
            # Generate video via server
            server_video_path = self._generate_video_via_server(shot)
            
            # Copy the generated video to our output location
            import shutil
            shutil.copy2(server_video_path, output_path)
            
            generation_time = time.time() - start_time
            self.logger.info(f"Shot {shot_index} completed in {generation_time:.1f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to generate shot {shot_index}: {e}")
            raise
    
    def run(self) -> int:
        """Run the video renderer service."""
        try:
            self.logger.info("Starting video renderer service")
            
            # Validate inputs
            if not self.validate_inputs():
                return 1
            
            # Create output directory
            output_dir = Path(self.config.io.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load storyboard
            storyboard = self._load_storyboard()
            self.logger.info(f"Loaded storyboard with {len(storyboard)} shots")
            
            # Check video server availability
            if not self._check_video_server():
                return 1
            
            # Generate videos for each shot
            for i, shot in enumerate(storyboard):
                self.logger.info(f"Processing shot {i+1}/{len(storyboard)}")
                self._generate_shot(shot, i)
            
            # Validate outputs
            if not self.validate_outputs():
                return 1
            
            self.logger.info("Video renderer service completed successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Video renderer service failed: {e}")
            return 1


def main():
    """Entry point for the video renderer service."""
    run_service(VideoRendererService, "Video Renderer")


if __name__ == "__main__":
    main()
