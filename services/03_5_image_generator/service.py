"""
Text-to-image generator service implementation.

Generates reference images from storyboard prompts for use with Wan2.2-S2V video generation.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import requests

from ..base import BaseService, run_service
from .config import ImageGeneratorConfig


class ImageGeneratorService(BaseService):
    """Service for generating reference images from storyboard prompts."""
    
    def _create_config(self, config_data: dict) -> ImageGeneratorConfig:
        """Create configuration object from data."""
        return ImageGeneratorConfig(**config_data)
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        input_file = Path(self.config.io.input)
        if not input_file.exists():
            self.logger.error(f"Input storyboard file not found: {input_file}")
            return False
        
        # Check if image server is available
        try:
            response = requests.get(f"{self.config.image_server.url}/health", timeout=5)
            if response.status_code != 200:
                self.logger.error(f"Image server not available at {self.config.image_server.url}")
                return False
            
            health_data = response.json()
            if not health_data.get("model_loaded", False):
                self.logger.error("Image server model not loaded")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to image server: {e}")
            return False
        
        return True
    
    def validate_outputs(self) -> bool:
        """Validate that outputs were created successfully."""
        output_file = Path(self.config.io.output)
        if not output_file.exists():
            self.logger.error(f"Output file not created: {output_file}")
            return False
        
        image_dir = Path(self.config.io.image_dir)
        if not image_dir.exists():
            self.logger.error(f"Image directory not created: {image_dir}")
            return False
        
        # Check if any images were generated
        image_files = list(image_dir.glob("*.png"))
        if not image_files:
            self.logger.error("No images were generated")
            return False
        
        self.logger.info(f"Generated {len(image_files)} reference images")
        return True
    
    def _load_storyboard(self) -> List[Dict]:
        """Load storyboard from JSON file."""
        with open(self.config.io.input, 'r', encoding='utf-8') as f:
            storyboard = json.load(f)
        
        if not isinstance(storyboard, list):
            raise ValueError("Storyboard must be a list of shots")
        
        return storyboard
    
    def _enhance_prompt(self, original_prompt: str) -> str:
        """Enhance prompt for better image generation."""
        if not self.config.processing.enhance_prompts:
            return original_prompt
        
        # Add style suffix
        enhanced = f"{original_prompt}, {self.config.consistency.style_prompt_suffix}"
        
        # Ensure it's optimized for static images (not video descriptions)
        video_terms = ["motion", "movement", "walking", "running", "dancing", "moving"]
        for term in video_terms:
            if term in enhanced.lower():
                enhanced = enhanced.replace(term, "posed")
        
        return enhanced
    
    def _generate_image_via_server(self, prompt: str, seed: int, shot_index: int) -> str:
        """Generate image by calling the image server API."""
        enhanced_prompt = self._enhance_prompt(prompt)
        
        self.logger.info(f"Generating image via server: {enhanced_prompt[:50]}...")
        
        # Prepare request payload for Qwen-Image
        payload = {
            "prompt": enhanced_prompt,
            "negative_prompt": self.config.image_server.negative_prompt,
            "width": self.config.model.resolution[0],
            "height": self.config.model.resolution[1],
            "num_inference_steps": self.config.image_server.num_inference_steps,
            "guidance_scale": self.config.image_server.guidance_scale,
            "seed": seed
        }
        
        # Make request to image server
        try:
            response = requests.post(
                f"{self.config.image_server.url}/generate",
                json=payload,
                timeout=self.config.image_server.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                image_path = result["image_path"]
                self.logger.info(f"Server generated image in {result.get('generation_time', 0):.1f}s: {image_path}")
                return image_path
            else:
                self.logger.error(f"Image server request failed: {response.status_code} - {response.text}")
                raise RuntimeError(f"Image server failed: {response.status_code}")
                
        except requests.exceptions.Timeout:
            self.logger.error("Image server request timed out")
            raise RuntimeError("Image server timeout")
        except Exception as e:
            self.logger.error(f"Failed to call image server: {e}")
            raise RuntimeError(f"Image server communication failed: {e}")
    
    def _generate_reference_image(self, shot: Dict, shot_index: int) -> str:
        """Generate a reference image for a storyboard shot."""
        start_time = time.time()
        
        # Create output filename
        shot_id = f"ref_{shot_index:03d}_{shot['start']:.1f}s_{shot['end']:.1f}s"
        image_filename = f"{shot_id}.png"
        output_path = Path(self.config.io.image_dir) / image_filename
        
        # Skip if already exists
        if self.config.processing.skip_existing and output_path.exists():
            self.logger.info(f"Reference image {shot_index} already exists, skipping")
            return str(output_path)
        
        try:
            # Calculate seed for consistency
            seed = shot.get("seed", 12345)
            if self.config.consistency.use_seed:
                seed += self.config.consistency.seed_offset
            
            # Generate image via server
            server_image_path = self._generate_image_via_server(
                shot["prompt"], 
                seed, 
                shot_index
            )
            
            # Copy the generated image to our output location
            import shutil
            shutil.copy2(server_image_path, output_path)
            
            generation_time = time.time() - start_time
            self.logger.info(f"Reference image {shot_index} completed in {generation_time:.1f}s")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate reference image {shot_index}: {e}")
            raise
    
    def _save_enhanced_storyboard(self, enhanced_storyboard: List[Dict]) -> None:
        """Save enhanced storyboard with reference image paths."""
        output_path = Path(self.config.io.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_storyboard, f, indent=2)
        
        self.logger.info(f"Saved enhanced storyboard to {output_path}")
    
    def run(self) -> int:
        """Run the text-to-image generator service."""
        try:
            self.logger.info("Starting text-to-image generator service")
            
            # Validate inputs
            if not self.validate_inputs():
                return 1
            
            # Create output directories
            image_dir = Path(self.config.io.image_dir)
            image_dir.mkdir(parents=True, exist_ok=True)
            
            # Load storyboard
            storyboard = self._load_storyboard()
            self.logger.info(f"Loaded storyboard with {len(storyboard)} shots")
            
            # Generate reference images for each shot
            enhanced_storyboard = []
            
            for i, shot in enumerate(storyboard):
                self.logger.info(f"Processing shot {i+1}/{len(storyboard)}")
                
                # Generate reference image
                ref_image_path = self._generate_reference_image(shot, i)
                
                # Add ref_image path to shot
                enhanced_shot = shot.copy()
                enhanced_shot["ref_image"] = ref_image_path
                
                enhanced_storyboard.append(enhanced_shot)
            
            # Save enhanced storyboard
            self._save_enhanced_storyboard(enhanced_storyboard)
            
            # Validate outputs
            if not self.validate_outputs():
                return 1
            
            self.logger.info("Text-to-image generator service completed successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Text-to-image generator service failed: {e}")
            return 1


def main():
    """Entry point for the text-to-image generator service."""
    run_service(ImageGeneratorService, "Text-to-Image Generator")


if __name__ == "__main__":
    main()
