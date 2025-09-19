"""
FastAPI server for video generation models (Wan2.2-S2V-14B and others).
"""

import argparse
import logging
import os
import time
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import GenerateVideoRequest, GenerateVideoResponse, HealthResponse, ModelInfoResponse


class VideoServer:
    """FastAPI server for video generation models."""
    
    def __init__(self, model_name: str = "Wan-AI/Wan2.2-TI2V-5B", port: int = 8004):
        """Initialize the video server."""
        self.model_name = model_name
        self.port = port
        self.pipeline = None
        self.device = self._get_device()
        self.output_dir = Path("out/video_server_temp")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Video Generation Server",
            description="FastAPI server for text-to-video generation",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Load model
        self._load_model()
    
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            memory_usage = None
            if torch.cuda.is_available() and self.pipeline is not None:
                memory_usage = {
                    "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
                    "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
                }
            
            return HealthResponse(
                status="healthy",
                model_loaded=self.pipeline is not None,
                model_name=self.model_name if self.pipeline else None,
                device=self.device,
                memory_usage=memory_usage
            )
        
        @self.app.get("/model/info", response_model=ModelInfoResponse)
        async def model_info():
            """Get model information."""
            return ModelInfoResponse(
                model_name=self.model_name,
                model_type="text-to-video",
                max_frames=64,  # Wan2.2 supports up to 64 frames
                max_resolution=[1024, 1024],  # Maximum resolution
                supported_formats=["mp4", "gif"],
                parameters=5000000000 if "5B" in self.model_name else (14000000000 if "14B" in self.model_name else None),
                device=self.device
            )
        
        @self.app.post("/generate", response_model=GenerateVideoResponse)
        async def generate_video(request: GenerateVideoRequest):
            """Generate video using the loaded model."""
            try:
                if self.pipeline is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")
                
                start_time = time.time()
                
                # Generate video
                video_path = self._generate_video(request)
                
                generation_time = time.time() - start_time
                
                return GenerateVideoResponse(
                    video_path=str(video_path),
                    frames_generated=request.num_frames,
                    generation_time=generation_time,
                    model_name=self.model_name,
                    parameters={
                        "prompt": request.prompt,
                        "num_frames": request.num_frames,
                        "width": request.width,
                        "height": request.height,
                        "num_inference_steps": request.num_inference_steps,
                        "guidance_scale": request.guidance_scale,
                        "fps": request.fps,
                        "seed": request.seed
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Video generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _load_model(self):
        """Load the video generation model."""
        try:
            self.logger.info(f"Loading video model: {self.model_name}")
            
            if "cogvideo" in self.model_name.lower():
                self._load_cogvideo_model()
            elif "wan" in self.model_name.lower():
                self._load_wan_model()
            elif "animatediff" in self.model_name.lower():
                self._load_animatediff_model()
            elif "stable-video" in self.model_name.lower():
                self._load_svd_model()
            else:
                self._load_generic_video_model()
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # Don't set pipeline to None, let the exception propagate
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _load_wan_model(self):
        """Load Wan2.2 model using custom loading approach."""
        try:
            self.logger.info(f"Loading Wan2.2 model: {self.model_name}")
            
            # Check if we have the Wan2.2 repository
            import os
            wan_repo_path = "./Wan2.2"
            if not os.path.exists(wan_repo_path):
                self.logger.error(f"Wan2.2 repository not found at {wan_repo_path}")
                self.logger.error("Please clone the repository: git clone https://github.com/Wan-Video/Wan2.2.git")
                raise RuntimeError("Wan2.2 repository not found")
            
            # Download model if not exists
            model_dir = f"./Wan2.2-TI2V-5B"
            if not os.path.exists(model_dir):
                self.logger.info("Downloading Wan2.2-TI2V-5B model...")
                import subprocess
                result = subprocess.run([
                    "huggingface-cli", "download", "Wan-AI/Wan2.2-TI2V-5B", 
                    "--local-dir", model_dir
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.logger.error(f"Failed to download model: {result.stderr}")
                    raise RuntimeError(f"Model download failed: {result.stderr}")
                
                self.logger.info("Model downloaded successfully")
            
            # Import Wan2.2 modules
            import sys
            sys.path.append(wan_repo_path)
            
            try:
                # Import Wan2.2 modules
                from wan.textimage2video import WanTI2V
                from wan.configs import WAN_CONFIGS
                
                # Get the TI2V-5B config
                cfg = WAN_CONFIGS['ti2v-5B']
                
                # Initialize Wan2.2 pipeline with proper config
                self.pipeline = WanTI2V(
                    config=cfg,
                    checkpoint_dir=model_dir,
                    device_id=0,
                    rank=0,
                    t5_cpu=True,  # Use CPU for T5 to save GPU memory
                    convert_model_dtype=True,  # Convert to appropriate dtype
                    init_on_cpu=True  # Initialize on CPU first
                )
                
                self.logger.info(f"Wan2.2 TI2V-5B model loaded successfully on {self.device}")
                
            except ImportError as e:
                self.logger.error(f"Failed to import Wan2.2 modules: {e}")
                self.logger.error("Make sure Wan2.2 repository is properly set up")
                raise RuntimeError(f"Wan2.2 import failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Wan2.2 model: {e}")
            raise RuntimeError(f"Wan2.2 model loading failed: {e}")
    
    def _load_animatediff_model(self):
        """Load AnimateDiff model."""
        try:
            from diffusers import AnimateDiffPipeline, DPMSolverMultistepScheduler
            
            self.pipeline = AnimateDiffPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                variant="fp16" if self.device != "cpu" else None,
            )
            
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
            
            self.logger.info(f"AnimateDiff model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load AnimateDiff model: {e}")
            raise RuntimeError(f"AnimateDiff model loading failed: {e}")
    
    def _load_svd_model(self):
        """Load Stable Video Diffusion model."""
        try:
            from diffusers import StableVideoDiffusionPipeline
            
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                variant="fp16" if self.device != "cpu" else None,
            )
            
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
            
            self.logger.info(f"SVD model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load SVD model: {e}")
            raise RuntimeError(f"SVD model loading failed: {e}")
    
    def _load_generic_video_model(self):
        """Load generic video model with DiffusionPipeline."""
        try:
            from diffusers import DiffusionPipeline
            
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                variant="fp16" if self.device != "cpu" else None,
                trust_remote_code=True
            )
            
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
            
            self.logger.info(f"Generic video model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load generic video model: {e}")
            raise RuntimeError(f"Generic video model loading failed: {e}")
    
    def _load_cogvideo_model(self):
        """Load CogVideoX model."""
        try:
            from diffusers import CogVideoXPipeline
            
            
            try:
                self.logger.info(f"Trying CogVideoX loading strategy:")
                self.pipeline = CogVideoXPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16
                )
                self.logger.info(f"Successfully loaded CogVideoX with strategy")
            except Exception as e:
                self.logger.warning(f"Strategy failed: {str(e)}")
            
            
            # Enable memory optimizations for CogVideoX
            if hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
                self.pipeline.enable_sequential_cpu_offload()
            if hasattr(self.pipeline.vae, 'enable_tiling'):
                self.pipeline.vae.enable_tiling()
            if hasattr(self.pipeline.vae, 'enable_slicing'):
                self.pipeline.vae.enable_slicing()
            
            self.logger.info(f"CogVideoX model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load CogVideoX model: {e}")
            raise RuntimeError(f"CogVideoX model loading failed: {e}")
    
    def _generate_video(self, request: GenerateVideoRequest) -> Path:
        """Generate video using the loaded pipeline."""
        
        # Create unique output filename
        video_id = str(uuid.uuid4())[:8]
        output_path = self.output_dir / f"video_{video_id}.mp4"
        
        self.logger.info(f"Generating video: {request.prompt[:50]}... -> {output_path}")
        
        # Setup generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(request.seed)
        
        # Check if model is properly loaded
        if not hasattr(self.pipeline, '__call__') or self.pipeline is None:
            error_msg = f"Model not properly loaded. Current pipeline: {type(self.pipeline)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Prepare generation parameters based on model type
        if "cogvideo" in self.model_name.lower():
            # CogVideoX-specific parameters
            generation_params = {
                "prompt": request.prompt,
                "num_videos_per_prompt": 1,
                "num_frames": request.num_frames,
                "height": request.height,
                "width": request.width,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "generator": generator,
            }
            
            # Add image if provided (for I2V mode with CogVideoX-I2V)
            if request.ref_image_path:
                from PIL import Image
                ref_image = Image.open(request.ref_image_path)
                ref_image = ref_image.resize((request.width, request.height))
                generation_params["image"] = ref_image
        elif "wan" in self.model_name.lower():
            # Use Wan2.2's custom generation method
            generation_params = {
                "prompt": request.prompt,
                "size": (request.width, request.height),
                "num_frames": request.num_frames,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": generator.initial_seed() if generator else 42,
            }
            
            # Add image if provided (for I2V mode)
            if request.ref_image_path:
                from PIL import Image
                image = Image.open(request.ref_image_path)
                generation_params["image"] = image
        else:
            # Standard diffusers parameters
            generation_params = {
                "prompt": request.prompt,
                "num_frames": request.num_frames,
                "height": request.height,
                "width": request.width,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "generator": generator,
            }
            
            # Add image if provided (for I2V mode)
            if request.ref_image_path:
                from PIL import Image
                ref_image = Image.open(request.ref_image_path)
                ref_image = ref_image.resize((request.width, request.height))
                generation_params["image"] = ref_image
        
        # Add negative prompt if supported
        if request.negative_prompt and hasattr(self.pipeline, 'negative_prompt'):
            generation_params["negative_prompt"] = request.negative_prompt
        
        # Add Wan2.2-TI2V specific parameters
        if "wan" in self.model_name.lower():
            if request.motion_bucket_id is not None:
                generation_params["motion_bucket_id"] = request.motion_bucket_id
            if request.noise_aug_strength is not None:
                generation_params["noise_aug_strength"] = request.noise_aug_strength
        
        # Generate video using appropriate method
        try:
            if "wan" in self.model_name.lower():
                # Use Wan2.2's custom generation method
                result = self.pipeline.generate(
                    prompt=generation_params["prompt"],
                    img=generation_params.get("image"),  # Optional image for I2V
                    size=generation_params["size"],
                    frame_num=generation_params["num_frames"],
                    shift=5.0,  # Default shift from config
                    sample_solver="euler",  # Default solver
                    sampling_steps=generation_params["num_inference_steps"],
                    guide_scale=generation_params["guidance_scale"],
                    seed=generation_params["seed"],
                )
            else:
                # Use standard diffusers
                result = self.pipeline(**generation_params)
            
            # Extract frames - different models return results differently
            if hasattr(result, 'frames'):
                frames = result.frames[0]  # Most text-to-video models
            elif hasattr(result, 'videos'):
                frames = result.videos[0]  # Some alternative formats
            else:
                frames = result  # Direct frame output
            
            # Save video using diffusers export utility
            from diffusers.utils import export_to_video
            export_to_video(frames, str(output_path), fps=request.fps)
            
            self.logger.info(f"Video generation completed successfully: {output_path}")
            
        except Exception as e:
            # Log detailed error information
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"Video generation failed: {str(e)}")
            self.logger.error(f"Full error traceback: {error_details}")
            
            # Clean up any partial output file
            if output_path.exists():
                try:
                    output_path.unlink()
                    self.logger.info(f"Cleaned up partial output file: {output_path}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up partial file: {cleanup_error}")
            
            # Re-raise the exception with more context
            raise RuntimeError(f"Video generation failed for model '{self.model_name}': {str(e)}")
        
        return output_path
    
    
    def run(self):
        """Run the FastAPI server."""
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )


def main():
    """Entry point for the video server."""
    parser = argparse.ArgumentParser(description="Video Generation Server")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Wan-AI/Wan2.2-S2V-14B",
        help="Video generation model name"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8004,
        help="Port to run the server on"
    )
    
    args = parser.parse_args()
    
    server = VideoServer(args.model_name, args.port)
    server.run()


if __name__ == "__main__":
    main()
