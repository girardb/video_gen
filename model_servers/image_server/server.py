"""
FastAPI server for image generation models (SDXL and others).
"""

import argparse
import logging
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import GenerateImageRequest, GenerateImageResponse, HealthResponse, ModelInfoResponse


class ImageServer:
    """FastAPI server for image generation models."""
    
    def __init__(self, model_name: str = "Qwen/Qwen-Image", port: int = 8005):
        """Initialize the image server."""
        self.model_name = model_name
        self.port = port
        self.pipeline = None
        self.device = self._get_device()
        self.output_dir = Path("out/image_server_temp")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Image Generation Server",
            description="FastAPI server for text-to-image generation",
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
            return HealthResponse(
                status="healthy" if self.pipeline is not None else "unhealthy",
                model_loaded=self.pipeline is not None,
                model_name=self.model_name if self.pipeline is not None else None,
                device=self.device
            )
        
        @self.app.get("/model/info", response_model=ModelInfoResponse)
        async def model_info():
            """Get model information."""
            return ModelInfoResponse(
                model_name=self.model_name,
                model_type="text-to-image",
                max_resolution=1664 if "qwen" in self.model_name.lower() else (1024 if "xl" in self.model_name.lower() else 512),
                supported_formats=["png", "jpg"],
                device=self.device
            )
        
        @self.app.post("/generate", response_model=GenerateImageResponse)
        async def generate_image(request: GenerateImageRequest):
            """Generate image using the loaded model."""
            try:
                if self.pipeline is None:
                    raise HTTPException(status_code=503, detail="Model not loaded - server initialization failed")
                
                start_time = time.time()
                
                # Generate image
                image_path = self._generate_image(request)
                
                generation_time = time.time() - start_time
                
                return GenerateImageResponse(
                    image_path=str(image_path),
                    generation_time=generation_time,
                    model_name=self.model_name,
                    parameters={
                        "prompt": request.prompt,
                        "negative_prompt": request.negative_prompt,
                        "width": request.width,
                        "height": request.height,
                        "num_inference_steps": request.num_inference_steps,
                        "guidance_scale": request.guidance_scale,
                        "seed": request.seed
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Image generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _load_model(self):
        """Load the image generation model."""
        try:
            self.logger.info(f"Loading image model: {self.model_name}")
            
            if "qwen" in self.model_name.lower():
                self._load_qwen_image_model()
            elif "xl" in self.model_name.lower():
                self._load_sdxl_model()
            else:
                self._load_sd_model()
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _load_sdxl_model(self):
        """Load SDXL model."""
        try:
            from diffusers import StableDiffusionXLPipeline
            
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                variant="fp16" if self.device != "cpu" else None,
                use_safetensors=True
            )
            
            # Enable optimizations
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
            
            self.logger.info(f"SDXL model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load SDXL model: {e}")
            raise RuntimeError(f"SDXL model loading failed: {e}")
    
    def _load_sd_model(self):
        """Load standard Stable Diffusion model."""
        try:
            from diffusers import StableDiffusionPipeline
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                variant="fp16" if self.device != "cpu" else None,
                use_safetensors=True
            )
            
            # Enable optimizations
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
            
            self.logger.info(f"SD model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load SD model: {e}")
            raise RuntimeError(f"SD model loading failed: {e}")
    
    def _load_qwen_image_model(self):
        """Load Qwen-Image model."""
        try:
            from diffusers import DiffusionPipeline
            
            # Qwen-Image uses bfloat16 for optimal performance
            torch_dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
            
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True  # Qwen models may need custom code
            )
            
            # Enable optimizations
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
            
            self.logger.info(f"Qwen-Image model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Qwen-Image model: {e}")
            raise RuntimeError(f"Qwen-Image model loading failed: {e}")
    
    def _generate_image(self, request: GenerateImageRequest) -> Path:
        """Generate image using the loaded pipeline."""
        
        # Create unique output filename
        image_id = str(uuid.uuid4())[:8]
        output_path = self.output_dir / f"image_{image_id}.png"
        
        self.logger.info(f"Generating image: {request.prompt[:50]}... -> {output_path}")
        
        # Setup generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(request.seed)
        
        # Generate image - no fallbacks, fail fast
        if not hasattr(self.pipeline, '__call__') or self.pipeline is None:
            raise RuntimeError("No valid pipeline loaded - model failed to initialize")
        
        try:
            # Prepare generation parameters
            generation_params = {
                "prompt": self._enhance_prompt_for_qwen(request.prompt),
                "height": request.height,
                "width": request.width,
                "num_inference_steps": request.num_inference_steps,
                "generator": generator,
            }
            
            # Add negative prompt if provided
            if request.negative_prompt:
                generation_params["negative_prompt"] = request.negative_prompt
            
            # Qwen-Image uses different parameter names
            if "qwen" in self.model_name.lower():
                generation_params["true_cfg_scale"] = request.guidance_scale
            else:
                generation_params["guidance_scale"] = request.guidance_scale
            
            # Generate image
            result = self.pipeline(**generation_params)
            
            # Save image
            image = result.images[0]
            image.save(output_path)
            
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise RuntimeError(f"Image generation failed: {e}")
        
        return output_path
    
    def _enhance_prompt_for_qwen(self, prompt: str) -> str:
        """Enhance prompt with Qwen-Image magic suffixes for better quality."""
        if "qwen" not in self.model_name.lower():
            return prompt
        
        # Detect language and add appropriate magic suffix
        # Simple heuristic: if contains Chinese characters, use Chinese suffix
        import re
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', prompt))
        
        if has_chinese:
            magic_suffix = ", 超清，4K，电影级构图."
        else:
            magic_suffix = ", Ultra HD, 4K, cinematic composition."
        
        # Avoid double-adding if already present
        if "4K" not in prompt and "cinematic" not in prompt:
            return prompt + magic_suffix
        
        return prompt
    

    
    def run(self):
        """Run the FastAPI server."""
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )


def main():
    """Entry point for the image server."""
    parser = argparse.ArgumentParser(description="Image Generation Server")
    parser.add_argument(
        "--model-name",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Image generation model name"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8005,
        help="Port to run the server on"
    )
    
    args = parser.parse_args()
    
    server = ImageServer(args.model_name, args.port)
    server.run()


if __name__ == "__main__":
    main()
