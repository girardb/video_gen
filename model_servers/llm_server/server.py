"""
FastAPI server for local LLM models (Llama/Mistral).
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import GenerateRequest, GenerateResponse, HealthResponse, ModelInfoResponse


class LLMServer:
    """FastAPI server for local LLM models."""
    
    def __init__(self, model_path: str, port: int = 8001):
        """Initialize the LLM server."""
        self.model_path = model_path
        self.port = port
        self.model = None
        self.model_name = "placeholder"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="LLM Server",
            description="FastAPI server for local LLM models",
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
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model_loaded=self.model is not None,
                model_name=self.model_name if self.model else None
            )
        
        @self.app.get("/model/info", response_model=ModelInfoResponse)
        async def model_info():
            """Get model information."""
            return ModelInfoResponse(
                model_name=self.model_name,
                model_path=self.model_path,
                max_context_length=4096,  # Placeholder
                parameters=7000000000,  # Placeholder for 7B model
                quantization=None
            )
        
        @self.app.post("/generate", response_model=GenerateResponse)
        async def generate_text(request: GenerateRequest):
            """Generate text using the loaded model."""
            try:
                if self.model is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")
                
                # Generate text using the loaded model
                generated_text = self._generate_text(
                    request.prompt,
                    request.system_prompt,
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                    request.stop_sequences
                )
                
                return GenerateResponse(
                    text=generated_text,
                    tokens_generated=len(generated_text.split()),
                    model_name=self.model_name
                )
                
            except Exception as e:
                self.logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _load_model(self):
        """Load the LLM model."""
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            
            # Try different loading strategies based on model type
            if "mistral" in self.model_path.lower() or "llama" in self.model_path.lower():
                self._load_llama_model()
            elif "gpt" in self.model_path.lower():
                self._load_gpt_model()
            else:
                self._load_generic_model()
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def _load_llama_model(self):
        """Load Llama/Mistral model with llama-cpp-python."""
        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=4,
                n_gpu_layers=-1  # Use GPU if available
            )
            self.model_name = os.path.basename(self.model_path)
            self.logger.info(f"Model loaded successfully with llama-cpp: {self.model_name}")
            
        except ImportError:
            self.logger.warning("llama-cpp-python not available, using placeholder")
            self.model = "placeholder"
            self.model_name = os.path.basename(self.model_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to load with llama-cpp: {e}, using placeholder")
            self.model = "placeholder"
            self.model_name = os.path.basename(self.model_path)
    
    def _load_gpt_model(self):
        """Load GPT model with transformers."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model_name = os.path.basename(self.model_path)
            self.logger.info(f"Model loaded successfully with transformers: {self.model_name}")
            
        except ImportError:
            self.logger.warning("transformers not available, using placeholder")
            self.model = "placeholder"
            self.model_name = os.path.basename(self.model_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to load with transformers: {e}, using placeholder")
            self.model = "placeholder"
            self.model_name = os.path.basename(self.model_path)
    
    def _load_generic_model(self):
        """Load generic model with transformers."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model_name = os.path.basename(self.model_path)
            self.logger.info(f"Model loaded successfully with transformers: {self.model_name}")
            
        except ImportError:
            self.logger.warning("transformers not available, using placeholder")
            self.model = "placeholder"
            self.model_name = os.path.basename(self.model_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to load with transformers: {e}, using placeholder")
            self.model = "placeholder"
            self.model_name = os.path.basename(self.model_path)
    
    def _generate_text(self, prompt: str, system_prompt: Optional[str], 
                      max_tokens: int, temperature: float, top_p: float, 
                      stop_sequences: Optional[List[str]]) -> str:
        """Generate text using the loaded model."""
        
        self.logger.info(f"Model type: {type(self.model)}")
        self.logger.info(f"Model has create_completion: {hasattr(self.model, 'create_completion')}")
        self.logger.info(f"Model has generate: {hasattr(self.model, 'generate')}")
        self.logger.info(f"Model value: {self.model}")
        
        # If we have a real model loaded
        if hasattr(self.model, 'create_completion'):
            # Use llama-cpp model
            self.logger.info("Using llama-cpp model")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.model.create_completion(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences or [],
                stream=False
            )
            
            return response['choices'][0]['message']['content']
        
        elif hasattr(self.model, 'generate') and hasattr(self, 'tokenizer'):
            # Use transformers model
            self.logger.info("Using transformers model")
            
            # Format input text
            if system_prompt:
                full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = prompt
                
            self.logger.info(f"Full prompt for model: {full_prompt}")
            
            # Tokenize and generate
            # Set padding token if not set to avoid attention mask warnings
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                padding=True, 
                add_special_tokens=True
            )
            
            import torch
            
            # Get model device and move all inputs there
            device = next(self.model.parameters()).device
            self.logger.info(f"Model device: {device}")
            
            # Move all input tensors to the model's device
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            self.logger.info(f"Input ids device: {input_ids.device}")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the response
            response = response[len(full_prompt):].strip()
            
            self.logger.info(f"Generated response: {response}")
            return response
        
        else:
            # No usable model found - raise error instead of fallback
            self.logger.error("No usable model found - model failed to load properly")
            raise RuntimeError(f"Model failed to load properly. Model type: {type(self.model)}, Model value: {self.model}")
    
    def run(self):
        """Run the FastAPI server."""
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )


def main():
    """Entry point for the LLM server."""
    parser = argparse.ArgumentParser(description="LLM Server")
    parser.add_argument(
        "--model-path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        help="Path to the model"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to run the server on"
    )
    
    args = parser.parse_args()
    
    server = LLMServer(args.model_path, args.port)
    server.run()


if __name__ == "__main__":
    main() 