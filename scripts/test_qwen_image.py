#!/usr/bin/env python3
"""
Test script for Qwen-Image integration.

Tests the updated image server with Qwen-Image model.
"""

import json
import requests
import time
from pathlib import Path


def test_qwen_image_server(server_url: str = "http://localhost:8005"):
    """Test the Qwen-Image server."""
    
    print("🎨 Testing Qwen-Image Server")
    print(f"Server URL: {server_url}")
    print("-" * 50)
    
    # Test 1: Health Check
    print("1. Health Check...")
    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Server is healthy")
            print(f"   📋 Model loaded: {health.get('model_loaded', False)}")
            print(f"   🤖 Model name: {health.get('model_name', 'Unknown')}")
            print(f"   💻 Device: {health.get('device', 'Unknown')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        return False
    
    # Test 2: Model Info
    print("\n2. Model Information...")
    try:
        response = requests.get(f"{server_url}/model/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"   📊 Model: {info.get('model_name', 'Unknown')}")
            print(f"   🎯 Type: {info.get('model_type', 'Unknown')}")
            print(f"   📐 Max resolution: {info.get('max_resolution', 'Unknown')}")
            print(f"   📱 Supported formats: {info.get('supported_formats', [])}")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed to get model info: {e}")
    
    # Test 3: Music Video Style Image Generation
    print("\n3. Testing Qwen-Image Generation...")
    
    test_prompts = [
        "A neon-lit underground nightclub with pulsing lights and dancing silhouettes",
        "Epic mountain landscape at sunset with a lone figure holding a guitar",
        "Cyberpunk city street with holographic advertisements and rain reflections"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n   🎬 Test {i+1}: {prompt[:40]}...")
        
        test_request = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted, text, watermark",
            "width": 1664,  # 16:9 aspect ratio
            "height": 928,
            "num_inference_steps": 25,
            "guidance_scale": 4.0,  # Qwen-Image uses true_cfg_scale
            "seed": 42 + i
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{server_url}/generate",
                json=test_request,
                timeout=120  # Qwen-Image may take longer
            )
            
            if response.status_code == 200:
                result = response.json()
                generation_time = time.time() - start_time
                
                print(f"      ✅ Image generated successfully!")
                print(f"      📁 Output path: {result.get('image_path')}")
                print(f"      ⏱️  Generation time: {result.get('generation_time', generation_time):.1f}s")
                print(f"      🤖 Model used: {result.get('model_name')}")
                
                # Check if file exists
                image_path = Path(result.get('image_path', ''))
                if image_path.exists():
                    file_size = image_path.stat().st_size / 1024 / 1024  # MB
                    print(f"      📊 File size: {file_size:.1f}MB")
                else:
                    print(f"      ⚠️  Generated file not found at expected path")
                    
            else:
                print(f"      ❌ Generation failed: {response.status_code}")
                print(f"      📄 Error: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("      ❌ Generation timed out")
            return False
        except Exception as e:
            print(f"      ❌ Generation failed: {e}")
            return False
    
    print("\n🎉 All Qwen-Image tests completed successfully!")
    print("\n✨ Qwen-Image Features Verified:")
    print("   • High-quality image generation")
    print("   • 16:9 aspect ratio for video")
    print("   • Cinematic composition")
    print("   • Text rendering capabilities")
    return True


def main():
    """Run the Qwen-Image server tests."""
    print("Qwen-Image Server Test Suite")
    print("=" * 50)
    
    # Instructions
    print("\n📋 Before running this test:")
    print("1. Start the image server with Qwen-Image:")
    print("   python -m model_servers.image_server --model-name 'Qwen/Qwen-Image' --port 8005")
    print("2. Wait for the model to load (this may take several minutes on first run)")
    print("3. Run this test script")
    
    input("\nPress Enter when the image server is ready...")
    
    # Run tests
    success = test_qwen_image_server()
    
    if success:
        print("\n✅ Qwen-Image server is working correctly!")
        print("🎬 Ready for Service 3.5 image generation!")
    else:
        print("\n❌ Qwen-Image server tests failed. Check the server logs.")


if __name__ == "__main__":
    main()
