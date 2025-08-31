#!/usr/bin/env python3
"""
Test script for the video generation server.

This script demonstrates how to start and test the video server with Wan2.2-S2V-14B.
"""

import json
import requests
import time
from pathlib import Path


def test_video_server(server_url: str = "http://localhost:8004"):
    """Test the video generation server."""
    
    print("🎬 Testing Video Generation Server")
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
            if health.get('memory_usage'):
                mem = health['memory_usage']
                print(f"   🧠 GPU Memory: {mem.get('allocated', 0):.1f}GB allocated")
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
            print(f"   🎞️  Max frames: {info.get('max_frames', 'Unknown')}")
            print(f"   📐 Max resolution: {info.get('max_resolution', 'Unknown')}")
            print(f"   📱 Supported formats: {info.get('supported_formats', [])}")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed to get model info: {e}")
    
    # Test 3: Simple Video Generation
    print("\n3. Testing Video Generation...")
    print("   ⏳ Generating a simple test video...")
    
    test_request = {
        "prompt": "A beautiful sunset over mountains with golden light",
        "num_frames": 16,
        "width": 512,
        "height": 512,
        "num_inference_steps": 10,  # Reduced for faster testing
        "guidance_scale": 7.5,
        "fps": 8,
        "seed": 42
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{server_url}/generate",
            json=test_request,
            timeout=300  # 5 minutes max
        )
        
        if response.status_code == 200:
            result = response.json()
            generation_time = time.time() - start_time
            
            print(f"   ✅ Video generated successfully!")
            print(f"   📁 Output path: {result.get('video_path')}")
            print(f"   🎞️  Frames: {result.get('frames_generated')}")
            print(f"   ⏱️  Generation time: {result.get('generation_time', generation_time):.1f}s")
            print(f"   🤖 Model used: {result.get('model_name')}")
            
            # Check if file exists
            video_path = Path(result.get('video_path', ''))
            if video_path.exists():
                file_size = video_path.stat().st_size / 1024 / 1024  # MB
                print(f"   📊 File size: {file_size:.1f}MB")
            else:
                print(f"   ⚠️  Generated file not found at expected path")
                
        else:
            print(f"   ❌ Generation failed: {response.status_code}")
            print(f"   📄 Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ❌ Generation timed out")
        return False
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        return False
    
    print("\n🎉 All tests completed successfully!")
    return True


def main():
    """Run the video server tests."""
    print("Video Server Test Suite")
    print("=" * 50)
    
    # Instructions
    print("\n📋 Before running this test:")
    print("1. Start the video server:")
    print("   python -m model_servers.video_server --model-name 'Wan-AI/Wan2.2-S2V-14B' --port 8004")
    print("2. Wait for the model to load (this may take several minutes on first run)")
    print("3. Run this test script")
    
    input("\nPress Enter when the video server is ready...")
    
    # Run tests
    success = test_video_server()
    
    if success:
        print("\n✅ Video server is working correctly!")
    else:
        print("\n❌ Video server tests failed. Check the server logs.")


if __name__ == "__main__":
    main()
