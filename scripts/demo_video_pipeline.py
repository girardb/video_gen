#!/usr/bin/env python3
"""
Demo script showing the complete video generation pipeline.

This demonstrates the full workflow from storyboard to generated videos.
"""

import json
import subprocess
import time
from pathlib import Path


# Demo storyboard
demo_storyboard = [
    {
        "start": 0.0,
        "end": 4.0,
        "prompt": "wide establishing shot of cavemen around a glowing campfire at twilight with warm orange lighting",
        "motion": "slow dolly",
        "seed": 12345,
        "model": "wan2.2",
        "ref_image": None
    },
    {
        "start": 4.0,
        "end": 8.0,
        "prompt": "medium shot of a caveman drinking from a primitive cup with satisfied expression and firelight dancing",
        "motion": "zoom in",
        "seed": 12346,
        "model": "wan2.2",
        "ref_image": None
    },
    {
        "start": 8.0,
        "end": 12.0,
        "prompt": "close up of the caveman's eyes widening in amazement as magical sparkles appear around the drink",
        "motion": "push in",
        "seed": 12347,
        "model": "wan2.2",
        "ref_image": None
    }
]


def setup_demo_data():
    """Set up demo data files."""
    print("📁 Setting up demo data...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save demo storyboard
    storyboard_file = data_dir / "storyboard.json"
    with open(storyboard_file, 'w') as f:
        json.dump(demo_storyboard, f, indent=2)
    
    print(f"   ✅ Created demo storyboard: {storyboard_file}")
    print(f"   📊 Contains {len(demo_storyboard)} shots")
    return storyboard_file


def check_server_status(server_url: str, server_name: str) -> bool:
    """Check if a server is running and healthy."""
    import requests
    
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            if health.get("model_loaded", False):
                print(f"   ✅ {server_name} is ready ({health.get('model_name', 'Unknown')})")
                return True
            else:
                print(f"   ⚠️  {server_name} is running but model not loaded")
                return False
        else:
            print(f"   ❌ {server_name} health check failed")
            return False
    except Exception:
        print(f"   ❌ {server_name} is not available")
        return False


def main():
    """Run the video generation pipeline demo."""
    print("🎬 Video Generation Pipeline Demo")
    print("=" * 50)
    
    # Setup
    print("\n1. Setting up demo data...")
    storyboard_file = setup_demo_data()
    
    # Check servers
    print("\n2. Checking server status...")
    video_server_ready = check_server_status("http://localhost:8004", "Video Server")
    
    if not video_server_ready:
        print("\n❌ Video server is not ready!")
        print("\n📋 To start the video server:")
        print("   python -m model_servers.video_server --model-name 'Wan-AI/Wan2.2-S2V-14B' --port 8004")
        print("\n⚠️  Note: First startup will download ~32GB model files")
        return
    
    # Run video renderer
    print("\n3. Running video renderer service...")
    print("   🎯 Generating videos from storyboard...")
    
    try:
        start_time = time.time()
        
        # Run the video renderer service
        result = subprocess.run([
            "python", "-m", "services.04_video_renderer",
            "--config", "configs/04.yaml"
        ], capture_output=True, text=True, timeout=1800)  # 30 minutes max
        
        total_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✅ Video generation completed in {total_time:.1f}s")
            
            # Check output directory
            output_dir = Path("out/clips")
            if output_dir.exists():
                video_files = list(output_dir.glob("*.mp4"))
                print(f"   📁 Generated {len(video_files)} video files:")
                
                total_size = 0
                for video_file in video_files:
                    size_mb = video_file.stat().st_size / 1024 / 1024
                    total_size += size_mb
                    print(f"      • {video_file.name} ({size_mb:.1f}MB)")
                
                print(f"   📊 Total size: {total_size:.1f}MB")
                print(f"   📁 Output directory: {output_dir.absolute()}")
            else:
                print("   ⚠️  Output directory not found")
        else:
            print(f"   ❌ Video generation failed!")
            print(f"   📄 Error output: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("   ❌ Video generation timed out")
    except Exception as e:
        print(f"   ❌ Video generation failed: {e}")
    
    print("\n🎬 Demo completed!")
    print("\n📋 Next steps:")
    print("   • Check generated videos in out/clips/")
    print("   • Run service 05_video_assembler to combine clips with music")
    print("   • Adjust configs/04.yaml for different generation settings")


if __name__ == "__main__":
    main()
