#!/usr/bin/env python3
"""
Test script for the video renderer service.

This script demonstrates how to use the video renderer with a simple test storyboard.
"""

import json
import tempfile
import yaml
from pathlib import Path

# Simple test storyboard
test_storyboard = [
    {
        "start": 0.0,
        "end": 4.0,
        "prompt": "wide shot of a beautiful sunset over mountains with golden lighting",
        "motion": "slow pan",
        "seed": 12345,
        "model": "sdxl",
        "ref_image": None
    },
    {
        "start": 4.0,
        "end": 8.0,
        "prompt": "close up of waves crashing on a rocky shore with dramatic lighting",
        "motion": "zoom in",
        "seed": 12346,
        "model": "sdxl", 
        "ref_image": None
    }
]

# Test configuration
test_config = {
    "io": {
        "input": "test_data/storyboard.json",
        "output_dir": "test_output/clips",
        "ref_image": None
    },
    "engine": {
        "type": "AnimateDiff",
        "animatediff": {
            "model": "guoyww/animatediff-sdxl",
            "pipeline_class": "AnimateDiffPipeline",
            "num_inference_steps": 10,  # Reduced for faster testing
            "guidance_scale": 7.5
        }
    },
    "video": {
        "fps": 8,  # Lower for faster generation
        "resolution": [512, 512],
        "format": "mp4",
        "codec": "h264",
        "bitrate": "2M"
    },
    "generation": {
        "batch_size": 1,
        "num_frames": 16,  # Fixed for testing
        "motion_bucket_id": 127,
        "fps_id": 6
    },
    "consistency": {
        "use_seed": True,
        "use_ref_image": False,
        "seed_offset": 0
    },
    "lora": {
        "enabled": False,
        "path": "lora/",
        "weight": 0.8
    },
    "performance": {
        "device": "auto",
        "memory_efficient": True,
        "compile": False,
        "enable_xformers": True
    }
}

def main():
    """Run the video renderer test."""
    print("Setting up test data...")
    
    # Create test directories
    test_data_dir = Path("test_data")
    test_output_dir = Path("test_output")
    test_data_dir.mkdir(exist_ok=True)
    test_output_dir.mkdir(exist_ok=True)
    
    # Save test storyboard
    storyboard_file = test_data_dir / "storyboard.json"
    with open(storyboard_file, 'w') as f:
        json.dump(test_storyboard, f, indent=2)
    print(f"Created test storyboard: {storyboard_file}")
    
    # Save test config
    config_file = test_data_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f, indent=2)
    print(f"Created test config: {config_file}")
    
    print("\nTo run the video renderer service:")
    print(f"python -m services.04_video_renderer --config {config_file}")
    
    print("\nNote: First run will download models and may take some time.")
    print("Check test_output/clips/ for generated video files.")

if __name__ == "__main__":
    main()
