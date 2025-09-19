"""
Unit tests for the video renderer service.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from services.04_video_renderer.service import VideoRendererService
from services.04_video_renderer.config import VideoRendererConfig


class TestVideoRendererService:
    """Test cases for VideoRendererService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test storyboard
        self.test_storyboard = [
            {
                "start": 0.0,
                "end": 4.0,
                "prompt": "wide shot of a caveman dancing around a campfire with beautiful lighting",
                "motion": "slow dolly",
                "seed": 12345,
                "model": "sdxl",
                "ref_image": None
            },
            {
                "start": 4.0,
                "end": 8.0,
                "prompt": "close up of the caveman's face with dramatic lighting and atmosphere",
                "motion": "zoom in",
                "seed": 12346,
                "model": "sdxl",
                "ref_image": None
            }
        ]
        
        # Create test config file
        self.config_data = {
            "io": {
                "input": str(self.temp_path / "storyboard.json"),
                "output_dir": str(self.temp_path / "clips"),
                "ref_image": None
            },
            "engine": {
                "type": "AnimateDiff",
                "animatediff": {
                    "model": "guoyww/animatediff-sdxl",
                    "pipeline_class": "AnimateDiffPipeline",
                    "num_inference_steps": 2,  # Reduced for testing
                    "guidance_scale": 7.5
                }
            },
            "video": {
                "fps": 8,  # Reduced for testing
                "resolution": [256, 256],  # Reduced for testing
                "format": "mp4",
                "codec": "h264",
                "bitrate": "1M"
            },
            "generation": {
                "batch_size": 1,
                "num_frames": 8,  # Fixed small number for testing
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
                "device": "cpu",  # Force CPU for testing
                "memory_efficient": True,
                "compile": False,
                "enable_xformers": False
            }
        }
        
        self.config_file = self.temp_path / "config.yaml"
        import yaml
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config_data, f)
        
        # Create test storyboard file
        storyboard_file = self.temp_path / "storyboard.json"
        with open(storyboard_file, 'w') as f:
            json.dump(self.test_storyboard, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_creation(self):
        """Test configuration creation."""
        service = VideoRendererService(str(self.config_file))
        assert isinstance(service.config, VideoRendererConfig)
        assert service.config.engine.type == "AnimateDiff"
        assert service.config.video.fps == 8
        assert service.config.performance.device == "cpu"
    
    def test_validate_inputs_success(self):
        """Test successful input validation."""
        service = VideoRendererService(str(self.config_file))
        assert service.validate_inputs() is True
    
    def test_validate_inputs_missing_storyboard(self):
        """Test input validation with missing storyboard file."""
        # Remove storyboard file
        (self.temp_path / "storyboard.json").unlink()
        
        service = VideoRendererService(str(self.config_file))
        assert service.validate_inputs() is False
    
    def test_load_storyboard(self):
        """Test storyboard loading."""
        service = VideoRendererService(str(self.config_file))
        storyboard = service._load_storyboard()
        
        assert len(storyboard) == 2
        assert storyboard[0]["start"] == 0.0
        assert storyboard[0]["end"] == 4.0
        assert "caveman" in storyboard[0]["prompt"]
    
    def test_calculate_num_frames(self):
        """Test frame calculation."""
        service = VideoRendererService(str(self.config_file))
        
        # 4 second duration at 8 fps = 32 frames
        frames = service._calculate_num_frames(4.0)
        assert frames == 32
        
        # Very short duration should get minimum frames
        frames = service._calculate_num_frames(0.5)
        assert frames == 8  # minimum
    
    def test_get_device(self):
        """Test device selection."""
        service = VideoRendererService(str(self.config_file))
        device = service._get_device()
        assert device == "cpu"  # As specified in test config
    
    @patch('services.04_video_renderer.service.AnimateDiffPipeline')
    @patch('services.04_video_renderer.service.DPMSolverMultistepScheduler')
    def test_load_animatediff_pipeline(self, mock_scheduler, mock_pipeline):
        """Test AnimateDiff pipeline loading."""
        # Mock the pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance
        mock_pipeline_instance.scheduler = Mock()
        
        # Mock scheduler
        mock_scheduler_instance = Mock()
        mock_scheduler.from_config.return_value = mock_scheduler_instance
        
        service = VideoRendererService(str(self.config_file))
        service._load_animatediff_pipeline("cpu")
        
        # Verify pipeline was loaded
        mock_pipeline.from_pretrained.assert_called_once()
        assert service.pipeline == mock_pipeline_instance
    
    @patch('services.04_video_renderer.service.export_to_video')
    def test_save_video(self, mock_export):
        """Test video saving."""
        import numpy as np
        
        service = VideoRendererService(str(self.config_file))
        frames = np.random.rand(8, 256, 256, 3)  # Mock frames
        output_path = self.temp_path / "test_video.mp4"
        
        service._save_video(frames, output_path, fps=8)
        
        mock_export.assert_called_once_with(frames, str(output_path), fps=8)
    
    @patch('services.04_video_renderer.service.AnimateDiffPipeline')
    @patch('services.04_video_renderer.service.export_to_video')
    def test_generate_text_to_video(self, mock_export, mock_pipeline):
        """Test text-to-video generation."""
        import numpy as np
        
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance
        
        # Mock generation result
        mock_result = Mock()
        mock_result.frames = [np.random.rand(8, 256, 256, 3)]
        mock_pipeline_instance.return_value = mock_result
        
        service = VideoRendererService(str(self.config_file))
        service.pipeline = mock_pipeline_instance
        
        shot = self.test_storyboard[0]
        frames = service._generate_text_to_video(shot)
        
        assert frames is not None
        mock_pipeline_instance.assert_called_once()
    
    @patch('services.04_video_renderer.service.AnimateDiffPipeline')
    @patch('services.04_video_renderer.service.export_to_video')
    def test_generate_shot(self, mock_export, mock_pipeline):
        """Test single shot generation."""
        import numpy as np
        
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance
        
        # Mock generation result
        mock_result = Mock()
        mock_result.frames = [np.random.rand(8, 256, 256, 3)]
        mock_pipeline_instance.return_value = mock_result
        
        service = VideoRendererService(str(self.config_file))
        service.pipeline = mock_pipeline_instance
        
        # Create output directory
        output_dir = self.temp_path / "clips"
        output_dir.mkdir(exist_ok=True)
        
        shot = self.test_storyboard[0]
        service._generate_shot(shot, 0)
        
        # Verify export was called
        mock_export.assert_called_once()
    
    def test_validate_outputs_no_directory(self):
        """Test output validation when output directory doesn't exist."""
        service = VideoRendererService(str(self.config_file))
        assert service.validate_outputs() is False
    
    def test_validate_outputs_no_videos(self):
        """Test output validation when no videos were generated."""
        # Create empty output directory
        output_dir = self.temp_path / "clips"
        output_dir.mkdir(exist_ok=True)
        
        service = VideoRendererService(str(self.config_file))
        assert service.validate_outputs() is False
    
    def test_validate_outputs_success(self):
        """Test successful output validation."""
        # Create output directory and a dummy video file
        output_dir = self.temp_path / "clips"
        output_dir.mkdir(exist_ok=True)
        (output_dir / "test_video.mp4").touch()
        
        service = VideoRendererService(str(self.config_file))
        assert service.validate_outputs() is True
