"""
Unit tests for the prompt compactor service.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from services.00_prompt_compactor.service import PromptCompactorService


class TestPromptCompactorService:
    """Test cases for the prompt compactor service."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return {
            "model": {
                "type": "local",
                "local": {
                    "model_path": "models/test-model",
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            "io": {
                "input_file": "data/song_brief.txt",
                "output_file": "data/suno_prompt.txt",
                "max_chars": 200
            },
            "prompt": {
                "system_template": "You are a music prompt specialist.",
                "user_template": "Convert this song brief: {input_text}"
            },
            "validation": {
                "banned_chars": ["#", "🎵"],
                "max_lines": 1,
                "min_chars": 10
            }
        }
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_file:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as output_file:
                    yield {
                        'config': config_file.name,
                        'input': input_file.name,
                        'output': output_file.name
                    }
        
        # Cleanup
        for file_path in [config_file.name, input_file.name, output_file.name]:
            Path(file_path).unlink(missing_ok=True)
    
    def test_service_initialization(self, sample_config, temp_files):
        """Test service initialization with valid config."""
        # Write config to temp file
        with open(temp_files['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        # Update config to use temp files
        sample_config['io']['input_file'] = temp_files['input']
        sample_config['io']['output_file'] = temp_files['output']
        
        with open(temp_files['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create input file
        with open(temp_files['input'], 'w') as f:
            f.write("A rock song with electric guitars")
        
        service = PromptCompactorService(temp_files['config'])
        assert service.config.model.type == "local"
        assert service.config.io.max_chars == 200
    
    def test_input_validation_success(self, sample_config, temp_files):
        """Test successful input validation."""
        # Write config to temp file
        sample_config['io']['input_file'] = temp_files['input']
        sample_config['io']['output_file'] = temp_files['output']
        
        with open(temp_files['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create valid input file
        with open(temp_files['input'], 'w') as f:
            f.write("A rock song with electric guitars")
        
        service = PromptCompactorService(temp_files['config'])
        assert service.validate_inputs() is True
    
    def test_input_validation_file_not_found(self, sample_config, temp_files):
        """Test input validation with missing file."""
        # Write config to temp file
        sample_config['io']['input_file'] = "nonexistent.txt"
        sample_config['io']['output_file'] = temp_files['output']
        
        with open(temp_files['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        service = PromptCompactorService(temp_files['config'])
        assert service.validate_inputs() is False
    
    def test_input_validation_file_too_large(self, sample_config, temp_files):
        """Test input validation with oversized file."""
        # Write config to temp file
        sample_config['io']['input_file'] = temp_files['input']
        sample_config['io']['output_file'] = temp_files['output']
        
        with open(temp_files['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create oversized input file (6KB)
        with open(temp_files['input'], 'w') as f:
            f.write("x" * 6144)  # 6KB
        
        service = PromptCompactorService(temp_files['config'])
        assert service.validate_inputs() is False
    
    def test_output_validation_success(self, sample_config, temp_files):
        """Test successful output validation."""
        # Write config to temp file
        sample_config['io']['input_file'] = temp_files['input']
        sample_config['io']['output_file'] = temp_files['output']
        
        with open(temp_files['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create valid output file
        with open(temp_files['output'], 'w') as f:
            f.write("energetic rock music with electric guitars and drums")
        
        service = PromptCompactorService(temp_files['config'])
        assert service.validate_outputs() is True
    
    def test_output_validation_too_long(self, sample_config, temp_files):
        """Test output validation with oversized content."""
        # Write config to temp file
        sample_config['io']['input_file'] = temp_files['input']
        sample_config['io']['output_file'] = temp_files['output']
        sample_config['io']['max_chars'] = 50
        
        with open(temp_files['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create oversized output file
        with open(temp_files['output'], 'w') as f:
            f.write("x" * 60)  # 60 chars > 50 limit
        
        service = PromptCompactorService(temp_files['config'])
        assert service.validate_outputs() is False
    
    def test_output_validation_banned_chars(self, sample_config, temp_files):
        """Test output validation with banned characters."""
        # Write config to temp file
        sample_config['io']['input_file'] = temp_files['input']
        sample_config['io']['output_file'] = temp_files['output']
        
        with open(temp_files['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create output file with banned character
        with open(temp_files['output'], 'w') as f:
            f.write("rock music # with guitars")
        
        service = PromptCompactorService(temp_files['config'])
        assert service.validate_outputs() is False
    
    def test_local_model_generation(self, sample_config, temp_files):
        """Test local model prompt generation."""
        # Write config to temp file
        sample_config['io']['input_file'] = temp_files['input']
        sample_config['io']['output_file'] = temp_files['output']
        
        with open(temp_files['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create input file
        with open(temp_files['input'], 'w') as f:
            f.write("A rock song with electric guitars")
        
        service = PromptCompactorService(temp_files['config'])
        prompt = service._generate_with_local("A rock song with electric guitars")
        
        assert len(prompt) <= 200
        assert "rock" in prompt.lower()
        assert "#" not in prompt
        assert "\n" not in prompt
    
    @patch('services.00_prompt_compactor.service.openai')
    def test_openai_model_generation(self, mock_openai, sample_config, temp_files):
        """Test OpenAI model prompt generation."""
        # Update config for OpenAI
        sample_config['model']['type'] = 'openai'
        sample_config['model']['openai'] = {
            'model': 'gpt-4o-mini',
            'max_tokens': 100,
            'temperature': 0.7,
            'api_key_env': 'OPENAI_API_KEY'
        }
        sample_config['io']['input_file'] = temp_files['input']
        sample_config['io']['output_file'] = temp_files['output']
        
        with open(temp_files['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create input file
        with open(temp_files['input'], 'w') as f:
            f.write("A rock song with electric guitars")
        
        # Mock OpenAI response
        mock_client = mock_openai.OpenAI.return_value
        mock_response = mock_client.chat.completions.create.return_value
        mock_response.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': 'energetic rock music with electric guitars and drums'})})]
        
        # Mock environment variable
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = PromptCompactorService(temp_files['config'])
            prompt = service._generate_with_openai("A rock song with electric guitars")
            
            assert prompt == "energetic rock music with electric guitars and drums"
            mock_client.chat.completions.create.assert_called_once()
    
    def test_service_run_success(self, sample_config, temp_files):
        """Test successful service run."""
        # Write config to temp file
        sample_config['io']['input_file'] = temp_files['input']
        sample_config['io']['output_file'] = temp_files['output']
        
        with open(temp_files['config'], 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create input file
        with open(temp_files['input'], 'w') as f:
            f.write("A rock song with electric guitars")
        
        service = PromptCompactorService(temp_files['config'])
        exit_code = service.run()
        
        assert exit_code == 0
        assert Path(temp_files['output']).exists()
        
        # Check output content
        with open(temp_files['output'], 'r') as f:
            content = f.read().strip()
            assert len(content) <= 200
            assert len(content) >= 10 