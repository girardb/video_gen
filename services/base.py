"""
Base service class for all micro-services in the music video generation pipeline.
"""

import argparse
import logging
import logging.config
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel


class ServiceConfig(BaseModel):
    """Base configuration model for services."""
    pass


class BaseService(ABC):
    """Base class for all micro-services."""
    
    def __init__(self, config_path: str, **kwargs):
        """Initialize the service with configuration."""
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _load_config(self, config_path: str) -> ServiceConfig:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return self._create_config(config_data)
    
    def _create_config(self, config_data: Dict[str, Any]) -> ServiceConfig:
        """Create configuration object from data."""
        # Override in subclasses to use specific config classes
        return ServiceConfig(**config_data)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create log filename with date
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        service_name = self.__class__.__name__.lower()
        log_file = log_dir / f"{date_str}_{service_name}.log"
        
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                },
            },
            "handlers": {
                "default": {
                    "level": "INFO",
                    "formatter": "standard",
                    "class": "logging.StreamHandler",
                },
                "file": {
                    "level": "INFO",
                    "formatter": "standard",
                    "class": "logging.FileHandler",
                    "filename": str(log_file),
                    "mode": "a",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default", "file"],
                    "level": "INFO",
                    "propagate": False
                }
            }
        }
        
        logging.config.dictConfig(logging_config)
    
    @abstractmethod
    def run(self) -> int:
        """
        Run the service.
        
        Returns:
            int: Exit code (0 for success, non-zero for error)
        """
        pass
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        return True
    
    def validate_outputs(self) -> bool:
        """Validate that outputs were created successfully."""
        return True


def create_argument_parser(service_name: str) -> argparse.ArgumentParser:
    """Create a standardized argument parser for services."""
    parser = argparse.ArgumentParser(
        description=f"{service_name} service for music video generation pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser


def run_service(service_class: type, service_name: str) -> None:
    """Standard entry point for services."""
    parser = create_argument_parser(service_name)
    args = parser.parse_args()
    
    try:
        service = service_class(args.config)
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        exit_code = service.run()
        sys.exit(exit_code)
        
    except Exception as e:
        logging.error(f"Service failed: {e}")
        sys.exit(1) 