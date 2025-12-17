"""
Utility functions for the RAG backend
"""
import logging
from typing import Any, Dict


def setup_logging(level: str = "INFO"):
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration settings"""
    # Placeholder for config validation logic
    return True