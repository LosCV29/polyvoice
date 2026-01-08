"""Tool system for PolyVoice.

This module provides a registry-based tool system that:
- Builds tool definitions dynamically based on enabled features
- Dispatches tool calls to appropriate handlers
- Provides consistent error handling
"""
from .registry import ToolRegistry
from .definitions import build_tools

__all__ = [
    "ToolRegistry",
    "build_tools",
]
