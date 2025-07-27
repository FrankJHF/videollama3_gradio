"""
Utility package for VideoLLaMA3 GUI

This package provides various utility functions for video processing,
frame sampling, and other common operations.
"""

from .video_sampler import (
    VideoSampler,
    VideoSamplerError,
    extract_uniform_frames,
    extract_frames_by_interval,
    get_video_metadata
)

__all__ = [
    'VideoSampler',
    'VideoSamplerError', 
    'extract_uniform_frames',
    'extract_frames_by_interval',
    'get_video_metadata'
]

__version__ = '1.0.0'