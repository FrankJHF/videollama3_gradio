#!/usr/bin/env python3
"""
High-efficiency Video Frame Sampling Utility

This module provides optimized video frame sampling with timestamp extraction,
designed for memory-efficient, storage-friendly, and compute-efficient operations.
Based on decord VideoReader for direct frame access without full video traversal.

Author: Claude Code
License: MIT
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import warnings

try:
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    warnings.warn("decord not available. Please install: pip install decord")

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    warnings.warn("ffmpeg-python not available. Install for enhanced metadata: pip install ffmpeg-python")

# Configure logging
logger = logging.getLogger(__name__)


class VideoSamplerError(Exception):
    """Custom exception for VideoSampler errors"""
    pass


class VideoSampler:
    """
    High-efficiency video frame sampler with timestamp extraction.
    
    Features:
    - Direct frame access via index (O(1) complexity)
    - Multiple sampling strategies (uniform, time-based, adaptive)
    - Memory-efficient batch processing
    - Precise timestamp calculation
    - Hardware acceleration support
    - Comprehensive error handling
    
    Example:
        >>> sampler = VideoSampler("video.mp4")
        >>> frames, timestamps = sampler.uniform_sample(180)
        >>> print(f"Extracted {len(frames)} frames spanning {timestamps[-1]:.2f}s")
    """
    
    def __init__(self, 
                 video_path: str,
                 use_gpu: bool = False,
                 num_threads: int = 2,
                 preload_metadata: bool = True):
        """
        Initialize VideoSampler.
        
        Args:
            video_path: Path to video file
            use_gpu: Whether to use GPU acceleration (requires CUDA)
            num_threads: Number of decoding threads
            preload_metadata: Whether to preload video metadata
        
        Raises:
            VideoSamplerError: If video file invalid or decord unavailable
        """
        if not DECORD_AVAILABLE:
            raise VideoSamplerError("decord library is required but not available")
        
        self.video_path = video_path
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        
        # Validate video file
        if not os.path.exists(video_path):
            raise VideoSamplerError(f"Video file not found: {video_path}")
        
        try:
            # Initialize VideoReader
            ctx = gpu(0) if use_gpu else cpu(0)
            self.vreader = VideoReader(video_path, ctx=ctx, num_threads=num_threads)
            
            # Extract basic metadata
            self.total_frames = len(self.vreader)
            self.fps = self.vreader.get_avg_fps()
            self.duration = self.total_frames / self.fps
            
            # Enhanced metadata via ffmpeg if available
            self.metadata = {}
            if preload_metadata and FFMPEG_AVAILABLE:
                self.metadata = self._extract_enhanced_metadata()
                
        except Exception as e:
            raise VideoSamplerError(f"Failed to initialize video reader: {str(e)}")
    
    def _extract_enhanced_metadata(self) -> Dict[str, Any]:
        """Extract enhanced metadata using ffmpeg"""
        try:
            probe = ffmpeg.probe(self.video_path)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            metadata = {
                'codec': video_stream.get('codec_name', 'unknown'),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'bit_rate': int(video_stream.get('bit_rate', 0)),
                'duration': float(video_stream.get('duration', self.duration)),
                'pixel_format': video_stream.get('pix_fmt', 'unknown'),
                'frame_count': int(video_stream.get('nb_frames', self.total_frames))
            }
            
            # Update duration and frame count if more accurate
            if metadata['duration'] > 0 and metadata['frame_count'] > 0:
                self.duration = metadata['duration']
                self.total_frames = metadata['frame_count']
                self.fps = self.total_frames / self.duration
                
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to extract enhanced metadata: {e}")
            return {}
    
    def uniform_sample(self, 
                      num_frames: int,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Uniform sampling of video frames.
        
        Args:
            num_frames: Target number of frames to extract
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            
        Returns:
            Tuple of (frames, timestamps)
            - frames: numpy array of shape (N, H, W, C)
            - timestamps: list of timestamps in seconds
        """
        start_frame, end_frame = self._time_to_frame_range(start_time, end_time)
        effective_frames = end_frame - start_frame
        
        if num_frames >= effective_frames:
            # Sample all available frames
            indices = np.arange(start_frame, end_frame, dtype=int)
        else:
            # Uniform sampling
            indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        
        return self._extract_frames_by_indices(indices)
    
    def time_based_sample(self, 
                         interval_seconds: float,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Time-interval based sampling.
        
        Args:
            interval_seconds: Interval between samples in seconds
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            
        Returns:
            Tuple of (frames, timestamps)
        """
        start_frame, end_frame = self._time_to_frame_range(start_time, end_time)
        
        frame_interval = max(1, int(interval_seconds * self.fps))
        indices = np.arange(start_frame, end_frame, frame_interval, dtype=int)
        
        return self._extract_frames_by_indices(indices)
    
    def key_moments_sample(self, 
                          moments: List[float],
                          context_frames: int = 0) -> Tuple[np.ndarray, List[float]]:
        """
        Sample frames at specific time moments.
        
        Args:
            moments: List of time moments in seconds
            context_frames: Additional frames around each moment
            
        Returns:
            Tuple of (frames, timestamps)
        """
        indices = []
        
        for moment in moments:
            center_frame = int(moment * self.fps)
            center_frame = max(0, min(center_frame, self.total_frames - 1))
            
            if context_frames > 0:
                # Add context frames around the key moment
                start_ctx = max(0, center_frame - context_frames)
                end_ctx = min(self.total_frames, center_frame + context_frames + 1)
                indices.extend(range(start_ctx, end_ctx))
            else:
                indices.append(center_frame)
        
        # Remove duplicates and sort
        indices = sorted(set(indices))
        indices = np.array(indices, dtype=int)
        
        return self._extract_frames_by_indices(indices)
    
    def adaptive_sample(self, 
                       num_frames: int,
                       density_factor: float = 1.5,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Adaptive sampling with higher density at beginning and end.
        
        Args:
            num_frames: Target number of frames
            density_factor: Higher values increase density at endpoints
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            
        Returns:
            Tuple of (frames, timestamps)
        """
        start_frame, end_frame = self._time_to_frame_range(start_time, end_time)
        effective_frames = end_frame - start_frame
        
        if num_frames >= effective_frames:
            indices = np.arange(start_frame, end_frame, dtype=int)
        else:
            # Generate adaptive distribution
            t = np.linspace(0, 1, num_frames)
            # Apply density function: higher density at start and end
            adaptive_t = t + density_factor * np.sin(np.pi * t) * (t * (1 - t))
            adaptive_t = adaptive_t / adaptive_t[-1]  # Normalize
            
            indices = start_frame + (adaptive_t * (effective_frames - 1)).astype(int)
        
        return self._extract_frames_by_indices(indices)
    
    def _time_to_frame_range(self, 
                           start_time: Optional[float], 
                           end_time: Optional[float]) -> Tuple[int, int]:
        """Convert time range to frame range"""
        start_frame = 0 if start_time is None else max(0, int(start_time * self.fps))
        end_frame = self.total_frames if end_time is None else min(self.total_frames, int(end_time * self.fps))
        
        if start_frame >= end_frame:
            raise VideoSamplerError(f"Invalid time range: start_frame({start_frame}) >= end_frame({end_frame})")
        
        return start_frame, end_frame
    
    def _extract_frames_by_indices(self, indices: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Extract frames by frame indices"""
        if len(indices) == 0:
            raise VideoSamplerError("No frames to extract")
        
        # Validate indices
        valid_indices = indices[(indices >= 0) & (indices < self.total_frames)]
        if len(valid_indices) != len(indices):
            logger.warning(f"Filtered {len(indices) - len(valid_indices)} invalid frame indices")
            indices = valid_indices
        
        try:
            # Batch extraction - key efficiency optimization
            frames = self.vreader.get_batch(indices.tolist()).asnumpy()
            timestamps = [idx / self.fps for idx in indices]
            
            logger.info(f"Extracted {len(frames)} frames from {self.video_path}")
            return frames, timestamps
            
        except Exception as e:
            raise VideoSamplerError(f"Failed to extract frames: {str(e)}")
    
    def get_video_info(self) -> Dict[str, Any]:
        """Get comprehensive video information"""
        info = {
            'path': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'duration': self.duration,
            'use_gpu': self.use_gpu,
            'num_threads': self.num_threads
        }
        info.update(self.metadata)
        return info
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        if hasattr(self, 'vreader'):
            del self.vreader
    
    def __repr__(self):
        return (f"VideoSampler(path='{self.video_path}', "
                f"frames={self.total_frames}, fps={self.fps:.2f}, "
                f"duration={self.duration:.2f}s)")


# Convenience functions for quick usage
def extract_uniform_frames(video_path: str, 
                          num_frames: int = 180,
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None) -> Tuple[np.ndarray, List[float]]:
    """
    Quick uniform frame extraction.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        
    Returns:
        Tuple of (frames, timestamps)
    """
    with VideoSampler(video_path) as sampler:
        return sampler.uniform_sample(num_frames, start_time, end_time)


def extract_frames_by_interval(video_path: str,
                              interval_seconds: float,
                              start_time: Optional[float] = None,
                              end_time: Optional[float] = None) -> Tuple[np.ndarray, List[float]]:
    """
    Quick interval-based frame extraction.
    
    Args:
        video_path: Path to video file
        interval_seconds: Interval between frames in seconds
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        
    Returns:
        Tuple of (frames, timestamps)
    """
    with VideoSampler(video_path) as sampler:
        return sampler.time_based_sample(interval_seconds, start_time, end_time)


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Quick video metadata extraction.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video metadata
    """
    with VideoSampler(video_path) as sampler:
        return sampler.get_video_info()


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_sampler.py <video_path> [num_frames]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    
    try:
        # Test uniform sampling
        frames, timestamps = extract_uniform_frames(video_path, num_frames)
        print(f"✓ Extracted {len(frames)} frames")
        print(f"  Time span: {timestamps[0]:.2f}s - {timestamps[-1]:.2f}s")
        print(f"  Frame shape: {frames.shape}")
        
        # Test metadata extraction
        metadata = get_video_metadata(video_path)
        print(f"✓ Video metadata: {metadata}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)