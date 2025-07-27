#!/usr/bin/env python3
"""
Universal utility for saving frames and timestamps analysis results.

This module provides a standardized way to save video analysis results, ensuring
strict preservation of order for both frames and timestamps.

Features:
- Auto-detects frame formats (numpy arrays, PIL Images, lists)
- Saves frames as images with timestamp-based naming
- Exports timestamps to CSV in original order
- Creates organized results directory structure
- Handles various input formats gracefully

Usage:
    from utils.save_frames import save_analysis_results
    save_analysis_results(frames, timestamps, "my_analysis")
"""

import os
import csv
import numpy as np
from pathlib import Path
from typing import List, Union, Any
import logging

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


def save_analysis_results(
    frames: Union[List, np.ndarray, Any],
    timestamps: List[float],
    analysis_name: str = "analysis",
    output_dir: str = "results"
) -> Path:
    """
    Save frames and timestamps with strict order preservation.
    
    Args:
        frames: Video frames in any supported format
        timestamps: Corresponding timestamps in seconds
        analysis_name: Name for this analysis (used in directory naming)
        output_dir: Base output directory
    
    Returns:
        Path to the results directory
    
    Raises:
        ValueError: If frames and timestamps lengths don't match
        ImportError: If PIL is not available
    """
    
    if not PIL_AVAILABLE:
        raise ImportError(
            "Pillow (PIL) is required for image saving. "
            "Install with: pip install Pillow"
        )
    
    if len(frames) != len(timestamps):
        raise ValueError(
            f"Frames count ({len(frames)}) must match timestamps count ({len(timestamps)})"
        )
    
    # Create results directory
    results_dir = Path(output_dir) / analysis_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(frames)} frames to {results_dir}")
    
    # Convert frames to standardized numpy format
    frames_array = _normalize_frames(frames)
    
    # Save frames as images
    saved_files = _save_frames_as_images(frames_array, timestamps, results_dir)
    
    # Save timestamps as CSV
    _save_timestamps_csv(timestamps, results_dir)
    
    logger.info(f"Saved {len(saved_files)} frames and timestamps successfully")
    return results_dir


def _normalize_frames(frames: Union[List, np.ndarray, Any]) -> np.ndarray:
    """Convert various frame formats to standardized numpy array."""
    
    if isinstance(frames, list):
        if len(frames) == 0:
            return np.array([])
        
        # Handle PIL Images
        if hasattr(frames[0], 'save'):  # PIL Image check
            frames = [np.array(frame) for frame in frames]
        
        frames = np.array(frames)
    
    elif hasattr(frames, 'numpy'):  # Handle torch tensors
        frames = frames.numpy()
    
    # Ensure we have a numpy array
    if not isinstance(frames, np.ndarray):
        frames = np.array(frames)
    
    # Handle channel format conversion
    if frames.ndim == 4:
        # (N, C, H, W) -> (N, H, W, C)
        if frames.shape[1] in [1, 3, 4]:  # Common channel sizes
            frames = np.transpose(frames, (0, 2, 3, 1))
    
    return frames


def _save_frames_as_images(
    frames: np.ndarray, 
    timestamps: List[float], 
    output_dir: Path
) -> List[Path]:
    """Save frames as images with timestamp-based naming."""
    
    saved_files = []
    
    for idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        # Ensure frame is in correct format
        if frame.dtype != np.uint8:
            if frame.max() > 1.0:
                frame = frame.astype(np.uint8)
            else:
                frame = (frame * 255).astype(np.uint8)
        
        # Create filename with zero-padded index and precise timestamp
        filename = f"frame_{idx:04d}_time_{timestamp:.4f}s.jpg"
        filepath = output_dir / filename
        
        try:
            img = Image.fromarray(frame)
            # Convert RGB to BGR if needed for display
            if img.mode == 'RGB':
                img.save(filepath, 'JPEG', quality=95)
            elif img.mode == 'RGBA':
                # Convert RGBA to RGB
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])
                rgb_img.save(filepath, 'JPEG', quality=95)
            else:
                img.save(filepath, 'JPEG', quality=95)
            
            saved_files.append(filepath)
            
        except Exception as e:
            logger.warning(f"Failed to save frame {idx}: {e}")
    
    return saved_files


def _save_timestamps_csv(timestamps: List[float], output_dir: Path) -> None:
    """Save timestamps to CSV file preserving original order."""
    
    csv_path = output_dir / "timestamps.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_index", "timestamp_seconds"])
        
        for idx, timestamp in enumerate(timestamps):
            writer.writerow([idx, f"{timestamp:.6f}"])
    
    logger.info(f"Timestamps saved to {csv_path}")


def quick_save(
    frames: Union[List, np.ndarray, Any],
    timestamps: List[float],
    prefix: str = "analysis"
) -> Path:
    """
    Quick save without custom directory naming.
    
    Args:
        frames: Video frames
        timestamps: Corresponding timestamps
        prefix: Prefix for results directory
    
    Returns:
        Path to saved directory
    """
    return save_analysis_results(frames, timestamps, prefix)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Save video analysis results")
    parser.add_argument("frames_path", help="Path to numpy file containing frames")
    parser.add_argument("timestamps_path", help="Path to file containing timestamps (one per line)")
    parser.add_argument("-n", "--name", default="analysis", help="Analysis name")
    parser.add_argument("-o", "--output", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        frames = np.load(args.frames_path)
        timestamps = [float(line.strip()) for line in open(args.timestamps_path)]
        
        results_dir = save_analysis_results(frames, timestamps, args.name, args.output)
        print(f"Results saved to: {results_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)