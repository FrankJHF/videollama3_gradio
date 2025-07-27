#!/usr/bin/env python3
"""
Test Suite for VideoSampler

Comprehensive tests to validate the VideoSampler functionality,
performance, and integration with VideoLLaMA3.
"""

import sys
import os
import time
import unittest
from typing import Tuple, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.video_sampler import (
    VideoSampler, 
    VideoSamplerError,
    extract_uniform_frames,
    extract_frames_by_interval,
    get_video_metadata
)
import numpy as np


class TestVideoSampler(unittest.TestCase):
    """Test cases for VideoSampler functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test video path"""
        cls.test_video = None
        
        # Look for test video in examples
        examples_dir = os.path.join(os.path.dirname(__file__), "../../examples")
        if os.path.exists(examples_dir):
            video_files = [f for f in os.listdir(examples_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            if video_files:
                cls.test_video = os.path.join(examples_dir, video_files[0])
        
        if not cls.test_video or not os.path.exists(cls.test_video):
            raise unittest.SkipTest("No test video available")
    
    def test_video_sampler_initialization(self):
        """Test VideoSampler initialization"""
        with VideoSampler(self.test_video) as sampler:
            self.assertGreater(sampler.total_frames, 0)
            self.assertGreater(sampler.fps, 0)
            self.assertGreater(sampler.duration, 0)
    
    def test_uniform_sampling(self):
        """Test uniform frame sampling"""
        with VideoSampler(self.test_video) as sampler:
            frames, timestamps = sampler.uniform_sample(60)
            
            self.assertEqual(len(frames), 60)
            self.assertEqual(len(timestamps), 60)
            self.assertTrue(all(isinstance(t, float) for t in timestamps))
            self.assertTrue(timestamps == sorted(timestamps))  # Should be sorted
    
    def test_time_based_sampling(self):
        """Test time-based sampling"""
        with VideoSampler(self.test_video) as sampler:
            frames, timestamps = sampler.time_based_sample(2.0)  # Every 2 seconds
            
            self.assertGreater(len(frames), 0)
            self.assertEqual(len(frames), len(timestamps))
            
            # Check intervals are approximately 2 seconds
            if len(timestamps) > 1:
                intervals = np.diff(timestamps)
                avg_interval = np.mean(intervals)
                self.assertAlmostEqual(avg_interval, 2.0, delta=0.5)
    
    def test_key_moments_sampling(self):
        """Test key moments sampling"""
        with VideoSampler(self.test_video) as sampler:
            key_moments = [1.0, 5.0, 10.0]
            frames, timestamps = sampler.key_moments_sample(key_moments, context_frames=2)
            
            self.assertGreaterEqual(len(frames), len(key_moments))
            self.assertEqual(len(frames), len(timestamps))
            
            # Check that key moments are represented
            for moment in key_moments:
                if moment < sampler.duration:
                    closest_timestamp = min(timestamps, key=lambda t: abs(t - moment))
                    self.assertLess(abs(closest_timestamp - moment), 1.0)
    
    def test_adaptive_sampling(self):
        """Test adaptive sampling"""
        with VideoSampler(self.test_video) as sampler:
            frames, timestamps = sampler.adaptive_sample(100, density_factor=2.0)
            
            self.assertEqual(len(frames), 100)
            self.assertEqual(len(timestamps), 100)
            
            # Check that start and end have higher density
            intervals = np.diff(timestamps)
            start_interval = np.mean(intervals[:5])
            middle_interval = np.mean(intervals[45:55])
            end_interval = np.mean(intervals[-5:])
            
            # Start and end should have smaller intervals (higher density)
            self.assertLessEqual(start_interval, middle_interval)
            self.assertLessEqual(end_interval, middle_interval)
    
    def test_time_range_sampling(self):
        """Test sampling with time range restrictions"""
        with VideoSampler(self.test_video) as sampler:
            start_time = sampler.duration * 0.25
            end_time = sampler.duration * 0.75
            
            frames, timestamps = sampler.uniform_sample(50, start_time, end_time)
            
            self.assertEqual(len(frames), 50)
            self.assertGreaterEqual(min(timestamps), start_time - 0.1)  # Small tolerance
            self.assertLessEqual(max(timestamps), end_time + 0.1)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test invalid video path
        with self.assertRaises(VideoSamplerError):
            VideoSampler("nonexistent_video.mp4")
        
        # Test invalid time range
        with VideoSampler(self.test_video) as sampler:
            with self.assertRaises(VideoSamplerError):
                sampler.uniform_sample(10, start_time=100, end_time=50)
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test extract_uniform_frames
        frames, timestamps = extract_uniform_frames(self.test_video, 30)
        self.assertEqual(len(frames), 30)
        self.assertEqual(len(timestamps), 30)
        
        # Test extract_frames_by_interval
        frames2, timestamps2 = extract_frames_by_interval(self.test_video, 3.0)
        self.assertGreater(len(frames2), 0)
        
        # Test get_video_metadata
        metadata = get_video_metadata(self.test_video)
        self.assertIn('total_frames', metadata)
        self.assertIn('fps', metadata)
        self.assertIn('duration', metadata)
    
    def test_performance(self):
        """Test performance benchmarks"""
        frame_counts = [50, 100, 180]
        
        for num_frames in frame_counts:
            start_time = time.time()
            
            frames, timestamps = extract_uniform_frames(self.test_video, num_frames)
            
            extraction_time = time.time() - start_time
            
            # Performance assertions
            self.assertEqual(len(frames), num_frames)
            self.assertLess(extraction_time, 5.0)  # Should complete within 5 seconds
            
            frames_per_second = num_frames / extraction_time
            self.assertGreater(frames_per_second, 10)  # At least 10 frames/sec
    
    def test_memory_efficiency(self):
        """Test memory usage is reasonable"""
        frames, timestamps = extract_uniform_frames(self.test_video, 100)
        
        # Calculate memory usage
        frame_memory = frames.nbytes
        memory_per_frame = frame_memory / len(frames)
        
        # Should be reasonable for typical video resolution
        self.assertLess(memory_per_frame, 2 * 1024 * 1024)  # Less than 2MB per frame


class TestVideoLLaMA3Integration(unittest.TestCase):
    """Test integration with VideoLLaMA3 workflow"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test video"""
        cls.test_video = TestVideoSampler.test_video
        if not cls.test_video:
            raise unittest.SkipTest("No test video available")
    
    def test_videollama3_input_format(self):
        """Test compatibility with VideoLLaMA3 input format"""
        with VideoSampler(self.test_video) as sampler:
            frames, timestamps = sampler.uniform_sample(160)
            
            # Create VideoLLaMA3 compatible input
            videollama3_input = {
                "type": "video",
                "timestamps": timestamps,
                "num_frames": len(frames),
                "frames": frames
            }
            
            # Validate format
            self.assertEqual(videollama3_input["type"], "video")
            self.assertEqual(videollama3_input["num_frames"], 160)
            self.assertEqual(len(videollama3_input["timestamps"]), 160)
            self.assertEqual(videollama3_input["frames"].shape[0], 160)
    
    def test_emergency_detection_sampling(self):
        """Test optimal sampling for emergency detection scenarios"""
        with VideoSampler(self.test_video) as sampler:
            # Adaptive sampling for emergency detection
            frames, timestamps = sampler.adaptive_sample(180, density_factor=1.5)
            
            # Analyze temporal distribution
            quarter_duration = sampler.duration / 4
            quarters = [
                (0, quarter_duration),
                (quarter_duration, 2*quarter_duration), 
                (2*quarter_duration, 3*quarter_duration),
                (3*quarter_duration, sampler.duration)
            ]
            
            frame_counts = []
            for start, end in quarters:
                count = sum(1 for t in timestamps if start <= t < end)
                frame_counts.append(count)
            
            # First and last quarters should have more frames (higher density)
            self.assertGreaterEqual(frame_counts[0], frame_counts[1])
            self.assertGreaterEqual(frame_counts[3], frame_counts[2])


def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    print("\n=== Performance Benchmark ===")
    
    test_video = TestVideoSampler.test_video
    if not test_video:
        print("No test video available for benchmark")
        return
    
    frame_counts = [50, 100, 180, 360, 720]
    
    print(f"\nTesting video: {os.path.basename(test_video)}")
    
    with VideoSampler(test_video) as sampler:
        print(f"Video info: {sampler.total_frames} frames, {sampler.fps:.1f} fps, {sampler.duration:.1f}s")
    
    print(f"\n{'Frames':<8} {'Time (s)':<10} {'Speed (fps)':<12} {'Memory (MB)':<12}")
    print("-" * 50)
    
    for num_frames in frame_counts:
        start_time = time.time()
        
        frames, timestamps = extract_uniform_frames(test_video, num_frames)
        
        extraction_time = time.time() - start_time
        speed = num_frames / extraction_time
        memory_mb = frames.nbytes / 1024 / 1024
        
        print(f"{num_frames:<8} {extraction_time:<10.3f} {speed:<12.1f} {memory_mb:<12.1f}")


def main():
    """Run all tests and benchmarks"""
    print("VideoSampler Test Suite")
    print("=" * 40)
    
    # Check if decord is available
    try:
        import decord
        print("✓ decord library available")
    except ImportError:
        print("✗ decord library not available - tests will be skipped")
        print("Install with: pip install decord")
        return
    
    # Run unit tests
    print("\nRunning unit tests...")
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestVideoSampler))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestVideoLLaMA3Integration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance benchmark if tests passed
    if result.wasSuccessful():
        run_performance_benchmark()
        print("\n✓ All tests passed successfully!")
    else:
        print(f"\n✗ {len(result.failures + result.errors)} test(s) failed")


if __name__ == "__main__":
    main()