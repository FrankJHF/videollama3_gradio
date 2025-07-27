# Video Sampler Examples

This directory contains practical examples demonstrating how to use the VideoSampler utility.

## Quick Start

```python
from utils.video_sampler import extract_uniform_frames

# Extract 180 frames uniformly
frames, timestamps = extract_uniform_frames("video.mp4", 180)
print(f"Extracted {len(frames)} frames spanning {timestamps[-1]:.2f}s")
```

## Advanced Usage

```python
from utils.video_sampler import VideoSampler

# Create sampler instance
with VideoSampler("emergency_video.mp4") as sampler:
    # Multiple sampling strategies
    frames1, ts1 = sampler.uniform_sample(180)           # Uniform
    frames2, ts2 = sampler.time_based_sample(1.0)        # Every 1 second
    frames3, ts3 = sampler.key_moments_sample([10, 30])  # At specific times
    frames4, ts4 = sampler.adaptive_sample(120)          # Adaptive density
    
    # Get metadata
    info = sampler.get_video_info()
    print(f"Video: {info['fps']:.2f} fps, {info['duration']:.2f}s")
```

## Performance Comparison

| Method | Memory Usage | Speed | Timestamp Accuracy |
|--------|-------------|-------|-------------------|
| VideoSampler | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| OpenCV Sequential | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| FFmpeg Export | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Error Handling

The VideoSampler includes comprehensive error handling:

- File validation
- Frame index bounds checking
- Hardware compatibility detection
- Graceful degradation for missing dependencies

## Integration with VideoLLaMA3

The VideoSampler is designed to be compatible with the existing VideoLLaMA3 pipeline:

```python
# Replace the existing load_video call
from utils.video_sampler import extract_uniform_frames

# Old way
frames, timestamps = load_video(video_path, max_frames=160)

# New way - more efficient
frames, timestamps = extract_uniform_frames(video_path, 160)
```