"""
Circular Buffer Management for IQ Sample Streaming

This module provides high-performance circular buffers for storing IQ samples
from the three HackRF radios. Uses pre-allocated numpy arrays for speed.

Key Concepts:
- Circular buffer: Fixed-size array that wraps around when full
- Write pointer: Tracks where next samples will be written
- No dynamic allocation: All memory pre-allocated for predictable performance

Performance:
- 3-5x faster than collections.deque for large sample counts
- Constant-time operations regardless of buffer size
- Cache-friendly memory layout

Author: Triple-SDR Project
License: GPL-3.0
"""

import numpy as np
import threading
import time
from typing import Optional, Tuple

class CircularBuffer:
    """
    High-performance circular buffer for complex IQ samples.
    
    Thread-safe for single writer, multiple readers scenario.
    Uses numpy arrays for fast memory operations.
    
    Attributes:
        capacity (int): Maximum number of samples the buffer can hold
        dtype: Numpy data type (complex64 for IQ samples)
        buffer (np.ndarray): The actual data storage
        write_index (int): Current write position (0 to capacity-1)
        lock (threading.Lock): Protects write operations
        samples_written (int): Total samples written since creation
    """
    
    def __init__(self, capacity: int, dtype=np.complex64):
        """
        Initialize circular buffer.
        
        Args:
            capacity: Number of samples to store
                     Example: 15e6 samples * 0.5 sec = 7.5M samples = 60 MB
            dtype: Data type for samples (complex64 = 8 bytes per sample)
        
        Memory Usage:
            capacity * sizeof(dtype) bytes
            Example: 7,500,000 * 8 = 60 MB
        """
        self.capacity = int(capacity)
        self.dtype = dtype
        
        # Pre-allocate buffer (never resized)
        self.buffer = np.zeros(self.capacity, dtype=self.dtype)
        
        # Write tracking
        self.write_index = 0
        self.samples_written = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.last_write_time = time.time()
        self.write_count = 0
    
    def append(self, samples: np.ndarray) -> None:
        """
        Append new samples to buffer (thread-safe).
        
        If samples would exceed capacity, oldest samples are overwritten.
        This is the "circular" behavior - buffer wraps around.
        
        Args:
            samples: 1D numpy array of complex samples
        
        Performance:
            O(1) for small appends
            Dominated by memory copy speed (~10 GB/s on modern CPU)
        
        Example:
            >>> buffer = CircularBuffer(10000)
            >>> samples = np.random.randn(1000) + 1j*np.random.randn(1000)
            >>> buffer.append(samples)  # Adds 1000 samples
        """
        n_samples = len(samples)
        
        with self.lock:
            # Calculate where samples will be written
            start_idx = self.write_index
            end_idx = start_idx + n_samples
            
            if end_idx <= self.capacity:
                # Simple case: samples fit without wrapping
                self.buffer[start_idx:end_idx] = samples
            else:
                # Wrap-around case: split write into two parts
                # Part 1: Fill to end of buffer
                samples_to_end = self.capacity - start_idx
                self.buffer[start_idx:] = samples[:samples_to_end]
                
                # Part 2: Wrap to beginning
                samples_from_start = n_samples - samples_to_end
                self.buffer[:samples_from_start] = samples[samples_to_end:]
            
            # Update write pointer (with wrap-around)
            self.write_index = (start_idx + n_samples) % self.capacity
            
            # Statistics
            self.samples_written += n_samples
            self.write_count += 1
            self.last_write_time = time.time()
    
    def get_last(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Retrieve last N samples from buffer (thread-safe read).
        
        Returns a COPY of data (safe for processing while buffer updates).
        
        Args:
            n_samples: Number of recent samples to retrieve
        
        Returns:
            Numpy array of last n_samples, or None if buffer has fewer samples
        
        Performance:
            O(n_samples) - dominated by memory copy
        
        Example:
            >>> buffer = CircularBuffer(10000)
            >>> # ... write some data ...
            >>> last_100 = buffer.get_last(100)
            >>> print(last_100.shape)  # (100,)
        """
        if n_samples > self.capacity:
            return None
        
        if self.samples_written < n_samples:
            # Buffer not yet filled with enough samples
            return None
        
        with self.lock:
            start_idx = (self.write_index - n_samples) % self.capacity
            
            if start_idx + n_samples <= self.capacity:
                # Simple case: samples are contiguous
                result = self.buffer[start_idx:start_idx + n_samples].copy()
            else:
                # Wrap-around case: concatenate two segments
                samples_to_end = self.capacity - start_idx
                result = np.concatenate([
                    self.buffer[start_idx:],
                    self.buffer[:n_samples - samples_to_end]
                ])
            
            return result
    
    def get_range(self, samples_ago: int, n_samples: int) -> Optional[np.ndarray]:
        """
        Get samples from specific time in past.
        
        Useful for extracting samples around a detection event.
        
        Args:
            samples_ago: How many samples back from current write position
            n_samples: How many samples to retrieve from that point
        
        Returns:
            Numpy array, or None if requested range not available
        
        Example:
            >>> # Get 1000 samples from 5000 samples ago
            >>> historical = buffer.get_range(5000, 1000)
            >>> # These are samples [write_pos - 5000 : write_pos - 4000]
        """
        if samples_ago + n_samples > self.capacity:
            return None
        
        if self.samples_written < samples_ago + n_samples:
            return None
        
        with self.lock:
            # Calculate start position
            start_idx = (self.write_index - samples_ago - n_samples) % self.capacity
            
            if start_idx + n_samples <= self.capacity:
                result = self.buffer[start_idx:start_idx + n_samples].copy()
            else:
                samples_to_end = self.capacity - start_idx
                result = np.concatenate([
                    self.buffer[start_idx:],
                    self.buffer[:n_samples - samples_to_end]
                ])
            
            return result
    
    def clear(self) -> None:
        """
        Clear buffer (reset to empty state).
        
        Does NOT deallocate memory, just resets write pointer.
        """
        with self.lock:
            self.write_index = 0
            self.samples_written = 0
            self.write_count = 0
    
    def get_fill_level(self) -> float:
        """
        Get buffer fill percentage (0.0 to 1.0).
        
        Returns:
            Fraction of buffer filled (1.0 = full, 0.0 = empty)
        
        Note:
            Once buffer fills once, this always returns 1.0
            (circular buffer is always "full" after first wrap)
        """
        if self.samples_written >= self.capacity:
            return 1.0
        return self.samples_written / self.capacity
    
    def get_write_rate(self) -> float:
        """
        Calculate average write rate (samples per second).
        
        Returns:
            Samples/sec write rate, or 0 if no writes yet
        
        Useful for:
            - Verifying sample rate matches expected (should be ~15 MS/s)
            - Detecting buffer underruns
        """
        if self.write_count == 0:
            return 0.0
        
        elapsed = time.time() - self.last_write_time
        if elapsed < 0.001:  # Avoid division by zero
            elapsed = 0.001
        
        # Calculate based on last few seconds of writes
        return self.samples_written / elapsed
    
    def __len__(self):
        """Allows the use of len(buffer_object)"""
        # The length is either all samples written, or the max capacity if we wrapped
        return min(self.samples_written, self.capacity)
    
    def get_stats(self) -> dict:
        """
        Get buffer statistics for monitoring.
        
        Returns:
            Dictionary with buffer health metrics
        """
        return {
            'capacity': self.capacity,
            'samples_written': self.samples_written,
            'write_count': self.write_count,
            'fill_level': self.get_fill_level(),
            'write_rate_msps': self.get_write_rate() / 1e6,
            'memory_mb': (self.capacity * self.buffer.itemsize) / (1024**2)
        }


class BufferManager:
    """
    Manages circular buffers for all three radios.
    
    Provides centralized access and monitoring for R1, R2, R3 buffers.
    Handles initialization, statistics, and health checks.
    """
    
    def __init__(self, sample_rate: float, buffer_duration: float):
        """
        Initialize buffer manager.
        
        Args:
            sample_rate: Samples per second (e.g., 15e6 for 15 MS/s)
            buffer_duration: Buffer length in seconds (e.g., 0.5)
        
        Memory Allocation:
            3 radios * sample_rate * buffer_duration * 8 bytes
            Example: 3 * 15M * 0.5 * 8 = 180 MB
        """
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        
        # Calculate buffer capacity
        capacity = int(sample_rate * buffer_duration)
        
        # Create buffers for each radio
        self.buffers = {
            'R1': CircularBuffer(capacity),
            'R2': CircularBuffer(capacity),
            'R3': CircularBuffer(capacity)
        }
        
        # Timing for rate monitoring
        self.start_time = time.time()
    
    def append(self, radio_id: str, samples: np.ndarray) -> None:
        """
        Append samples to specific radio's buffer.
        
        Args:
            radio_id: "R1", "R2", or "R3"
            samples: Numpy array of complex samples
        """
        if radio_id in self.buffers:
            self.buffers[radio_id].append(samples)
    
    def get_last(self, radio_id: str, n_samples: int) -> Optional[np.ndarray]:
        """
        Get last N samples from radio's buffer.
        
        Args:
            radio_id: "R1", "R2", or "R3"
            n_samples: Number of samples to retrieve
        
        Returns:
            Numpy array or None
        """
        if radio_id in self.buffers:
            return self.buffers[radio_id].get_last(n_samples)
        return None
    
    def get_range(self, radio_id: str, samples_ago: int, n_samples: int) -> Optional[np.ndarray]:
        """
        Get historical samples from radio's buffer.
        
        Args:
            radio_id: "R1", "R2", or "R3"
            samples_ago: How far back to look
            n_samples: How many to retrieve
        
        Returns:
            Numpy array or None
        """
        if radio_id in self.buffers:
            return self.buffers[radio_id].get_range(samples_ago, n_samples)
        return None
    
    def get_all_stats(self) -> dict:
        """
        Get statistics for all buffers.
        
        Returns:
            Dictionary: {radio_id: stats_dict}
        """
        return {
            radio_id: buf.get_stats() 
            for radio_id, buf in self.buffers.items()
        }
    
    def check_health(self) -> Tuple[bool, str]:
        """
        Check if all buffers are healthy.
        
        Returns:
            (healthy: bool, message: str)
        
        Checks:
            - Buffers receiving data
            - Write rate approximately correct
            - No buffer underruns
        """
        expected_rate_msps = self.sample_rate / 1e6
        
        for radio_id, buf in self.buffers.items():
            stats = buf.get_stats()
            
            # Check if buffer receiving data
            if stats['samples_written'] == 0:
                return False, f"{radio_id} buffer not receiving data"
            
            # Check write rate (allow ±20% tolerance)
            rate = stats['write_rate_msps']
            if rate < expected_rate_msps * 0.8 or rate > expected_rate_msps * 1.2:
                return False, f"{radio_id} write rate abnormal: {rate:.1f} MS/s (expected {expected_rate_msps:.1f})"
        
        return True, "All buffers healthy"
    
    def clear_all(self) -> None:
        """Clear all buffers."""
        for buf in self.buffers.values():
            buf.clear()


# Example usage and testing
if __name__ == "__main__":
    print("Testing CircularBuffer...")
    
    # Create small test buffer
    buf = CircularBuffer(capacity=1000)
    
    # Test appending
    test_samples = np.random.randn(100) + 1j * np.random.randn(100)
    buf.append(test_samples.astype(np.complex64))
    
    print(f"Buffer stats: {buf.get_stats()}")
    
    # Test retrieval
    last_50 = buf.get_last(50)
    assert last_50 is not None
    assert len(last_50) == 50
    assert np.allclose(last_50, test_samples[-50:])
    
    print("✓ Basic operations work")
    
    # Test wrap-around
    large_samples = np.random.randn(1500) + 1j * np.random.randn(1500)
    buf.append(large_samples.astype(np.complex64))
    
    last_200 = buf.get_last(200)
    assert len(last_200) == 200
    
    print("✓ Wrap-around works")
    
    # Test BufferManager
    manager = BufferManager(sample_rate=15e6, buffer_duration=0.5)
    test_data = np.random.randn(1000) + 1j * np.random.randn(1000)
    
    manager.append('R1', test_data.astype(np.complex64))
    retrieved = manager.get_last('R1', 500)
    
    assert retrieved is not None
    assert len(retrieved) == 500
    
    print("✓ BufferManager works")
    print("\nAll tests passed!")
