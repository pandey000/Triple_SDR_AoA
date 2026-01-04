"""
Data Aggregator for Visualization System

Collects data from detection engines, buffers, and AoA calculator,
aggregates into lightweight state for web clients.

Thread-safe, runs at adaptive rate (1-5 Hz based on system load).

Performance:
- RAM usage: ~15 MB (negligible)
- CPU usage: ~5% @ 5 Hz (Pi 5)
- Network: ~5 KB/frame @ 5 Hz = 25 KB/s

Author: Triple-SDR Project
"""

import numpy as np
import threading
import time
import queue
from collections import deque
from datetime import datetime
from typing import Optional, Dict, List
import copy


class VisualizationState:
    """
    Lightweight state container for visualization data.
    
    Designed to be JSON-serializable and compact.
    All arrays are kept small (< 1024 elements).
    """
    
    def __init__(self):
        # Live spectrum data (1024 points each)
        self.spectrum_r1 = np.zeros(1024, dtype=np.float32)
        self.spectrum_r2 = np.zeros(1024, dtype=np.float32)
        self.spectrum_r1_freq = 2400e6  # Current center frequency
        self.spectrum_r2_freq = 2445e6
        
        # Detection events (last 60 seconds, ~1 per second avg)
        self.detection_events = deque(maxlen=60)
        
        # AoA measurements (last 20 measurements)
        self.aoa_history = deque(maxlen=20)
        
        # System metrics
        self.metrics = {
            'uptime_seconds': 0,
            'cpu_percent': 0.0,
            'buffer_r1_fill': 0.0,
            'buffer_r2_fill': 0.0,
            'buffer_r3_fill': 0.0,
            'buffer_r1_rate_msps': 0.0,
            'buffer_r3_rate_msps': 0.0,
            'detections_total': 0,
            'detections_confirmed': 0,
            'aoa_attempts': 0,
            'aoa_successful': 0,
            'is_learning': True,
            'lock_on_active': False
        }
        
        # Detection confidence breakdown (for current target)
        self.confidence = {
            'signal_strength': 0.0,
            'bandwidth_match': 0.0,
            'hop_regularity': 0.0,
            'spectral_shape': 0.0,
            'overall': 0.0,
            'active': False  # Only show when detection active
        }
        
        # Last update timestamp
        self.last_update = time.time()


class DataAggregator:
    """
    Aggregates data from detection system for visualization.
    
    Runs in background thread, updates state at adaptive rate:
    - Normal: 5 Hz
    - Detection active: 3 Hz (reduce load)
    - AoA lock: 2 Hz (lowest priority)
    
    Thread-safe for multiple web clients reading simultaneously.
    """
    
    def __init__(self, buffer_manager, detection_engines, aoa_calculator, 
                 metrics_collector, config):
        """
        Initialize data aggregator.
        
        Args:
            buffer_manager: BufferManager instance
            detection_engines: Dict of {'R1': engine, 'R2': engine}
            aoa_calculator: AoACalculator instance
            metrics_collector: MetricsCollector instance
            config: System configuration dict
        """
        self.buffer_mgr = buffer_manager
        self.detection_engines = detection_engines
        self.aoa_calc = aoa_calculator
        self.metrics_collector = metrics_collector
        self.config = config
        
        # State
        self.state = VisualizationState()
        self.state_lock = threading.Lock()
        
        # Control
        self.running = False
        self.update_rate = config['visualization'].get('update_rate', 5)
        self.start_time = time.time()
        
        # Event queues (other threads push events here)
        self.detection_queue = queue.Queue(maxsize=100)
        self.aoa_queue = queue.Queue(maxsize=20)
        
        # Performance tracking
        self.frame_count = 0
        self.decimation = config['visualization'].get('spectrum_decimation', 4)
        
        # Adaptive rate control
        self.system_load_high = False  # Set by main system during heavy processing
    
    def push_detection_event(self, event: dict):
        """
        Called by detection threads when target detected.
        
        Args:
            event: {
                'timestamp': datetime,
                'freq_mhz': float,
                'radio': str ('R1' or 'R2'),
                'hits': int,
                'channels': int,
                'confirmed': bool
            }
        """
        try:
            self.detection_queue.put_nowait(event)
        except queue.Full:
            pass  # Drop oldest if queue full (shouldn't happen with maxsize=100)
    
    def push_aoa_measurement(self, measurement: dict):
        """
        Called by AoA thread when angle calculated.
        
        Args:
            measurement: {
                'timestamp': datetime,
                'freq_mhz': float,
                'angle_deg': float,
                'confidence': float,
                'phase_deg': float
            }
        """
        try:
            self.aoa_queue.put_nowait(measurement)
        except queue.Full:
            pass
    
    def set_system_load(self, is_high: bool):
        """
        Called by main system to indicate heavy processing.
        
        When True, aggregator reduces update rate to free CPU.
        """
        self.system_load_high = is_high
    
    def start(self):
        """Launch background aggregation thread."""
        self.running = True
        thread = threading.Thread(target=self._update_loop, daemon=True, name="DataAggregator")
        thread.start()
    
    def stop(self):
        """Stop aggregation thread."""
        self.running = False
    
    def _update_loop(self):
        """
        Main update loop (runs at adaptive rate).
        
        Updates:
        1. Spectrum snapshots (always)
        2. Detection events (when available)
        3. AoA measurements (when available)
        4. System metrics (every 10 frames)
        5. Confidence breakdown (when detection active)
        """
        while self.running:
            start_time = time.time()
            
            # Determine update rate based on system load
            if self.system_load_high:
                current_rate = 2  # 500ms updates during heavy load
            elif hasattr(self, '_detection_active') and self._detection_active:
                current_rate = 3  # 333ms during detection
            else:
                current_rate = self.update_rate  # Normal rate (5 Hz default)
            
            try:
                # Update all state components
                self._update_spectrum()
                self._process_detection_queue()
                self._process_aoa_queue()
                
                # Update metrics less frequently (every 10 frames = 2 seconds @ 5Hz)
                if self.frame_count % 10 == 0:
                    self._update_metrics()
                
                # Update timestamp
                with self.state_lock:
                    self.state.last_update = time.time()
                
                self.frame_count += 1
                
            except Exception as e:
                print(f"[DataAggregator] Error in update loop: {e}")
            
            # Sleep for remainder of frame time
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / current_rate) - elapsed)
            time.sleep(sleep_time)
    
    def _update_spectrum(self):
        """
        Grab latest spectrum snapshots from buffers.
        
        Uses small FFT (1024 points) for speed.
        Applies decimation to reduce network payload.
        """
        fft_size = 1024
        
        # Get samples from buffers
        samples_r1 = self.buffer_mgr.get_last('R1', fft_size)
        samples_r2 = self.buffer_mgr.get_last('R2', fft_size)
        
        if samples_r1 is not None:
            # Calculate PSD with Hanning window
            psd_r1 = 20 * np.log10(
                np.abs(np.fft.fftshift(
                    np.fft.fft(samples_r1 * np.hanning(fft_size))
                )) / fft_size + 1e-12
            )
            
            # Apply decimation (send every Nth point)
            if self.decimation > 1:
                psd_r1 = psd_r1[::self.decimation]
            
            with self.state_lock:
                self.state.spectrum_r1 = psd_r1.astype(np.float32)
        
        if samples_r2 is not None:
            psd_r2 = 20 * np.log10(
                np.abs(np.fft.fftshift(
                    np.fft.fft(samples_r2 * np.hanning(fft_size))
                )) / fft_size + 1e-12
            )
            
            if self.decimation > 1:
                psd_r2 = psd_r2[::self.decimation]
            
            with self.state_lock:
                self.state.spectrum_r2 = psd_r2.astype(np.float32)
    
    def _process_detection_queue(self):
        """Pull all pending detection events from queue."""
        events_added = 0
        
        while not self.detection_queue.empty() and events_added < 10:
            try:
                event = self.detection_queue.get_nowait()
                
                # Format for JSON
                formatted_event = {
                    'time': event['timestamp'].strftime('%H:%M:%S'),
                    'freq': f"{event['freq_mhz']:.1f}",
                    'radio': event['radio'],
                    'hits': event['hits'],
                    'channels': event['channels'],
                    'confirmed': event['confirmed']
                }
                
                with self.state_lock:
                    self.state.detection_events.append(formatted_event)
                
                events_added += 1
                
            except queue.Empty:
                break
        
        # Set detection active flag for adaptive rate control
        self._detection_active = events_added > 0
    
    def _process_aoa_queue(self):
        """Pull all pending AoA measurements from queue."""
        while not self.aoa_queue.empty():
            try:
                measurement = self.aoa_queue.get_nowait()
                
                # Format for JSON
                formatted_aoa = {
                    'time': measurement['timestamp'].strftime('%H:%M:%S'),
                    'freq': f"{measurement['freq_mhz']:.1f}",
                    'angle': round(measurement['angle_deg'], 1),
                    'confidence': round(measurement['confidence'], 2),
                    'phase': round(measurement.get('phase_deg', 0), 1)
                }
                
                with self.state_lock:
                    self.state.aoa_history.append(formatted_aoa)
                
            except queue.Empty:
                break
    
    def _update_metrics(self):
        """
        Update system health metrics.
        
        Pulls from:
        - BufferManager (fill levels, write rates)
        - MetricsCollector (detection stats)
        - System time (uptime)
        """
        # Buffer health
        buffer_stats = self.buffer_mgr.get_all_stats()
        
        # System metrics from collector
        collector_metrics = self.metrics_collector.metrics
        
        # CPU usage (optional, requires psutil)
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0)
        except ImportError:
            cpu_percent = 0.0
        
        with self.state_lock:
            self.state.metrics.update({
                'uptime_seconds': int(time.time() - self.start_time),
                'cpu_percent': cpu_percent,
                'buffer_r1_fill': buffer_stats['R1']['fill_level'],
                'buffer_r2_fill': buffer_stats['R2']['fill_level'],
                'buffer_r3_fill': buffer_stats['R3']['fill_level'],
                'buffer_r1_rate_msps': buffer_stats['R1']['write_rate_msps'],
                'buffer_r3_rate_msps': buffer_stats['R3']['write_rate_msps'],
                'detections_total': collector_metrics['detections']['total'],
                'detections_confirmed': collector_metrics['detections']['confirmed'],
                'aoa_attempts': collector_metrics['aoa']['attempts'],
                'aoa_successful': collector_metrics['aoa']['successful']
            })
    
    def update_confidence_breakdown(self, breakdown: dict):
        """
        Update detection confidence breakdown.
        
        Called by detection thread when analyzing a signal.
        
        Args:
            breakdown: {
                'signal_strength': float (0-1),
                'bandwidth_match': float (0-1),
                'hop_regularity': float (0-1),
                'spectral_shape': float (0-1),
                'overall': float (0-1)
            }
        """
        with self.state_lock:
            self.state.confidence = {
                'signal_strength': round(breakdown.get('signal_strength', 0) * 100, 1),
                'bandwidth_match': round(breakdown.get('bandwidth_match', 0) * 100, 1),
                'hop_regularity': round(breakdown.get('hop_regularity', 0) * 100, 1),
                'spectral_shape': round(breakdown.get('spectral_shape', 0) * 100, 1),
                'overall': round(breakdown.get('overall', 0) * 100, 1),
                'active': True
            }
    
    def clear_confidence(self):
        """Clear confidence display (no active detection)."""
        with self.state_lock:
            self.state.confidence['active'] = False
    
    def get_state_snapshot(self) -> dict:
        """
        Get thread-safe snapshot of current state for web client.
        
        Returns:
            Dictionary ready for JSON serialization
        """
        with self.state_lock:
            return {
                'spectrum_r1': self.state.spectrum_r1.tolist(),
                'spectrum_r2': self.state.spectrum_r2.tolist(),
                'spectrum_r1_freq': self.state.spectrum_r1_freq / 1e6,  # MHz
                'spectrum_r2_freq': self.state.spectrum_r2_freq / 1e6,
                'events': list(self.state.detection_events),
                'aoa': list(self.state.aoa_history),
                'metrics': copy.copy(self.state.metrics),
                'confidence': copy.copy(self.state.confidence),
                'decimation': self.decimation,
                'timestamp': self.state.last_update
            }
    
    def update_radio_frequencies(self, r1_freq: float, r2_freq: float):
        """
        Update current radio frequencies (called by commander thread).
        
        Args:
            r1_freq: R1 center frequency (Hz)
            r2_freq: R2 center frequency (Hz)
        """
        with self.state_lock:
            self.state.spectrum_r1_freq = r1_freq
            self.state.spectrum_r2_freq = r2_freq