#!/usr/bin/env python3
"""
AoA Calibration Validation and Field Testing Script

This script allows you to verify calibration accuracy by measuring
AoA of a source at a known location. Compare measured angle to
expected angle to validate system performance.

Usage:
    python calibration_check.py

Features:
    - Measure AoA at known source positions
    - Calculate expected angle from geometry
    - Compare measured vs expected (validation)
    - Multiple measurement averaging
    - Detailed error analysis and diagnostics

Author: Triple-SDR Project
License: GPL-3.0
"""

import numpy as np
import zmq
import yaml
import time
import threading
import pmt
from pathlib import Path
from datetime import datetime
from colorama import init, Fore, Style
import sys

# Initialize colorama
init(autoreset=True)

# Import from project
from core.buffers import BufferManager


class AoAValidator:
    """
    Validates AoA calibration by comparing measured angles to ground truth.
    
    This class:
    - Captures samples from R1 and R3
    - Calculates AoA using calibrated phase offset
    - Compares to user-provided expected angle
    - Provides detailed error analysis
    """
    
    def __init__(self, config, buffer_manager, zmq_context):
        """
        Initialize validator.
        
        Args:
            config: Loaded configuration dictionary
            buffer_manager: BufferManager instance
            zmq_context: ZMQ context for networking
        """
        self.config = config
        self.buffer_mgr = buffer_manager
        self.zmq_ctx = zmq_context
        self.sample_rate = config['hardware']['sample_rate']
        
        # Physical constants
        self.SPEED_OF_LIGHT = 299792458.0  # m/s
        self.antenna_spacing = config['aoa']['antenna_spacing']
        
        # Load calibration
        self.calibration_offset = None
        self.calibration_table = None
        
        if 'calibration_table' in config['aoa']:
            # Multi-frequency calibration
            self.calibration_table = config['aoa']['calibration_table']
            print(f"{Fore.GREEN}[Calibration] Loaded multi-frequency table "
                  f"({len(self.calibration_table)} points){Style.RESET_ALL}")
        elif 'calibration_offset' in config['aoa']:
            # Single frequency calibration
            self.calibration_offset = config['aoa']['calibration_offset']
            print(f"{Fore.GREEN}[Calibration] Loaded single-frequency offset: "
                  f"{np.degrees(self.calibration_offset):.2f}°{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}[Error] No calibration found in config!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please run calibration.py first.{Style.RESET_ALL}")
            sys.exit(1)
    
    def get_calibration_offset(self, freq_hz):
        """
        Get calibration offset for specific frequency.
        
        If multi-frequency table exists, interpolates between points.
        Otherwise uses single global offset.
        
        Args:
            freq_hz: Target frequency
        
        Returns:
            Calibration offset in radians
        """
        if self.calibration_table is not None:
            # Interpolate from table
            freqs = sorted([float(f) for f in self.calibration_table.keys()])
            offsets = [float(self.calibration_table[f]) for f in freqs]
            
            offset = np.interp(freq_hz, freqs, offsets)
            return offset
        else:
            # Use global offset
            return self.calibration_offset
    
    def freeze_and_capture(self, target_freq_hz, duration_sec=2.0):
        """
        Tune radios and capture synchronized samples.
        
        Args:
            target_freq_hz: Frequency to measure at
            duration_sec: Capture duration
        
        Returns:
            Tuple of (samples_r1, samples_r3)
        """
        print(f"{Fore.CYAN}[Tuning] Locking radios to {target_freq_hz/1e6:.1f} MHz...{Style.RESET_ALL}")
        
        # Send tune commands
        cmd_sock = self.zmq_ctx.socket(zmq.PUSH)
        cmd_sock.connect(f"tcp://127.0.0.1:{self.config['network']['zmq_cmd']}")
        
        for channel_idx in range(3):
            cmd_sock.send(pmt.serialize_str(pmt.intern(f"{channel_idx}:{target_freq_hz}")))
        
        cmd_sock.close()
        time.sleep(1.0)  # PLL settling
        
        # Capture samples
        print(f"{Fore.CYAN}[Capture] Recording {duration_sec}s of samples...{Style.RESET_ALL}")
        
        def capture_radio(radio_id, zmq_port):
            sock = self.zmq_ctx.socket(zmq.SUB)
            sock.connect(f"tcp://127.0.0.1:{zmq_port}")
            sock.setsockopt(zmq.SUBSCRIBE, b'')
            sock.setsockopt(zmq.RCVTIMEO, 100)
            
            start_time = time.time()
            while time.time() - start_time < duration_sec:
                try:
                    payload = sock.recv()
                    samples = np.frombuffer(payload, dtype=np.complex64)
                    self.buffer_mgr.append(radio_id, samples)
                except zmq.Again:
                    continue
            sock.close()
        
        t1 = threading.Thread(target=capture_radio, args=('R1', self.config['network']['zmq_r1']))
        t3 = threading.Thread(target=capture_radio, args=('R3', self.config['network']['zmq_r3']))
        
        t1.start()
        t3.start()
        t1.join()
        t3.join()
        
        # Retrieve samples
        samples_r1 = self.buffer_mgr.get_last('R1', int(self.sample_rate * duration_sec))
        samples_r3 = self.buffer_mgr.get_last('R3', int(self.sample_rate * duration_sec))
        
        return samples_r1, samples_r3
    
    def calculate_aoa(self, samples_r1, samples_r3, freq_hz):
        """
        Calculate angle of arrival from samples.
        
        Uses the same algorithm as the main system for consistency.
        
        Args:
            samples_r1: R1 samples
            samples_r3: R3 samples
            freq_hz: Measurement frequency
        
        Returns:
            Dictionary with results:
            {
                'success': bool,
                'angle_deg': float,
                'phase_deg': float,
                'confidence': float,
                'error': str (if failed)
            }
        """
        # Cross-correlation for alignment
        corr_window = 2000
        corr = np.correlate(samples_r1[:corr_window], samples_r3[:corr_window], mode='full')
        offset = int(np.argmax(np.abs(corr)) - corr_window)
        
        # Check correlation quality
        peak = np.max(np.abs(corr))
        mean = np.mean(np.abs(corr))
        confidence = peak / mean
        
        if confidence < 4.0:
            return {
                'success': False,
                'error': f'Poor correlation (confidence={confidence:.2f})',
                'confidence': confidence
            }
        
        # Extract aligned windows
        window_size = 2000
        hanning = np.hanning(window_size)
        
        # Find signal peak
        power = np.abs(samples_r1) ** 2
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(power, size=100)
        peak_idx = np.argmax(smoothed[1000:-1000]) + 1000
        
        # Extract windows
        r1_win = samples_r1[peak_idx:peak_idx + window_size] * hanning
        r3_win = samples_r3[peak_idx + offset:peak_idx + offset + window_size] * hanning
        
        # Calculate raw phase difference
        phase_raw = np.angle(np.mean(r1_win * np.conj(r3_win)))
        
        # Apply calibration correction
        cal_offset = self.get_calibration_offset(freq_hz)
        phase_calibrated = phase_raw - cal_offset
        
        # Normalize to [-π, π]
        phase_calibrated = np.arctan2(np.sin(phase_calibrated), np.cos(phase_calibrated))
        
        # Calculate wavelength
        wavelength = self.SPEED_OF_LIGHT / freq_hz
        
        # Check antenna spacing validity
        if self.antenna_spacing > (wavelength / 2):
            return {
                'success': False,
                'error': f'Antenna spacing ({self.antenna_spacing*100:.1f}cm) too large for '
                        f'{freq_hz/1e6:.1f}MHz (max {wavelength/2*100:.1f}cm)'
            }
        
        # Calculate angle using phase interferometry
        arg = (phase_calibrated * wavelength) / (2 * np.pi * self.antenna_spacing)
        
        if abs(arg) > 1.0:
            return {
                'success': False,
                'error': f'Arcsin domain error (arg={arg:.3f}). Check calibration or signal quality.'
            }
        
        angle_rad = np.arcsin(arg)
        angle_deg = np.degrees(angle_rad)
        
        # Signal quality metrics
        signal_power = 10 * np.log10(np.mean(np.abs(r1_win)**2) + 1e-12)
        
        return {
            'success': True,
            'angle_deg': angle_deg,
            'phase_deg': np.degrees(phase_calibrated),
            'phase_raw_deg': np.degrees(phase_raw),
            'confidence': confidence,
            'signal_power_dbm': signal_power,
            'wavelength_cm': wavelength * 100,
            'correlation_offset': offset
        }
    
    def multiple_measurements(self, freq_hz, n_measurements=5):
        """
        Take multiple AoA measurements and return statistics.
        
        Args:
            freq_hz: Measurement frequency
            n_measurements: Number of measurements to average
        
        Returns:
            Dictionary with statistics
        """
        print(f"\n{Fore.CYAN}Taking {n_measurements} measurements...{Style.RESET_ALL}")
        
        angles = []
        all_results = []
        
        for i in range(n_measurements):
            print(f"{Fore.CYAN}[{i+1}/{n_measurements}] Measuring...{Style.RESET_ALL}", end=' ')
            
            samples_r1, samples_r3 = self.freeze_and_capture(freq_hz, duration_sec=1.5)
            
            if samples_r1 is None or samples_r3 is None:
                print(f"{Fore.RED}FAILED (no samples){Style.RESET_ALL}")
                continue
            
            result = self.calculate_aoa(samples_r1, samples_r3, freq_hz)
            
            if result['success']:
                angles.append(result['angle_deg'])
                all_results.append(result)
                print(f"{Fore.GREEN}OK ({result['angle_deg']:+.2f}°){Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}FAILED ({result['error']}){Style.RESET_ALL}")
            
            time.sleep(0.5)  # Brief pause between measurements
        
        if len(angles) == 0:
            return {'success': False, 'error': 'All measurements failed'}
        
        # Calculate statistics
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        median_angle = np.median(angles)
        
        return {
            'success': True,
            'measurements': angles,
            'mean_deg': mean_angle,
            'median_deg': median_angle,
            'std_dev_deg': std_angle,
            'min_deg': np.min(angles),
            'max_deg': np.max(angles),
            'all_results': all_results,
            'n_successful': len(angles),
            'n_total': n_measurements
        }


def calculate_expected_angle(distance_m, offset_m):
    """
    Calculate expected angle from geometric setup.
    
    Geometry:
          Source (phone/router)
             ↓
          ┌──┴──┐
          │  d  │ offset_m
          └──┬──┘
    ════════╪════════ Antenna axis
         distance_m
    
    θ = arctan(offset / distance)
    
    Args:
        distance_m: Distance from antenna to source (perpendicular)
        offset_m: Lateral offset from center
    
    Returns:
        Expected angle in degrees
    """
    if distance_m <= 0:
        return 0.0
    
    angle_rad = np.arctan(offset_m / distance_m)
    return np.degrees(angle_rad)


def print_banner():
    """Display startup banner."""
    print("\n" + "="*70)
    print(Fore.CYAN + Style.BRIGHT + " AoA CALIBRATION VALIDATION ".center(70, "=") + Style.RESET_ALL)
    print("="*70)
    print("Purpose: Verify AoA accuracy with known source positions")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def main():
    """Main validation workflow."""
    print_banner()
    
    # Load configuration
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"{Fore.RED}[Error] config.yaml not found!{Style.RESET_ALL}")
        return
    except Exception as e:
        print(f"{Fore.RED}[Error] Failed to load config: {e}{Style.RESET_ALL}")
        return
    
    # Check if calibration exists
    if 'calibration_offset' not in config['aoa'] and 'calibration_table' not in config['aoa']:
        print(f"{Fore.RED}[Error] No calibration found in config!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please run: python calibration.py{Style.RESET_ALL}")
        return
    
    # Display calibration info
    if 'calibration_timestamp' in config['aoa']:
        cal_time = datetime.fromisoformat(config['aoa']['calibration_timestamp'])
        age = datetime.now() - cal_time
        print(f"{Fore.CYAN}[Info] Calibration age: {age.total_seconds()/3600:.1f} hours{Style.RESET_ALL}")
        
        if age.total_seconds() > 21600:  # 6 hours
            print(f"{Fore.YELLOW}[Warning] Calibration is old. Consider recalibrating.{Style.RESET_ALL}")
    
    # Initialize validator
    print(f"\n{Fore.CYAN}[Init] Initializing validation system...{Style.RESET_ALL}")
    sample_rate = config['hardware']['sample_rate']
    buffer_mgr = BufferManager(sample_rate, 3.0)
    zmq_ctx = zmq.Context()
    
    validator = AoAValidator(config, buffer_mgr, zmq_ctx)
    
    # Get test parameters
    print(f"\n{Fore.YELLOW}TEST SETUP:{Style.RESET_ALL}")
    print("─" * 70)
    print("Place your signal source (phone hotspot or WiFi router) at a")
    print("KNOWN location relative to your antenna array.")
    print()
    print("Example setup:")
    print("  • Distance: 3.0 meters perpendicular to antenna axis")
    print("  • Offset: -0.5 meters (left of center)")
    print("  • Expected angle: arctan(-0.5/3.0) = -9.46°")
    print("─" * 70)
    
    # Frequency selection
    print("\nCommon WiFi channels:")
    print("  • 2412 MHz (Channel 1)")
    print("  • 2437 MHz (Channel 6)")
    print("  • 2462 MHz (Channel 11)")
    
    try:
        freq_mhz = float(input(f"\n{Fore.YELLOW}Signal frequency (MHz): {Style.RESET_ALL}").strip())
        freq_hz = freq_mhz * 1e6
    except ValueError:
        print(f"{Fore.RED}[Error] Invalid frequency{Style.RESET_ALL}")
        return
    
    # Geometry input
    print(f"\n{Fore.YELLOW}Enter measurement geometry:{Style.RESET_ALL}")
    try:
        distance = float(input("  Distance to source (meters): ").strip())
        offset = float(input("  Lateral offset from center (meters, -=left, +=right): ").strip())
    except ValueError:
        print(f"{Fore.RED}[Error] Invalid geometry{Style.RESET_ALL}")
        return
    
    # Calculate expected angle
    expected_angle = calculate_expected_angle(distance, offset)
    
    print(f"\n{Fore.CYAN}[Geometry] Expected angle: {expected_angle:+.2f}°{Style.RESET_ALL}")
    
    # Number of measurements
    n_meas = 5
    try:
        n_input = input(f"\nNumber of measurements [default: 5]: ").strip()
        if n_input:
            n_meas = int(n_input)
    except ValueError:
        print(f"{Fore.YELLOW}Using default: 5 measurements{Style.RESET_ALL}")
    
    # Ready check
    print(f"\n{Fore.GREEN}[Ready] Source at {expected_angle:+.2f}° from boresight{Style.RESET_ALL}")
    input(f"{Fore.YELLOW}Press Enter to start measurements...{Style.RESET_ALL}")
    
    # Perform measurements
    stats = validator.multiple_measurements(freq_hz, n_measurements=n_meas)
    
    if not stats['success']:
        print(f"\n{Fore.RED}[Error] Measurement failed: {stats['error']}{Style.RESET_ALL}")
        return
    
    # Display results
    print(f"\n{'='*70}")
    print(f"{Fore.GREEN}MEASUREMENT RESULTS{Style.RESET_ALL}")
    print(f"{'='*70}")
    
    print(f"\n{Fore.CYAN}Individual Measurements:{Style.RESET_ALL}")
    for i, angle in enumerate(stats['measurements'], 1):
        error = angle - expected_angle
        print(f"  [{i}] {angle:+7.2f}° (error: {error:+.2f}°)")
    
    print(f"\n{Fore.CYAN}Statistics:{Style.RESET_ALL}")
    print(f"  Mean:        {stats['mean_deg']:+7.2f}°")
    print(f"  Median:      {stats['median_deg']:+7.2f}°")
    print(f"  Std Dev:     {stats['std_dev_deg']:7.2f}°")
    print(f"  Range:       {stats['min_deg']:+7.2f}° to {stats['max_deg']:+7.2f}°")
    print(f"  Success:     {stats['n_successful']}/{stats['n_total']}")
    
    print(f"\n{Fore.CYAN}Validation:{Style.RESET_ALL}")
    print(f"  Expected:    {expected_angle:+7.2f}°")
    print(f"  Measured:    {stats['mean_deg']:+7.2f}°")
    
    error = stats['mean_deg'] - expected_angle
    print(f"  Error:       {error:+7.2f}°")
    
    # Error assessment
    print(f"\n{Fore.CYAN}Assessment:{Style.RESET_ALL}")
    
    if abs(error) < 3:
        print(f"  {Fore.GREEN}✓ EXCELLENT - Error < 3° (within specification){Style.RESET_ALL}")
    elif abs(error) < 5:
        print(f"  {Fore.GREEN}✓ GOOD - Error < 5° (acceptable){Style.RESET_ALL}")
    elif abs(error) < 10:
        print(f"  {Fore.YELLOW}⚠ MARGINAL - Error < 10° (consider recalibration){Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}✗ POOR - Error > 10° (recalibration required){Style.RESET_ALL}")
    
    if stats['std_dev_deg'] < 2:
        print(f"  {Fore.GREEN}✓ High stability (std dev < 2°){Style.RESET_ALL}")
    elif stats['std_dev_deg'] < 5:
        print(f"  {Fore.YELLOW}⚠ Moderate stability (std dev < 5°){Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}✗ Poor stability (std dev > 5°){Style.RESET_ALL}")
        print(f"     Possible causes: weak signal, multipath, interference")
    
    # Detailed diagnostics
    if stats['all_results']:
        avg_confidence = np.mean([r['confidence'] for r in stats['all_results']])
        avg_power = np.mean([r['signal_power_dbm'] for r in stats['all_results']])
        
        print(f"\n{Fore.CYAN}Diagnostics:{Style.RESET_ALL}")
        print(f"  Avg Correlation Confidence: {avg_confidence:.2f}")
        print(f"  Avg Signal Power:           {avg_power:.1f} dBm")
        
        if avg_confidence < 5:
            print(f"  {Fore.YELLOW}⚠ Low correlation confidence - check 10MHz clock{Style.RESET_ALL}")
        
        if avg_power < -60:
            print(f"  {Fore.YELLOW}⚠ Weak signal - move source closer or increase gain{Style.RESET_ALL}")
    
    # Save results
    save = input(f"\n{Fore.YELLOW}Save results to log? [Y/n]: {Style.RESET_ALL}").strip().lower()
    if save != 'n':
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(log_file, 'w') as f:
            f.write("AoA CALIBRATION VALIDATION RESULTS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Frequency: {freq_hz/1e6:.1f} MHz\n")
            f.write(f"Expected Angle: {expected_angle:+.2f}°\n")
            f.write(f"Measured Angle: {stats['mean_deg']:+.2f}°\n")
            f.write(f"Error: {error:+.2f}°\n")
            f.write(f"Std Dev: {stats['std_dev_deg']:.2f}°\n")
            f.write(f"\nIndividual Measurements:\n")
            for i, angle in enumerate(stats['measurements'], 1):
                f.write(f"  [{i}] {angle:+.2f}°\n")
        
        print(f"{Fore.GREEN}[Saved] Results written to {log_file}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Validation complete!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
    
    # Recommendations
    if abs(error) > 5 or stats['std_dev_deg'] > 5:
        print(f"\n{Fore.YELLOW}RECOMMENDATIONS:{Style.RESET_ALL}")
        if abs(error) > 5:
            print("  • Re-run calibration.py (systematic error detected)")
        if stats['std_dev_deg'] > 5:
            print("  • Check for multipath (indoor reflections)")
            print("  • Verify 10MHz clock connection (phase drift)")
            print("  • Move source to reduce interference")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}[Interrupted] Validation cancelled{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}[Fatal Error] {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()