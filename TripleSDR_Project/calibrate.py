#!/usr/bin/env python3
"""
Professional AoA System Calibration Script

This script performs phase offset calibration for the Triple-SDR system.
Supports multiple calibration methods and frequencies for maximum accuracy.

Usage:
    python calibration.py

Features:
    - Multiple calibration geometries (touching antennas or boresight)
    - Single or multi-frequency calibration
    - Comprehensive validation and error checking
    - Safe configuration file handling with automatic backups
    - Real-time signal quality monitoring

Author: Triple-SDR Project
License: GPL-3.0
"""
import sys
import numpy as np
import zmq
import yaml
import time
import threading
import pmt
import shutil
from pathlib import Path
from datetime import datetime
from scipy.ndimage import uniform_filter1d
from colorama import init, Fore, Style

from core.buffers import BufferManager

# Initialize colorama for colored output
init(autoreset=True)


class CalibrationSession:
    """
    Manages a single calibration session with all safety checks.
    
    This class handles:
    - Sample capture from R1 and R3
    - Cross-correlation for alignment
    - Phase measurement with stability checking
    - Signal quality validation
    """
    
    def __init__(self, config, buffer_manager, zmq_context):
        """
        Initialize calibration session.
        
        Args:
            config: Loaded system configuration dictionary
            buffer_manager: BufferManager instance for sample storage
            zmq_context: ZMQ context for networking
        """
        self.config = config
        self.buffer_mgr = buffer_manager
        self.zmq_ctx = zmq_context
        self.sample_rate = config['hardware']['sample_rate']
        
        # Quality thresholds
        self.MIN_SIGNAL_POWER = -70  # dBm (weaker = unreliable)
        self.MIN_CORRELATION_CONFIDENCE = 2.0  # Peak/mean ratio
        self.MAX_PHASE_STD = 0.3  # radians (~17 degrees)
        
    def freeze_radios(self, target_freq_hz):
        """
        Command all three radios to tune to calibration frequency.
        
        This stops the leapfrog scanning and locks all radios on the
        calibration source for synchronized capture.
        
        Args:
            target_freq_hz: Frequency in Hz to tune to
        """
        print(f"{Fore.CYAN}[Tuning] Locking all radios to {target_freq_hz/1e6:.1f} MHz...{Style.RESET_ALL}")
        
        cmd_sock = self.zmq_ctx.socket(zmq.PUSH)
        cmd_sock.connect(f"tcp://127.0.0.1:{self.config['network']['zmq_cmd']}")
        
        # Tune all 3 channels (R1=0, R3=1, R2=2)
        for channel_idx in range(3):
            cmd_sock.send(pmt.serialize_str(pmt.intern(f"{channel_idx}:{target_freq_hz}")))
        
        cmd_sock.close()
        
        # Wait for PLL settling
        settling_time = 1.0  # seconds
        print(f"{Fore.CYAN}[Settling] Waiting {settling_time}s for PLL stabilization...{Style.RESET_ALL}")
        time.sleep(settling_time)
    
    def capture_samples(self, duration_sec=2.0):
        req_samples = int(self.sample_rate * duration_sec)
        # Clear buffers before starting to ensure we get FRESH data
        self.buffer_mgr.clear_all()
        
        print(f"{Fore.CYAN}[Capture] Starting background threads...{Style.RESET_ALL}")
        
        def capture_radio(radio_id, zmq_port):
            sock = self.zmq_ctx.socket(zmq.SUB)
            sock.connect(f"tcp://127.0.0.1:{zmq_port}")
            sock.setsockopt(zmq.SUBSCRIBE, b'')
            sock.setsockopt(zmq.RCVTIMEO, 500)
            
            # Record until the main thread tells us we have enough
            while len(self.buffer_mgr.buffers[radio_id]) < (req_samples + 100000):
                try:
                    payload = sock.recv()
                    samples = np.frombuffer(payload, dtype=np.complex64)
                    self.buffer_mgr.append(radio_id, samples)
                except zmq.Again:
                    if not self.running: break # Safety exit
                    continue
            sock.close()

        # Launch threads
        t1 = threading.Thread(target=capture_radio, args=('R1', self.config['network']['zmq_r1']), daemon=True)
        t3 = threading.Thread(target=capture_radio, args=('R3', self.config['network']['zmq_r3']), daemon=True)
        t1.start(); t3.start()

        # WAIT UNTIL READY (The "Wait-until-Ready" Logic)
        print(f"{Fore.YELLOW}[Wait] Waiting for buffers to fill (30M samples)...{Style.RESET_ALL}")
        start_wait = time.time()
        timeout = duration_sec + 5.0 # 5 second grace period for Windows latency
        
        while time.time() - start_wait < timeout:
            r1_len = len(self.buffer_mgr.buffers['R1'])
            r3_len = len(self.buffer_mgr.buffers['R3'])
            
            # Show progress so you aren't staring at a blank screen
            progress = min(r1_len, r3_len) / req_samples * 100
            print(f"\rProgress: {progress:.1f}% ({r1_len}/{req_samples})", end="")
            
            if r1_len >= req_samples and r3_len >= req_samples:
                print(f"\n{Fore.GREEN}[Ready] Buffers full.{Style.RESET_ALL}")
                break
            time.sleep(0.1)
        else:
            print(f"\n{Fore.RED}[Error] Timeout reached. R1:{len(self.buffer_mgr.buffers['R1'])} R3:{len(self.buffer_mgr.buffers['R3'])}{Style.RESET_ALL}")

        # Final retrieval
        samples_r1 = self.buffer_mgr.get_last('R1', req_samples)
        samples_r3 = self.buffer_mgr.get_last('R3', req_samples)
        return samples_r1, samples_r3
    
    def check_signal_quality(self, samples_r1, samples_r3):
        """
        Validate that captured signals are strong enough for calibration.
        
        Weak signals lead to noisy phase measurements and poor calibration.
        
        Args:
            samples_r1: R1 sample array
            samples_r3: R3 sample array
        
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        # Calculate received power (RSSI)
        power_r1 = 10 * np.log10(np.mean(np.abs(samples_r1)**2) + 1e-12)
        power_r3 = 10 * np.log10(np.mean(np.abs(samples_r3)**2) + 1e-12)
        
        print(f"{Fore.CYAN}[Quality] R1 power: {power_r1:.1f} dBm{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[Quality] R3 power: {power_r3:.1f} dBm{Style.RESET_ALL}")
        
        # Check if signals are strong enough
        if power_r1 < self.MIN_SIGNAL_POWER:
            return False, f"R1 signal too weak ({power_r1:.1f} dBm). Move source closer or increase gain."
        
        if power_r3 < self.MIN_SIGNAL_POWER:
            return False, f"R3 signal too weak ({power_r3:.1f} dBm). Check antenna connection."
        
        # Check power balance (should be similar for both antennas)
        power_diff = abs(power_r1 - power_r3)
        if power_diff > 15:
            print(f"{Fore.YELLOW}[Warning] Large power imbalance ({power_diff:.1f} dB). "
                  f"Check antenna connections.{Style.RESET_ALL}")
        
        return True, "Signal quality OK"
    
    def find_signal_window(self, samples, window_size=2000):
        """
        Locate the strongest signal region in captured samples.
        
        Uses moving average to find peak power region, avoiding
        hardcoded sample indices.
        
        Args:
            samples: Sample array to search
            window_size: Size of analysis window
        
        Returns:
            Start index of strongest window
        """
        # Calculate power profile
        power_profile = np.abs(samples) ** 2
        
        # Smooth with moving average (100-sample window)
        smoothed = uniform_filter1d(power_profile, size=100)
        
        # Find peak, avoiding edges
        margin = 1000
        search_region = smoothed[margin:-margin]
        peak_idx = np.argmax(search_region) + margin
        
        # Ensure window fits in buffer
        if peak_idx + window_size > len(samples):
            peak_idx = len(samples) - window_size - 1
        
        return peak_idx
    
    def measure_phase_offset(self, samples_r1, samples_r3):
        """
        Calculate phase offset using cross-correlation and vector averaging.
        
        This is the core calibration measurement with comprehensive
        error checking and stability validation.
        
        Process:
        1. Cross-correlate to find sample alignment
        2. Validate correlation confidence
        3. Locate strongest signal window
        4. Calculate phase difference with Hanning window
        5. Check phase stability across multiple windows
        6. Return averaged phase offset
        
        Args:
            samples_r1: R1 samples
            samples_r3: R3 samples
        
        Returns:
            Tuple of (success: bool, offset: float, info: dict)
        """
        print(f"{Fore.CYAN}[Processing] Calculating phase offset...{Style.RESET_ALL}")
        
        # Step 1: Cross-correlation for sample alignment
        print(f"{Fore.CYAN}[Step 1/5] Cross-correlating for alignment...{Style.RESET_ALL}")
        corr_window = 2000
        corr = np.correlate(
            samples_r1[:corr_window], 
            samples_r3[:corr_window], 
            mode='full'
        )
        
        # Find alignment offset
        corr_offset = int(np.argmax(np.abs(corr)) - corr_window)
        
        # Step 2: Validate correlation confidence
        peak = np.max(np.abs(corr))
        mean = np.mean(np.abs(corr))
        confidence = peak / mean
        
        print(f"{Fore.CYAN}[Step 2/5] Correlation confidence: {confidence:.2f} "
              f"(threshold: {self.MIN_CORRELATION_CONFIDENCE:.1f}){Style.RESET_ALL}")
        
        if confidence < self.MIN_CORRELATION_CONFIDENCE:
            return False, 0.0, {
                'error': f'Poor sample alignment (confidence={confidence:.2f})',
                'suggestion': 'Check 10MHz clock connection or reduce RF interference'
            }
        
        # Step 3: Find strongest signal window
        print(f"{Fore.CYAN}[Step 3/5] Locating signal peak...{Style.RESET_ALL}")
        window_size = 2000
        peak_idx = self.find_signal_window(samples_r1, window_size)
        
        # Ensure R3 window is within bounds
        if peak_idx + corr_offset + window_size > len(samples_r3):
            return False, 0.0, {
                'error': 'Signal window out of bounds',
                'suggestion': 'Increase capture duration or check for buffer overflow'
            }
        
        # Step 4: Calculate phase with Hanning window
        print(f"{Fore.CYAN}[Step 4/5] Calculating phase difference...{Style.RESET_ALL}")
        hanning_window = np.hanning(window_size)
        
        r1_window = samples_r1[peak_idx:peak_idx + window_size] * hanning_window
        r3_window = samples_r3[peak_idx + corr_offset:peak_idx + corr_offset + window_size] * hanning_window
        
        # Vector averaging for phase
        phase_offset = float(np.angle(np.mean(r1_window * np.conj(r3_window))))
        
        # Step 5: Check phase stability
        print(f"{Fore.CYAN}[Step 5/5] Validating phase stability...{Style.RESET_ALL}")
        n_windows = 5
        phase_samples = []
        
        for i in range(n_windows):
            offset = i * (window_size // 2)  # Overlapping windows
            if peak_idx + offset + window_size > len(samples_r1):
                break
            
            win_r1 = samples_r1[peak_idx + offset:peak_idx + offset + window_size] * hanning_window
            win_r3 = samples_r3[peak_idx + corr_offset + offset:
                               peak_idx + corr_offset + offset + window_size] * hanning_window
            
            phase_samples.append(np.angle(np.mean(win_r1 * np.conj(win_r3))))
        
        phase_std = np.std(phase_samples)
        print(f"{Fore.CYAN}[Stability] Phase std dev: {np.degrees(phase_std):.2f}° "
              f"(threshold: {np.degrees(self.MAX_PHASE_STD):.1f}°){Style.RESET_ALL}")
        
        if phase_std > self.MAX_PHASE_STD:
            print(f"{Fore.YELLOW}[Warning] Phase unstable - signal may be weak or "
                  f"multipath present{Style.RESET_ALL}")
        
        # Return results
        info = {
            'correlation_confidence': confidence,
            'correlation_offset': corr_offset,
            'phase_std_deg': np.degrees(phase_std),
            'phase_samples': phase_samples,
            'signal_window_idx': peak_idx
        }
        
        return True, phase_offset, info


def print_banner():
    """Display startup banner."""
    print("\n" + "="*70)
    print(Fore.CYAN + Style.BRIGHT + " PROFESSIONAL AoA CALIBRATION SYSTEM ".center(70, "=") + Style.RESET_ALL)
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Purpose: Measure phase offset for angle-of-arrival calculations")
    print("="*70 + "\n")


def select_calibration_mode():
    """
    Interactive menu for calibration mode selection.
    
    Returns:
        Tuple of (mode: str, description: str)
    """
    print(Fore.YELLOW + "CALIBRATION MODE SELECTION:" + Style.RESET_ALL)
    print("─" * 70)
    print(f"{Fore.GREEN}[1] Standard (RECOMMENDED){Style.RESET_ALL}")
    print("    • Place antennas TOUCHING (d ≈ 0)")
    print("    • Source can be anywhere with strong signal")
    print("    • Most foolproof method")
    print("    • Single frequency calibration")
    print()
    print(f"{Fore.GREEN}[2] Boresight{Style.RESET_ALL}")
    print("    • Source placed EXACTLY at 0° (boresight)")
    print("    • Requires precise alignment (laser recommended)")
    print("    • Single frequency calibration")
    print()
    print(f"{Fore.CYAN}[3] Multi-Frequency (BEST ACCURACY){Style.RESET_ALL}")
    print("    • Calibrates at 4 frequencies across 2.4 GHz band")
    print("    • Antennas touching OR boresight geometry")
    print("    • Accounts for frequency-dependent phase shifts")
    print("    • Recommended for production use")
    print("─" * 70)
    
    while True:
        choice = input(f"\n{Fore.YELLOW}Select mode [1/2/3]: {Style.RESET_ALL}").strip()
        
        if choice == "1":
            return "standard", "Antennas touching, single frequency"
        elif choice == "2":
            return "boresight", "Boresight alignment, single frequency"
        elif choice == "3":
            return "multi_freq", "Multi-frequency calibration"
        else:
            print(f"{Fore.RED}Invalid choice. Please enter 1, 2, or 3.{Style.RESET_ALL}")


def backup_config(config_path):
    """
    Create timestamped backup of configuration file.
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        Path to backup file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = Path(f"config_backup_{timestamp}.yaml")
    
    shutil.copy(config_path, backup_path)
    print(f"{Fore.GREEN}[Backup] Configuration backed up to {backup_path}{Style.RESET_ALL}")
    
    return backup_path


def save_calibration(config, offsets, mode_description):
    """
    Safely save calibration to configuration file.
    
    Uses atomic write to prevent corruption on crash.
    
    Args:
        config: Configuration dictionary
        offsets: Dictionary of {freq_hz: offset_radians}
        mode_description: Description of calibration mode used
    """
    config_path = Path("config.yaml")
    
    # Backup existing config
    backup_config(config_path)
    
    # Update configuration
    if len(offsets) == 1:
        # Single frequency - use scalar value
        freq, offset = list(offsets.items())[0]
        config['aoa']['calibration_offset'] = float(offset)
        config['aoa']['calibration_frequency'] = float(freq)
    else:
        # Multi-frequency - use table
        config['aoa']['calibration_table'] = {
            int(freq): float(offset) for freq, offset in offsets.items()
        }
        # Remove old scalar calibration if present
        if 'calibration_offset' in config['aoa']:
            del config['aoa']['calibration_offset']
    
    # Add metadata
    config['aoa']['calibration_timestamp'] = datetime.now().isoformat()
    config['aoa']['calibration_mode'] = mode_description
    config['aoa']['calibration_note'] = (
        "Calibration valid for ~6 hours or until temperature changes >10°C. "
        "Re-calibrate if AoA accuracy degrades."
    )
    
    # Atomic write (crash-safe)
    temp_path = Path("config_temp.yaml")
    with open(temp_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    temp_path.replace(config_path)
    
    print(f"{Fore.GREEN}[Saved] Configuration updated successfully{Style.RESET_ALL}")


def validate_offsets(offsets):
    """
    Sanity check calibration results.
    
    Args:
        offsets: Dictionary of {freq_hz: offset_radians}
    
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    print(f"\n{Fore.CYAN}[Validation] Checking calibration results...{Style.RESET_ALL}")
    
    # Check 1: Offsets in reasonable range
    for freq, offset in offsets.items():
        if abs(offset) > np.pi:
            return False, f"Offset at {freq/1e6:.1f}MHz out of range: {offset:.3f} rad"
    
    # Check 2: Multi-frequency smoothness
    if len(offsets) > 1:
        freqs = sorted(offsets.keys())
        vals = [offsets[f] for f in freqs]
        diffs = np.diff(vals)
        
        max_jump = np.max(np.abs(diffs))
        if max_jump > 0.5:  # >~30° jump
            print(f"{Fore.YELLOW}[Warning] Large phase jump detected: "
                  f"{np.degrees(max_jump):.1f}°{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}          This may indicate measurement error or cable issues{Style.RESET_ALL}")
            
            response = input(f"{Fore.YELLOW}Continue anyway? [y/N]: {Style.RESET_ALL}").strip().lower()
            return response == 'y', "User override on phase jump warning"
    
    print(f"{Fore.GREEN}[Validation] All checks passed ✓{Style.RESET_ALL}")
    return True, "Calibration valid"


def main():
    """Main calibration workflow."""
    print_banner()
    
    # Load configuration
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"{Fore.RED}[Error] config.yaml not found!{Style.RESET_ALL}")
        print("Please ensure you're running from the project root directory.")
        return
    except Exception as e:
        print(f"{Fore.RED}[Error] Failed to load config.yaml: {e}{Style.RESET_ALL}")
        return
    
    # Select calibration mode
    mode, mode_desc = select_calibration_mode()
    
    # Initialize systems
    print(f"\n{Fore.CYAN}[Init] Initializing calibration system...{Style.RESET_ALL}")
    sample_rate = config['hardware']['sample_rate']
    buffer_mgr = BufferManager(sample_rate, 3.0)  # 3 second buffer
    zmq_ctx = zmq.Context()
    
    session = CalibrationSession(config, buffer_mgr, zmq_ctx)
    
    # Determine calibration frequencies
    if mode == "multi_freq":
        print(f"\n{Fore.YELLOW}Multi-frequency calibration will measure at:{Style.RESET_ALL}")
        cal_freqs = [2412e6, 2437e6, 2462e6, 2484e6]
        for f in cal_freqs:
            print(f"  • {f/1e6:.0f} MHz")
        print(f"\n{Fore.GREEN}[Setup] Antennas: Place touching OR at boresight (0°){Style.RESET_ALL}")
        print(f"{Fore.GREEN}[Setup] Source: Strong WiFi signal (hotspot or router){Style.RESET_ALL}")
    else:
        if mode == "standard":
            print(f"\n{Fore.GREEN}[Setup] Place AoA antennas TOUCHING (d ≈ 0){Style.RESET_ALL}")
            print(f"{Fore.GREEN}[Setup] Source: Any strong 2.4 GHz signal nearby{Style.RESET_ALL}")
        else:  # boresight
            print(f"\n{Fore.YELLOW}[Setup] Place source EXACTLY at 0° boresight{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[Setup] Distance: 2-4 meters{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[Setup] Use laser pointer or alignment jig!{Style.RESET_ALL}")
        
        print("\nCommon WiFi channels:")
        print("  • 2412 MHz (Channel 1)")
        print("  • 2437 MHz (Channel 6)")
        print("  • 2462 MHz (Channel 11)")
        
        freq_input = input(f"\n{Fore.YELLOW}Enter signal frequency (MHz): {Style.RESET_ALL}").strip()
        try:
            cal_freqs = [float(freq_input) * 1e6]
        except ValueError:
            print(f"{Fore.RED}[Error] Invalid frequency{Style.RESET_ALL}")
            return
    
    # Perform calibration measurements
    offsets = {}
    
    for freq in cal_freqs:
        print(f"\n{'='*70}")
        print(f"{Fore.CYAN}CALIBRATING AT {freq/1e6:.1f} MHz{Style.RESET_ALL}")
        print(f"{'='*70}")
        
        input(f"{Fore.YELLOW}Press Enter when ready...{Style.RESET_ALL}")
        
        # Freeze radios on calibration frequency
        session.freeze_radios(freq)
        
        # Capture samples
        samples_r1, samples_r3 = session.capture_samples(duration_sec=2.0)
        
        if samples_r1 is None or samples_r3 is None:
            print(f"{Fore.RED}[Error] Sample capture failed{Style.RESET_ALL}")
            continue
        
        # Check signal quality
        is_valid, msg = session.check_signal_quality(samples_r1, samples_r3)
        if not is_valid:
            print(f"{Fore.RED}[Error] {msg}{Style.RESET_ALL}")
            retry = input(f"{Fore.YELLOW}Retry this frequency? [Y/n]: {Style.RESET_ALL}").strip().lower()
            if retry != 'n':
                continue
            else:
                print(f"{Fore.RED}Skipping {freq/1e6:.1f} MHz{Style.RESET_ALL}")
                continue
        
        # Measure phase offset
        success, offset, info = session.measure_phase_offset(samples_r1, samples_r3)
        
        if not success:
            print(f"{Fore.RED}[Error] {info['error']}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[Suggestion] {info['suggestion']}{Style.RESET_ALL}")
            retry = input(f"{Fore.YELLOW}Retry this frequency? [Y/n]: {Style.RESET_ALL}").strip().lower()
            if retry != 'n':
                continue
            else:
                print(f"{Fore.RED}Skipping {freq/1e6:.1f} MHz{Style.RESET_ALL}")
                continue
        
        # Store result
        offsets[freq] = offset
        
        print(f"\n{Fore.GREEN}[Result] Phase offset: {offset:.6f} radians ({np.degrees(offset):.2f}°){Style.RESET_ALL}")
        print(f"{Fore.GREEN}[Quality] Correlation: {info['correlation_confidence']:.2f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[Quality] Stability: ±{info['phase_std_deg']:.2f}°{Style.RESET_ALL}")
    
    # Check if we got any valid measurements
    if len(offsets) == 0:
        print(f"\n{Fore.RED}[Error] No successful measurements. Calibration failed.{Style.RESET_ALL}")
        return
    
    # Validate results
    is_valid, msg = validate_offsets(offsets)
    if not is_valid:
        print(f"\n{Fore.RED}[Error] Validation failed: {msg}{Style.RESET_ALL}")
        return
    
    # Display results summary
    print(f"\n{'='*70}")
    print(f"{Fore.GREEN}CALIBRATION COMPLETE{Style.RESET_ALL}")
    print(f"{'='*70}")
    print(f"\n{Fore.CYAN}Measured Offsets:{Style.RESET_ALL}")
    for freq, offset in sorted(offsets.items()):
        print(f"  {freq/1e6:7.1f} MHz: {offset:+.6f} rad ({np.degrees(offset):+7.2f}°)")
    
    # Save to configuration
    print()
    save_calibration(config, offsets, mode_desc)
    
    print(f"\n{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Calibration successful!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}Next steps:{Style.RESET_ALL}")
    print(f"  1. Run: python calibration_check.py (validate calibration)")
    print(f"  2. Start system: python main.py")
    print(f"\n{Fore.YELLOW}Note: Recalibrate if:{Style.RESET_ALL}")
    print(f"  • Temperature changes >10°C")
    print(f"  • More than 6 hours pass")
    print(f"  • AoA accuracy degrades")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}[Interrupted] Calibration cancelled by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}[Fatal Error] {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()