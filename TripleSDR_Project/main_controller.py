#!/usr/bin/env python3
import sys
import time
import threading
import signal
import yaml
import logging
from pathlib import Path
from datetime import datetime
import socket
import numpy as np
import zmq
import pmt

# Import the modules we built in previous steps
from core.buffers import BufferManager
from core.detection import DetectionEngine, PersistenceRegistry
from core.aoa import AoACalculator
from filtering.temporal import TemporalFilter
from filtering.spectral import SpectralFilter
from utils.logger import setup_logging
from utils.metrics import MetricsCollector

# Visualization imports
from visualization.data_aggregator import DataAggregator
from visualization.web_server import start_web_server

class TripleSDRSystem:
    def __init__(self, config_path: str = "config.yaml"):
        # 1. Load Configuration 
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 2. Setup Logging 
        self.logger = setup_logging(
            log_dir=self.config['logging'].get('log_dir', 'logs'),
            console_level='INFO'
        )
        
        # 3. Initialize State 
        self.running = True
        self.is_learning = True
        self.lock_on_active = False
        self.zmq_ctx = zmq.Context()
        
        # 4. Initialize Subsystems 
        sample_rate = self.config['hardware']['sample_rate']
        self.buffer_manager = BufferManager(sample_rate, self.config['performance'].get('buffer_size', 1.0))
        self.aoa_calculator = AoACalculator(self.config['aoa'], sample_rate)
        self.metrics = MetricsCollector()
        
        self.detection_engines = {
            'R1': DetectionEngine(self.config, 'R1'),
            'R2': DetectionEngine(self.config, 'R2')
        }
        
        self.registry = PersistenceRegistry(
            threshold=self.config['detection']['persistence_threshold'],
            window=self.config['detection']['window_seconds'],
            min_channels=self.config['detection']['min_unique_channels']
        )
        
        self.temporal_filter = TemporalFilter(self.config)
        self.spectral_filter = SpectralFilter(self.config)

        self.radio_states = {
            'R1': {'freq': 2400e6, 'settling': False},
            'R2': {'freq': 2445e6, 'settling': False}
        }

        # NEW: Initialize visualization data aggregator
        self.data_aggregator = None
        if self.config['visualization'].get('enabled', False):
            self.data_aggregator = DataAggregator(
                buffer_manager=self.buffer_manager,
                detection_engines=self.detection_engines,
                aoa_calculator=self.aoa_calculator,
                metrics_collector=self.metrics,
                config=self.config
            )
            self.logger.info("[Init] Visualization data aggregator initialized")

    def _zmq_buffer_thread(self, radio_id):
        """Thread to pull IQ data into the CircularBuffer """
        port = self.config['network'][f'zmq_{radio_id.lower()}']
        sock = self.zmq_ctx.socket(zmq.SUB)
        sock.connect(f"tcp://127.0.0.1:{port}")
        sock.setsockopt(zmq.SUBSCRIBE, b'')
        sock.setsockopt(zmq.RCVHWM, 100) # [NEW] Prevent RAM overflow
        
        while self.running:
            try:
                payload = sock.recv(flags=zmq.NOBLOCK)
                samples = np.frombuffer(payload, dtype=np.complex64)
                self.buffer_manager.append(radio_id, samples)
            except zmq.Again:
                time.sleep(0.001)

    def _commander_thread(self):
        """Thread to handle Leapfrog frequency hopping """
        cmd_sock = self.zmq_ctx.socket(zmq.PUSH)
        cmd_sock.connect(f"tcp://127.0.0.1:{self.config['network']['zmq_cmd']}")
        
        # R1: 2400 to 2430 in 10MHz steps
        r1_freqs = np.arange(2405e6, 2435e6, 10e6) 
        # R2: 2445 to 2475 in 10MHz steps
        r2_freqs = np.arange(2445e6, 2475e6, 10e6)
        idx = 0
        
        while self.running:
            if not self.lock_on_active:
                f1, f2 = float(r1_freqs[idx % len(r1_freqs)]), float(r2_freqs[idx % len(r2_freqs)])
                
                self.radio_states['R1'].update({'freq': f1, 'settling': True})
                self.radio_states['R2'].update({'freq': f2, 'settling': True})
                
                # Send PMT commands to GRC 
                cmd_sock.send(pmt.serialize_str(pmt.intern(f"0:{f1}"))) # R1
                cmd_sock.send(pmt.serialize_str(pmt.intern(f"2:{f2}"))) # R2
                
                time.sleep(0.3) # Settling
                self.radio_states['R1']['settling'] = False
                self.radio_states['R2']['settling'] = False

                # NEW: Update visualization with current frequencies
                if self.data_aggregator is not None:
                    self.data_aggregator.update_radio_frequencies(f1, f2)
                
                time.sleep(1.2) # Dwell
                idx += 1
            else:
                time.sleep(0.1)

    def _detection_thread(self, radio_id):
        """The 'Brain' thread: Pulls data, runs detection, and triggers AoA."""
        self.logger.info(f"[{radio_id}] Detection thread started.")
        
        while self.running:
            # 1. Get recent data from the buffer (Wait if settling)
            if self.radio_states[radio_id]['settling']:
                time.sleep(0.1)
                continue
                
            # Pull 1024 samples for FFT
            samples = self.buffer_manager.get_last(radio_id, self.config['performance']['fft_size'])
            
            if samples is None or len(samples) < self.config['performance']['fft_size']:
                time.sleep(0.01)
                continue

            # 2. Get current frequency
            current_freq = self.radio_states[radio_id]['freq']

            # 3. Phase 1: Learning
            if self.is_learning:
                self.detection_engines[radio_id].learn_noise(current_freq, samples)
                time.sleep(0.01) # Yield CPU
                continue

            # 4. Phase 2: Detection
            if not self.lock_on_active:
                result = self.detection_engines[radio_id].detect(current_freq, samples)
                
                if result['detected']:
                    # --- RAW ALERT (What you asked for) ---
                    # Get the PRECISE frequency calculated by the detector
                    # If 'true_freq' is missing, fallback to the center freq
                    target_freq = result.get('true_freq', current_freq)
                    # self.logger.info(f"[{radio_id}] !!! RAW DETECTION !!! @ {target_freq/1e6} MHz")
                    
                    # Log internally but keep console clean
                    self.logger.debug(f"Raw Hit: {target_freq/1e6}MHz")
                    
                    # 5. Filter Logic (Bypassed for now to guarantee output)
                    if not self.spectral_filter.is_valid_signal(result['psd'], result['peak_idx']):
                        self.logger.debug(f"[{radio_id}] Spectral filter rejected WiFi/Noise")
                        continue

                    # 6. Persistence Check
                    confirmed, hits, chans = self.registry.record_hit(target_freq)
                    # self.logger.info(f"Hits: {hits} | Channels: {chans}")

                    # NEW: Push event to visualization
                    if self.data_aggregator is not None:
                        from datetime import datetime
                        self.data_aggregator.push_detection_event({
                            'timestamp': datetime.now(),
                            'freq_mhz': target_freq / 1e6,
                            'radio': radio_id,
                            'hits': hits,
                            'channels': chans,
                            'confirmed': confirmed
                        })

                    if confirmed:
                        self.logger.info(
                            f"[{radio_id}] Detection @ {target_freq/1e6}MHz | "
                            f"Hits: {hits}/{self.config['detection']['persistence_threshold']}| Channels: {chans}"
                        )
                        self.logger.info(f"[{radio_id}] >>> DRONE CONFIRMED! TRIGGERING AOA <<<")
                        threading.Thread(target=self._perform_aoa, 
                                       args=(target_freq, radio_id), 
                                       daemon=True).start()
            
            time.sleep(0.01) # Prevent CPU hogging

    def _perform_aoa(self, target_freq, detector_radio):
        """The Lock-On Procedure """
        self.lock_on_active = True

        # NEW: Signal high load to visualization
        if self.data_aggregator is not None:
            self.data_aggregator.set_system_load(True)
        
        # 1. SATURATION GUARD
        # Check current buffer for clipping before tuning R3
        check_samples = self.buffer_manager.get_last(detector_radio, 1000)
        if check_samples is not None:
            peak = np.max(np.abs(check_samples))
            if peak > 0.9: # 1.0 is the digital ceiling
                self.logger.warning(f"!!! SIGNAL SATURATED (Peak={peak:.2f}) !!!")
                self.logger.warning("AoA will be inaccurate. Reduce HackRF Gains!")
        
        # Tune R3 to target
        cmd_sock = self.zmq_ctx.socket(zmq.PUSH)
        cmd_sock.connect(f"tcp://127.0.0.1:{self.config['network']['zmq_cmd']}")
        cmd_sock.send(pmt.serialize_str(pmt.intern(f"1:{target_freq}")))
        time.sleep(0.4)
        
        # Extract 100ms from RAM 
        win = int(self.config['hardware']['sample_rate'] * 0.1)
        samples_r1 = self.buffer_manager.get_last('R1', win)
        samples_r3 = self.buffer_manager.get_last('R3', win)
        
        if samples_r1 is not None and samples_r3 is not None:
            result = self.aoa_calculator.calculate(samples_r1, samples_r3, target_freq)
            
            if result['success']:
                # Applying the "Moving Median" logic (Simple 3-hit buffer)
                if not hasattr(self, 'angle_history'): self.angle_history = []
                self.angle_history.append(result['angle_deg'])
                if len(self.angle_history) > 3: self.angle_history.pop(0)
                
                stable_angle = np.median(self.angle_history)
                direction = "RIGHT" if stable_angle > 0 else "LEFT"
                self.logger.info(f"[*] AoA FIX: {abs(stable_angle):.2f}° {direction}")

                # NEW: Push to visualization
                if self.data_aggregator is not None:
                    from datetime import datetime
                    self.data_aggregator.push_aoa_measurement({
                        'timestamp': datetime.now(),
                        'freq_mhz': target_freq / 1e6,
                        'angle_deg': stable_angle,
                        'confidence': result.get('confidence', 0),
                        'phase_deg': result.get('phase_deg', 0)
                    })
                
            else:
                self.logger.info(f"[AoA] Rejected: {result['error']}")
        
        self.lock_on_active = False

        # NEW: Signal high load to visualization
        if self.data_aggregator is not None:
            self.data_aggregator.set_system_load(False)

    def _check_grc_status(self):
        """Checks if ZMQ streams are active before starting."""
        self.logger.info("[Init] Checking GNU Radio status...")
        
        # We'll check Radio 1's port as a representative
        port = self.config['network']['zmq_r1']
        check_sock = self.zmq_ctx.socket(zmq.SUB)
        check_sock.connect(f"tcp://127.0.0.1:{port}")
        check_sock.setsockopt(zmq.SUBSCRIBE, b'')
        check_sock.setsockopt(zmq.RCVTIMEO, 2000) # 2-second timeout
        
        try:
            self.logger.info(f"[Init] Listening for data on port {port}...")
            check_sock.recv() # Try to get one packet
            self.logger.info("[Init] ✓ GNU Radio detected and streaming.")
            check_sock.close()
            return True
        except zmq.Again:
            self.logger.error("="*60)
            self.logger.error(" ERROR: GNU RADIO NOT DETECTED ".center(60))
            self.logger.error("="*60)
            self.logger.error(f" No data received on ZMQ port {port}.")
            self.logger.error(" Please ensure your GRC Flowgraph is RUNNING.")
            self.logger.error("="*60)
            check_sock.close()
            return False

    def start(self):
        """Start the detection system."""
        # Pre-Flight Check
        if not self._check_grc_status():
            self.logger.critical("Aborting startup: GRC is offline.")
            sys.exit(1)

        # 1. Start Buffer Threads (The Ears)
        for rid in ['R1', 'R2', 'R3']:
            threading.Thread(target=self._zmq_buffer_thread, args=(rid,), daemon=True).start()
        
        # 2. Start Commander (The Legs)
        threading.Thread(target=self._commander_thread, daemon=True).start()
        
        # 3. START DETECTION THREADS (The Brain) 
        for rid in ['R1', 'R2']:
            threading.Thread(target=self._detection_thread, args=(rid,), daemon=True).start()

        # NEW: Start visualization system
        if self.data_aggregator is not None:
            # Start data aggregator thread
            self.data_aggregator.start()
            self.logger.info("[Viz] Data aggregator started")
            
            # Start web server in separate thread
            web_port = self.config['visualization'].get('web_port', 8080)
            threading.Thread(
                target=start_web_server,
                args=(self.data_aggregator, web_port),
                daemon=True,
                name="WebServer"
            ).start()
            
            # Give server time to start
            time.sleep(1)
            
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            self.logger.info(f"[Viz] Dashboard available at:")
            self.logger.info(f"      http://localhost:{web_port}")
            self.logger.info(f"      http://{local_ip}:{web_port}")
        
        # Calibration 
        self.logger.info(f"CALIBRATING ({self.config['detection']['calibration_time']}s)...")
        time.sleep(self.config['detection']['calibration_time'])
        self.is_learning = False
        self.logger.info("LIVE MONITORING ACTIVE")

        while self.running:
            time.sleep(1)

if __name__ == "__main__":
    system = TripleSDRSystem()
    system.start()

