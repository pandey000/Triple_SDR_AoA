#!/usr/bin/env python3
import sys
import time
import threading
import signal
import yaml
import logging
import json
from pathlib import Path
from datetime import datetime
import socket
import numpy as np
import zmq
import pmt
import paho.mqtt.client as mqtt

# Reuse core modules
from core.buffers import BufferManager
from core.detection import DetectionEngine, PersistenceRegistry
from core.aoa import AoACalculator
from filtering.temporal import TemporalFilter
from filtering.spectral import SpectralFilter
from utils.logger import setup_logging
from utils.metrics import MetricsCollector

class TripleSDRSystemDistributed:
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
        
        # 4. MQTT Client Setup
        self.mqtt_client = mqtt.Client()
        try:
            self.mqtt_client.connect("localhost", 1883)
            self.logger.info("[Init] MQTT Client connected to localhost:1883")
        except Exception as e:
            self.logger.error(f"[Init] Failed to connect to MQTT broker: {e}")

        # 5. Initialize Subsystems 
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

    def _zmq_buffer_thread(self, radio_id):
        """Thread to pull IQ data into the CircularBuffer"""
        port = self.config['network'][f'zmq_{radio_id.lower()}']
        sock = self.zmq_ctx.socket(zmq.SUB)
        sock.connect(f"tcp://127.0.0.1:{port}")
        sock.setsockopt(zmq.SUBSCRIBE, b'')
        sock.setsockopt(zmq.RCVHWM, 100)
        
        while self.running:
            try:
                payload = sock.recv(flags=zmq.NOBLOCK)
                samples = np.frombuffer(payload, dtype=np.complex64)
                self.buffer_manager.append(radio_id, samples)
            except zmq.Again:
                time.sleep(0.001)

    def _commander_thread(self):
        """Thread to handle Leapfrog frequency hopping"""
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
                
                cmd_sock.send(pmt.serialize_str(pmt.intern(f"0:{f1}"))) # R1
                cmd_sock.send(pmt.serialize_str(pmt.intern(f"2:{f2}"))) # R2
                
                time.sleep(0.3) # Settling
                self.radio_states['R1']['settling'] = False
                self.radio_states['R2']['settling'] = False
                
                time.sleep(1.2) # Dwell
                idx += 1
            else:
                time.sleep(0.1)

    def _publish_mqtt(self, topic, payload_dict):
        """Helper to safely publish JSON to MQTT"""
        try:
            self.mqtt_client.publish(topic, json.dumps(payload_dict))
        except Exception:
            pass

    def _detection_thread(self, radio_id):
        """The 'Brain' thread: Pulls data, runs detection, and triggers AoA."""
        self.logger.info(f"[{radio_id}] Detection thread started.")
        
        while self.running:
            if self.radio_states[radio_id]['settling']:
                time.sleep(0.1)
                continue
                
            samples = self.buffer_manager.get_last(radio_id, self.config['performance']['fft_size'])
            
            if samples is None or len(samples) < self.config['performance']['fft_size']:
                time.sleep(0.01)
                continue

            current_freq = self.radio_states[radio_id]['freq']

            if self.is_learning:
                self.detection_engines[radio_id].learn_noise(current_freq, samples)
                time.sleep(0.01) 
                continue

            if not self.lock_on_active:
                result = self.detection_engines[radio_id].detect(current_freq, samples)
                
                # --- MQTT STREAMING ---
                if 'psd' in result:
                    spectrum_payload = {
                        'radio': radio_id,
                        'freq_hz': current_freq,
                        'psd': np.round(result['psd'], 2).tolist()
                    }
                    self._publish_mqtt("sdr/spectrum", spectrum_payload)

                if result['detected']:
                    target_freq = result.get('true_freq', current_freq)
                    self.logger.debug(f"Raw Hit: {target_freq/1e6}MHz")
                    
                    if not self.spectral_filter.is_valid_signal(result['psd'], result['peak_idx']):
                        continue

                    # Persistence Check
                    confirmed, hits, chans = self.registry.record_hit(target_freq)
                    
                    # Publish Event
                    event_payload = {
                        'timestamp': datetime.now().isoformat(),
                        'radio': radio_id,
                        'freq_hz': target_freq,
                        'hits': hits,
                        'channels': chans,
                        'confirmed': confirmed
                    }
                    self._publish_mqtt("sdr/events", event_payload)

                    if confirmed:
                        self.logger.info(f"[{radio_id}] >>> DRONE CONFIRMED! TRIGGERING AOA <<<")
                        self._publish_mqtt("drone/trigger", "START")
                        threading.Thread(target=self._perform_aoa, 
                                       args=(target_freq, radio_id), 
                                       daemon=True).start()
            
            time.sleep(0.01)

    def _perform_aoa(self, target_freq, detector_radio):
        self.lock_on_active = True
        
        # 1. SATURATION GUARD
        check_samples = self.buffer_manager.get_last(detector_radio, 1000)
        if check_samples is not None:
            peak = np.max(np.abs(check_samples))
            if peak > 0.9: 
                self.logger.warning(f"!!! SIGNAL SATURATED (Peak={peak:.2f}) !!!")
        
        # Tune R3 to target
        cmd_sock = self.zmq_ctx.socket(zmq.PUSH)
        cmd_sock.connect(f"tcp://127.0.0.1:{self.config['network']['zmq_cmd']}")
        cmd_sock.send(pmt.serialize_str(pmt.intern(f"1:{target_freq}")))
        time.sleep(0.4)
        
        # Extract 100ms
        win = int(self.config['hardware']['sample_rate'] * 0.1)
        samples_r1 = self.buffer_manager.get_last('R1', win)
        samples_r3 = self.buffer_manager.get_last('R3', win)
        
        if samples_r1 is not None and samples_r3 is not None:
            result = self.aoa_calculator.calculate(samples_r1, samples_r3, target_freq)
            
            if result['success']:
                if not hasattr(self, 'angle_history'): self.angle_history = []
                self.angle_history.append(result['angle_deg'])
                if len(self.angle_history) > 3: self.angle_history.pop(0)
                
                stable_angle = np.median(self.angle_history)
                self.logger.info(f"[*] AoA FIX: {abs(stable_angle):.2f}°")

                aoa_payload = {
                    'timestamp': datetime.now().isoformat(),
                    'freq_mhz': target_freq / 1e6,
                    'angle_deg': float(stable_angle),
                    'confidence': float(result.get('confidence', 0)),
                    'phase_deg': float(result.get('phase_deg', 0))
                }
                self._publish_mqtt("sdr/aoa", aoa_payload)
            else:
                self.logger.info(f"[AoA] Rejected: {result['error']}")
        
        self.lock_on_active = False

    def _check_grc_status(self):
        """Checks if ZMQ streams are active before starting."""
        self.logger.info("[Init] Checking GNU Radio status...")
        port = self.config['network']['zmq_r1']
        check_sock = self.zmq_ctx.socket(zmq.SUB)
        check_sock.connect(f"tcp://127.0.0.1:{port}")
        check_sock.setsockopt(zmq.SUBSCRIBE, b'')
        check_sock.setsockopt(zmq.RCVTIMEO, 2000) 
        
        try:
            check_sock.recv() 
            self.logger.info("[Init] ✓ GNU Radio detected.")
            check_sock.close()
            return True
        except zmq.Again:
            self.logger.error(" ERROR: GNU RADIO NOT DETECTED ".center(60))
            check_sock.close()
            return False

    def start(self):
        """Start the distributed system."""
        if not self._check_grc_status():
            self.logger.critical("Aborting startup: GRC is offline.")
            sys.exit(1)

        # 1. Start Buffer Threads
        for rid in ['R1', 'R2', 'R3']:
            threading.Thread(target=self._zmq_buffer_thread, args=(rid,), daemon=True).start()
        
        # 2. Start Commander
        threading.Thread(target=self._commander_thread, daemon=True).start()
        
        # 3. Start Detection Threads
        for rid in ['R1', 'R2']:
            threading.Thread(target=self._detection_thread, args=(rid,), daemon=True).start()

        self.logger.info(f"CALIBRATING ({self.config['detection']['calibration_time']}s)...")
        time.sleep(self.config['detection']['calibration_time'])
        self.is_learning = False
        self.logger.info("LIVE MONITORING ACTIVE (DISTRIBUTED MODE)")

        while self.running:
            try:
                stats = self.buffer_manager.get_all_stats()
                metrics_payload = {
                    'timestamp': datetime.now().isoformat(),
                    'uptime': time.time() - self.metrics.start_time,
                    'buffers': {
                        rid: stats[rid]['fill_level'] for rid in stats
                    },
                    'detections': self.metrics.metrics['detections']
                }
                self._publish_mqtt("sdr/metrics", metrics_payload)
            except Exception:
                pass
            time.sleep(1)

if __name__ == "__main__":
    system = TripleSDRSystemDistributed()
    system.start()
