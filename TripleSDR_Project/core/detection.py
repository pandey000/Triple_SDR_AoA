import numpy as np
import time
import threading

class DetectionEngine:
    """
    Statistical Energy Detection with Edge Masking and Overlap Logic.
    """
    def __init__(self, config, radio_id):
        self.config = config
        self.radio_id = radio_id
        # Remove 'detection' key if config is passed as sub-dict
        # Adjust based on how main_controller passes config
        if 'detection' in config:
            self.manual_offset = config['detection']['manual_offset']
        else:
            self.manual_offset = config.get('manual_offset', 15.0)
            
        self.calib_buffer = {}
        self.stats = {}
        self.fp_rate = {} 
        self.sample_rate = config['hardware']['sample_rate']
        self.fft_size = config['performance']['fft_size']

    def learn_noise(self, freq, samples):
        """Accumulates PSDs during the learning phase."""
        f_key = int(freq / 1e6)
        # Hanning window is critical for accurate noise floor mapping
        window = np.hanning(len(samples))
        psd = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples * window))) / len(samples) + 1e-12)
        
        if f_key not in self.calib_buffer:
            self.calib_buffer[f_key] = []
        
        self.calib_buffer[f_key].append(psd)
        
        # Finalize after 10 sweeps
        if len(self.calib_buffer[f_key]) >= 10:
            arr = np.array(self.calib_buffer[f_key])
            self.stats[f_key] = {
                'mean': np.mean(arr, axis=0),
                'std': np.std(arr, axis=0)
            }
            del self.calib_buffer[f_key]

    def detect(self, freq, samples):
        """
        Detects signals while ignoring DC spike and Edge Rolloff.
        Returns 'true_freq' for precise AoA targeting.
        """
        f_key = int(freq / 1e6)
        
        # 1. Calculate PSD with Hanning Window
        window = np.hanning(len(samples))
        psd = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples * window))) / len(samples) + 1e-12)
        
        # 2. APPLY MASKS (The Filters)
        center = len(psd) // 2
        
        # A. DC Notch (Kill the LO Leakage at Center)
        # We silence +/- 10 bins around the center
        psd[center - 10 : center + 10] = -120

        # B. Edge Mask (Kill the Rolloff False Positives)
        # We silence the first 100 and last 100 bins (approx 1.5 MHz each side)
        # The Overlapping Frequency Plan will cover these blind spots.
        psd[:100] = -120
        psd[-100:] = -120

        if f_key in self.stats:
            mean_noise = self.stats[f_key]['mean']
            std_noise = self.stats[f_key]['std']
            
            # Adaptive Penalty for noisy channels
            adaptive_offset = 0
            if self.fp_rate.get(f_key, 0) > 10:
                adaptive_offset = 5.0
            
            threshold = mean_noise + (3 * std_noise) + self.manual_offset + adaptive_offset
            
            # 3. Check for Violations
            violations = np.where(psd > threshold)[0]
            
            if len(violations) > 0:
                peak_idx = violations[np.argmax(psd[violations])]
                
                # 4. Calculate TRUE Frequency
                # Center freq is index 512. We calculate offset from there.
                bin_width_hz = self.sample_rate / self.fft_size
                freq_offset_hz = (peak_idx - (self.fft_size // 2)) * bin_width_hz
                true_freq_hz = freq + freq_offset_hz
                
                return {
                    'detected': True, 
                    'peak_idx': peak_idx, 
                    'psd': psd,
                    'true_freq': true_freq_hz # <--- Precise frequency for AoA
                }
        
        return {'detected': False}

    def report_false_positive(self, freq):
        f_key = int(freq / 1e6)
        self.fp_rate[f_key] = self.fp_rate.get(f_key, 0) + 1


class PersistenceRegistry:
    """
    Tracks signal hits across frequencies and time to confirm drone patterns.
    """
    def __init__(self, threshold=8, window=15.0, min_channels=3):
        self.threshold = threshold  
        self.window = window       
        self.min_channels = min_channels 
        self.hits = []              
        self.lock = threading.Lock()

    def record_hit(self, freq_hz):
        with self.lock:
            now = time.time()
            f_mhz = int(freq_hz / 1e6)
            self.hits.append((now, f_mhz))
            self.hits = [h for h in self.hits if now - h[0] < self.window]
            hit_count = len(self.hits)
            unique_channels = len(set([h[1] for h in self.hits]))
            is_confirmed = (hit_count >= self.threshold and unique_channels >= self.min_channels)
            return is_confirmed, hit_count, unique_channels