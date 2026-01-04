import numpy as np

class SpectralFilter:
    """
    Analyzes frequency-domain characteristics to distinguish drones from WiFi.
    Source: remaining_files_guide.md
    """
    def __init__(self, config):
        self.rolloff_threshold = config['spectral']['rolloff_threshold']
        self.par_min = config['spectral']['par_min']
        self.par_max = config['spectral']['par_max']
        
    def is_valid_signal(self, psd, peak_idx):
        """Apply spectral tests to classify the signal."""
        # 1. Peak-to-Average Ratio (PAR) Test
        peak_pwr = psd[peak_idx]
        avg_pwr = np.mean(psd)
        par = peak_pwr - avg_pwr
        if not (self.par_min < par < self.par_max):
            return False
            
        # 2. Spectral Rolloff (Slope) Test
        # Measures how quickly power decreases away from the peak.
        rolloff = self._calculate_rolloff(psd, peak_idx)
        if rolloff > self.rolloff_threshold:
            return False  # Too sharp/steep (likely WiFi)
            
        return True
        
    def _calculate_rolloff(self, psd, peak_idx):
        peak_power = psd[peak_idx]
        
        # Find -10dB points on both sides
        left_idx = peak_idx
        while left_idx > 0 and psd[left_idx] > (peak_power - 10):
            left_idx -= 1
            
        right_idx = peak_idx
        while right_idx < len(psd) - 1 and psd[right_idx] > (peak_power - 10):
            right_idx += 1
        
        # Calculate slope (simplified for 15MS/s and 1024 FFT)
        dist_left = peak_idx - left_idx
        dist_right = right_idx - peak_idx
        
        if dist_left == 0 or dist_right == 0: return 999
        return 10 / min(dist_left, dist_right)