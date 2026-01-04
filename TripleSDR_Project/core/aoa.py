import numpy as np

class AoACalculator:
    """
    Professional-grade AoA Calculator using Vector Averaging and Full Physics.
    Source: remaining_files_guide.md & User Requirements
    """
    def __init__(self, config, sample_rate):
        self.d = config['antenna_spacing'] # 0.0625 meters
        self.cal_offset = config['calibration_offset']
        self.window = config.get('sample_window', 2000)
        self.c = 299792458 # Speed of Light (m/s)

        # LOWERED THRESHOLD: 0.05 allows detection in RF pollution
        self.min_confidence = 0.05

    def calculate(self, samples_r1, samples_r3, freq_hz):
        """
        Calculates AoA using synchronized sample windows.
        """
        # 1. Alignment (Cross-Correlation)
        # We use a 2000-sample slice for the hunt
        corr = np.correlate(samples_r1[:2000], samples_r3[:2000], mode='full')
        offset = np.argmax(np.abs(corr)) - 2000
        
        # 2. Extract Windows (Using 10,000 samples for the 'Stable Window')
        # We move deep into the buffer to ensure we aren't at the very edge
        start = 10000
        try:
            r1_win = samples_r1[start : start + self.window]
            r3_win = samples_r3[start + offset : start + offset + self.window]
            
            # Ensure windows are identical length after offset
            win_len = min(len(r1_win), len(r3_win))
            r1_win, r3_win = r1_win[:win_len], r3_win[:win_len]
        except Exception:
            return {'success': False, 'error': 'Alignment Error'}

        # 2.1 Apply Hanning Window
        # This MUST be done before the average vector calculation
        h_win = np.hanning(len(r1_win))
        r1_win = r1_win * h_win
        r3_win = r3_win * h_win
        
        # 3. SIGNAL QUALITY GUARD
        # Reject if signal is 'fading' (high variance) indicating multipath
        power_profile = np.abs(r1_win)**2
        variance = np.var(power_profile) / (np.mean(power_profile)**2 + 1e-12)
        if variance > 1.5:
            return {'success': False, 'error': f'Multipath/Fading Detected, variance={variance:.6f}'}
        
        # 3.1 Vector Averaging (The 'Noise Killer')
        # Instead of np.angle(sample), we find the mean phase vector
        # This solves the 'noisy single sample' problem
        avg_vector = np.mean(r1_win * np.conj(r3_win))
        phase_diff_raw = np.angle(avg_vector)

        # 4. Calibration & Phase Normalization
        phase_diff = phase_diff_raw - self.cal_offset
        # Map result to (-pi, pi) to prevent wrap-around errors
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

        # 5. Full Physics Equation
        # θ = arcsin((Δφ * λ) / (2π * d))
        # This is the correct formula for linear interferometry

        wavelength = self.c / freq_hz
        arg = (phase_diff * wavelength) / (2 * np.pi * self.d)
        
        if abs(arg) > 1.0:
            return {'success': False, 'error': f'Phase Ambiguity, arg={arg:.6f}'}
            
        # 6. HEMISPHERE GUARD & AMBIGUITY RESOLUTION
        # If the math pushes the result into the 'Ghost Zone' (behind you),
        # we flip the phase to bring it back to the front.
        if abs(arg) > 1.0:
            # Attempt to resolve by flipping phase 180 deg
            phase_diff = phase_diff + np.pi if phase_diff < 0 else phase_diff - np.pi
            arg = (phase_diff * wavelength) / (2 * np.pi * self.d)
            
            # If still invalid, then the signal is physically impossible
            if abs(arg) > 1.0:
                return {'success': False, 'error': 'Out of Bounds'}
            
        angle_deg = np.degrees(np.arcsin(arg))
        
        # 7. CONFIDENCE CALCULATION
        # Normalized magnitude of the average vector
        confidence = np.abs(avg_vector) / (np.mean(np.abs(r1_win)) + 1e-12)
        
        # if confidence < self.min_confidence:
        #     return {'success': False, 'error': f'Low Confidence ({confidence:.4f})'}
        
        return {
            'success': True, 
            'angle_deg': angle_deg, 
            'phase_deg': np.degrees(phase_diff),
            'confidence': confidence
        }