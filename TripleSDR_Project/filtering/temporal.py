import numpy as np
import time

class TemporalFilter:
    """
    Analyzes burst duration and hop regularity.
    Source: remaining_files_guide.md
    """
    def __init__(self, config):
        self.max_duration = config['temporal']['burst_duration_max'] / 1000.0
        self.regularity_threshold = config['temporal']['hop_regularity_threshold']
        self.burst_history = [] # [(timestamp, freq_mhz), ...]

    def is_valid_timing(self, freq_mhz):
        """Checks if recent hits follow a regular hopping pattern."""
        now = time.time()
        self.burst_history.append((now, freq_mhz))
        
        # Keep only last 10 hits for analysis
        if len(self.burst_history) > 10:
            self.burst_history.pop(0)
            
        if len(self.burst_history) < 5:
            return True # Not enough data to reject yet
            
        # Calculate time between hops
        intervals = [self.burst_history[i+1][0] - self.burst_history[i][0] 
                     for i in range(len(self.burst_history)-1)]
        
        mean_int = np.mean(intervals)
        std_int = np.std(intervals)
        regularity = std_int / mean_int if mean_int > 0 else 999
        
        # Drones are very regular (low regularity score)
        return regularity < self.regularity_threshold