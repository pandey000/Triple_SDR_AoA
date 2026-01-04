import time
import json

class MetricsCollector:
    """Tracks detection hits, AoA success rate, and system uptime."""
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            'detections': {'total': 0, 'confirmed': 0},
            'aoa': {'attempts': 0, 'successful': 0},
            'performance': {'uptime': 0}
        }
        
    def record_detection(self, confirmed=False):
        self.metrics['detections']['total'] += 1
        if confirmed: self.metrics['detections']['confirmed'] += 1
        
    def record_aoa(self, success):
        self.metrics['aoa']['attempts'] += 1
        if success: self.metrics['aoa']['successful'] += 1
            
    def export(self, path):
        self.metrics['performance']['uptime'] = time.time() - self.start_time
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)