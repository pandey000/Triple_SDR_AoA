import numpy as np
import zmq
import time

def test_dual_coherence():
    ctx = zmq.Context()
    # Using R1 (60000) and R3 (60001)
    s1 = ctx.socket(zmq.SUB)
    s1.connect("tcp://127.0.0.1:60000")
    s1.setsockopt(zmq.SUBSCRIBE, b'')
    
    s3 = ctx.socket(zmq.SUB)
    s3.connect("tcp://127.0.0.1:60001")
    s3.setsockopt(zmq.SUBSCRIBE, b'')

    print("[*] Testing Sync between R1 and R3 using your high-quality cable...")
    time.sleep(1) # Warmup
    
    phase_history = []
    for i in range(50):
        try:
            # Non-blocking check for data
            b1 = s1.recv(zmq.NOBLOCK)
            b3 = s3.recv(zmq.NOBLOCK)
            
            d1 = np.frombuffer(b1, dtype=np.complex64)
            d3 = np.frombuffer(b3, dtype=np.complex64)

            # Find the smallest length between the two received buffers
            min_len = min(len(d1), len(d3), 1024) 
            
            if min_len > 0:
                # Multiply only the overlapping parts
                avg_vec = np.mean(d1[:min_len] * np.conj(d3[:min_len]))
                phase_history.append(np.angle(avg_vec))
            
            time.sleep(0.05)
        except zmq.Again:
            continue

    if not phase_history:
        print("[!] No data received. Check GRC ZMQ Sinks.")
        return

    std_dev = np.std(phase_history)
    print(f"\nResults over {len(phase_history)} samples:")
    print(f"Standard Deviation: {std_dev:.4f} radians")
    
    if std_dev < 0.1:
        print(">>> SUCCESS: The single-cable link is stable.")
    else:
        print(">>> FAIL: Still seeing jitter. Check connectors.")

if __name__ == "__main__":
    test_dual_coherence()