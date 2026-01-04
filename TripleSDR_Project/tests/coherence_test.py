import numpy as np
import zmq
import time

def test_hardware_sync():
    ctx = zmq.Context()
    # R1 and R3 must match your GRC ZMQ PUB Sink Ports
    s1 = ctx.socket(zmq.SUB)
    s1.connect("tcp://127.0.0.1:60000")
    s1.setsockopt(zmq.SUBSCRIBE, b'')
    s1.setsockopt(zmq.RCVTIMEO, 1000) # 1 second timeout

    s3 = ctx.socket(zmq.SUB)
    s3.connect("tcp://127.0.0.1:60001")
    s3.setsockopt(zmq.SUBSCRIBE, b'')
    s3.setsockopt(zmq.RCVTIMEO, 1000)

    print("[*] Warming up ZMQ connections (waiting for data)...")
    
    # WARMUP LOOP: Don't start testing until we get at least one packet
    for _ in range(10):
        try:
            s1.recv()
            s3.recv()
            print("[+] Data flow detected. Starting test...")
            break
        except zmq.Again:
            print("[.] Waiting...")
            time.sleep(0.5)
    else:
        print("[!] FATAL: No data on ports 60000/60001. Is GRC running?")
        return

    phase_history = []
    print("[*] Measuring Phase Stability (100 samples)...")
    
    for i in range(100):
        try:
            # We use blocking recv here to ensure we get a fresh pair
            b1 = s1.recv()
            b3 = s3.recv()
            
            data1 = np.frombuffer(b1, dtype=np.complex64)
            data3 = np.frombuffer(b3, dtype=np.complex64)
            
            # Use only the first 1024 samples to ensure we stay within packet bounds
            win = 1024
            if len(data1) >= win and len(data3) >= win:
                # Calculate the relative phase between the two signals
                # We use the angle of the complex cross-product
                avg_vec = np.mean(data1[:win] * np.conj(data3[:win]))
                phase_history.append(np.angle(avg_vec))
            
            if i % 10 == 0: print(f"Progress: {i}%")
            
        except Exception as e:
            continue

    if len(phase_history) < 50:
        print(f"[!] Error: Only captured {len(phase_history)} samples. Need more stability.")
        return

    # THE TRUTH REVEALED
    phase_std = np.std(phase_history)
    print("\n" + "="*40)
    print(f" PHASE STABILITY RESULTS ".center(40, "="))
    print(f"Phase Std Dev: {phase_std:.4f} radians")
    print(f"Phase Std Dev: {np.degrees(phase_std):.2f} degrees")
    print("="*40)
    print(phase_history)
    # 

    if phase_std < 0.1:
        print(">>> STATUS: EXCELLENT. Hardware is perfectly synced.")
    elif phase_std < 0.4:
        print(">>> STATUS: MARGINAL. AoA will be jittery. Check SMA cables.")
    else:
        print(">>> STATUS: FAIL. No sync. Radios are drifting independently.")

if __name__ == "__main__":
    test_hardware_sync()