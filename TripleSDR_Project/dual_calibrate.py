import numpy as np
import zmq
import yaml
import time
from colorama import Fore, Style, init

init(autoreset=True)

def run_dual_calibration():
    print(f"\n{Fore.YELLOW}{'='*60}")
    print(f"{Fore.WHITE}   DUAL-RADIO BORESIGHT CALIBRATION (R1 + R3)   ".center(60, "="))
    print(f"{Fore.YELLOW}{'='*60}\n")

    # 1. User Input for Frequency
    try:
        freq_mhz = float(input(f"{Fore.GREEN}Enter Calibration Frequency (MHz) [e.g. 2462]: {Style.RESET_ALL}"))
        freq_hz = freq_mhz * 1e6
    except ValueError:
        print(f"{Fore.RED}Invalid input. Exiting.")
        return

    ctx = zmq.Context()
    
    # Connect to R1 (Master) and R3 (AoA Slave)
    s1 = ctx.socket(zmq.SUB)
    s1.connect("tcp://127.0.0.1:60000")
    s1.setsockopt(zmq.SUBSCRIBE, b'')
    s1.setsockopt(zmq.RCVTIMEO, 2000)

    s3 = ctx.socket(zmq.SUB)
    s3.connect("tcp://127.0.0.1:60001")
    s3.setsockopt(zmq.SUBSCRIBE, b'')
    s3.setsockopt(zmq.RCVTIMEO, 2000)

    print(f"\n{Fore.CYAN}[Setup] 1. Antennas EXACTLY 6.25cm apart.")
    print(f"{Fore.CYAN}[Setup] 2. Source (Drone/Hotspot) EXACTLY at 0° Boresight.")
    print(f"{Fore.CYAN}[Setup] 3. R1 & R3 connected via High-Quality SMA Sync Cable.")
    input(f"\n{Fore.YELLOW}Press Enter to start 5-point stability check...{Style.RESET_ALL}")

    offsets = []
    
    # 2. 5-Point Stability Check
    for i in range(5):
        try:
            print(f"Capture {i+1}/5...", end='\r')
            b1 = s1.recv()
            b3 = s3.recv()
            d1 = np.frombuffer(b1, dtype=np.complex64)
            d3 = np.frombuffer(b3, dtype=np.complex64)

            # Basic vector averaging for phase
            # We use a 10k sample window for high precision
            win = 10000
            # Find the smallest length between the two received buffers
            min_len = min(len(d1), len(d3), 1024) 
            
            if min_len > 0:
                # Multiply only the overlapping parts
                avg_vec = np.mean(d1[:min_len] * np.conj(d3[:min_len]))
                phase = np.angle(avg_vec)
                offsets.append(phase)
                time.sleep(0.2)
        except zmq.Again:
            print(f"\n{Fore.RED}[Error] Timeout! Check if GRC is running.")
            return

    # 3. Analyze Results
    std_dev = np.std(offsets)
    mean_offset = np.mean(offsets)

    print(f"\n\n{Fore.WHITE}--- Results ---")
    print(f"Mean Phase Offset: {mean_offset:.6f} rad")
    print(f"Phase Stability (StdDev): {std_dev:.4f} rad")

    # 4. Success Logic
    if std_dev < 0.1:
        print(f"{Fore.GREEN}STATUS: SUCCESS. Hardware is stable.{Style.RESET_ALL}")
        
        # 5. Hemisphere Guard Check (Logical Warning)
        if abs(mean_offset) > (np.pi / 2):
            print(f"{Fore.MAGENTA}[Note] 180° Ambiguity detected. Math will auto-correct.{Style.RESET_ALL}")
        
        # Save to Config (Optional - you can do this manually for now)
        print(f"\n{Fore.YELLOW}Action: Update your config.yaml with:")
        print(f"{Fore.WHITE}aoa:\n  calibration_offset: {mean_offset:.6f}\n  calibration_frequency: {freq_hz}")
    else:
        print(f"{Fore.RED}STATUS: FAIL. Jitter too high. Re-tighten cables.{Style.RESET_ALL}")

if __name__ == "__main__":
    run_dual_calibration()