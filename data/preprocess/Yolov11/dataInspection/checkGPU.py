import subprocess
import sys

def detect_gpu():
    try:
        # Run nvidia-smi to get GPU info
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("NVIDIA GPU detected! Here's the detailed info:")
        print(result.stdout)
    except subprocess.CalledProcessError:
        print("No NVIDIA GPU detected or nvidia-smi not found. Ensure NVIDIA drivers are installed.")
    except FileNotFoundError:
        print("nvidia-smi command not found. This might mean no NVIDIA GPU or drivers not installed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    detect_gpu()