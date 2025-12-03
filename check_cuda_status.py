import torch

def check_cuda_status():
    print(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        print("CUDA ist verfügbar! ✅")
        print(f"CUDA Version (wie von PyTorch verwendet): {torch.version.cuda}")

        device_count = torch.cuda.device_count()
        print(f"Anzahl der CUDA-Geräte: {device_count}")

        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            # Compute Capability ist ein Tupel (major, minor)
            capability = torch.cuda.get_device_capability(i)
            print(f"\nGerät {i}: {device_name}")
            print(f"  - Compute Capability: {capability[0]}.{capability[1]}")
    else:
        print("CUDA ist nicht verfügbar. PyTorch läuft auf der CPU. ❌")

if __name__ == "__main__":
    check_cuda_status()