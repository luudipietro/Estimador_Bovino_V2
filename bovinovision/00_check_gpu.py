"""Etapa 0: detecta el backend de cálculo (ROCm/CUDA o CPU) y mide una multiplicación de matrices.
En Fedora con la RX 6700 XT exportar antes:  HSA_OVERRIDE_GFX_VERSION=10.3.0"""
import os
import platform
import time


def main():
    print(f"SO: {platform.platform()} | Python {platform.python_version()}")
    print(f"HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    try:
        import torch
    except ImportError:
        print("torch no está instalado (instalar extra 'cpu' o 'rocm').")
        return
    print(f"torch {torch.__version__} | hip={getattr(torch.version, 'hip', None)} | cuda={torch.version.cuda}")
    if torch.cuda.is_available():  # ROCm también se expone como 'cuda'
        dev = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("Sin GPU visible para torch -> CPU")
    x = torch.randn(4096, 4096, device=dev)
    for _ in range(2):
        (x @ x).sum().item()
    t = time.time()
    for _ in range(5):
        (x @ x).sum().item()
    print(f"backend={dev.type} | 5 matmul 4096x4096: {time.time() - t:.2f}s")


if __name__ == "__main__":
    main()
