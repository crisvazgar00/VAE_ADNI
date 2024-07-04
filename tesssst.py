import torch

import torch

# Verificar si CUDA está disponible
cuda_available = torch.cuda.is_available()
print(f"CUDA disponible: {cuda_available}")


if cuda_available:
    # Obtener el nombre del dispositivo CUDA
    device_name = torch.cuda.get_device_name(0)
    print(f"Nombre del dispositivo CUDA: {device_name}")
    
    # Obtener la capacidad de computación del dispositivo CUDA
    device_capability = torch.cuda.get_device_capability(0)
    print(f"Capacidad de computación del dispositivo CUDA: {device_capability}")
