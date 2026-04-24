import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    # Thử đẩy một tensor lên GPU
    x = torch.tensor([1.0, 2.0]).to("cuda")
    print(f"Success! Tensor is on: {x.device}")
else:
    print("Vẫn chưa nhận được GPU. Hãy kiểm tra lại môi trường ảo (Venv) của bạn.")