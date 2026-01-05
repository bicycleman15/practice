import torch
import torch.nn.functional as F

# Simple 1D convolution as matrix multiplication
# Input: 1D tensor, Kernel: 1D tensor

x = torch.arange(6).float()  # [0, 1, 2, 3, 4, 5]
kernel = torch.tensor([1., 2., 3.])  # kernel size = 3
K = kernel.shape[0]

print("Input:", x)
print("Kernel:", kernel)

# === Method 3: Convolution as Matrix Multiplication ===
# Step 1: Extract all patches using unfold (im2col)
patches = F.pad(x, (K-1, 0))
patches = patches.unfold(dimension=0, step=1, size=K)  # (L_out, K)
print("\nPatches (im2col):")
print(patches)

# # Step 2: Matrix multiply patches @ kernel
out_matmul = patches @ kernel  # (L_out,)
print("\nMatmul output:", out_matmul.shape)

# # Verify all methods match
# print("\n✓ All methods match!" if torch.allclose(out_conv.squeeze(), out_matmul) else "✗ Mismatch!")
