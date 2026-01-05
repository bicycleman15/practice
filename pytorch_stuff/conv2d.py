import torch
import torch.nn.functional as F

# === 2D Convolution (Images) via Matrix Multiplication ===

B, C_in, H, W = 2, 3, 8, 8   # batch, channels, height, width
C_out = 4
K = 3  # kernel size (K x K)
padding = 1

x = torch.randn((B, C_in, H, W))
kernel = torch.randn((C_out, C_in, K, K))

# Step 1: Pad input
x_padded = F.pad(x, [padding] * 4)  # (B, C_in, H+2, W+2)

# Step 2: Extract patches using unfold (im2col)
# unfold on H dimension, then on W dimension
patches = x_padded.unfold(2, K, 1).unfold(3, K, 1)  # (B, C_in, H_out, W_out, K, K)
H_out, W_out = patches.shape[2], patches.shape[3]

# Step 3: Reshape for matmul
# patches: (B, H_out*W_out, C_in*K*K)
patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, H_out, W_out, C_in, K, K)
patches = patches.reshape(B, H_out * W_out, C_in * K * K)

# kernel: (C_out, C_in*K*K)
kernel_flat = kernel.reshape(C_out, -1)

# Step 4: Matrix multiply
# (B, H_out*W_out, C_in*K*K) @ (C_in*K*K, C_out) -> (B, H_out*W_out, C_out)
out = patches @ kernel_flat.T

# Step 5: Reshape back to image
out = out.permute(0, 2, 1).reshape(B, C_out, H_out, W_out)

print("Matmul output shape:", out.shape)

# === Verify against F.conv2d ===
out_conv = F.conv2d(x, kernel, padding=padding)

print("Conv2d output shape:", out_conv.shape)
print("Max diff:", (out - out_conv).abs().max().item())
