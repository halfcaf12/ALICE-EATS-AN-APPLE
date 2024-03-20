import torch

# Assuming embeddings and another matrix (let's call it B for this example) are your inputs
# Simulate creating embeddings (512, 8) and B (7837, 8) as random tensors for demonstration
embeddings = torch.randn(512, 8, device='cuda')
B = torch.randn(7837, 8, device='cuda')

# Transpose B to match the dimensions for matrix multiplication
B_t = B.t()  # Now B_t is (8, 7837)

# Initialize an empty tensor for the result
result = torch.empty((embeddings.shape[0], B_t.shape[1]), device='cuda')

# Determine the chunk size for B (how many columns to take in each step)
chunk_size = 1024  # This is a parameter you might need to adjust based on your GPU's capacity

# Perform the multiplication in chunks
for i in range(0, B_t.shape[1], chunk_size):
    end = min(i + chunk_size, B_t.shape[1])
    # Multiply embeddings with a chunk of B_t
    result[:, i:end] = torch.matmul(embeddings, B_t[:, i:end])

print(f"Completed matrix multiplication, result shape: {result.shape}")

