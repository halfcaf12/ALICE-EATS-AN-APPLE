import torch
import numpy as np
import faiss

# Test of faiss sensitivity to utter insanity, consistently causes abort across faiss versions tested (>=1.6.0)

# Ensure we're using GPU for FAISS and PyTorch
faiss.omp_set_num_threads(1)  # Limit FAISS to 1 thread; adjust as needed
torch.set_num_threads(1)  # Ensure PyTorch doesn't spawn too many threads

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simulate creating embeddings for 7837 hits, each with 8 dimensions (features)
# Assuming this is the kind of data that would be fed into FAISS for edge construction
embeddings = torch.randn(7837, 8, device=device)

# Convert PyTorch tensor to NumPy array as FAISS works with CPU-based NumPy arrays
embeddings_np = embeddings.cpu().numpy()

# Initialize a FAISS index for L2 distance, to simulate finding nearest neighbors
d = 8  # Dimensionality of each embedding
index = faiss.IndexFlatL2(d)
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()  # Use default GPU resources for FAISS
    index = faiss.index_cpu_to_gpu(res, 0, index)

# Add our embeddings to the index
index.add(embeddings_np)

# Attempt a FAISS operation that might cause the issue
# For example, finding the 10 nearest neighbors for each embedding
D, I = index.search(embeddings_np, 10)

# Check the result shapes
print(f"Distances shape: {D.shape}, Indices shape: {I.shape}")


