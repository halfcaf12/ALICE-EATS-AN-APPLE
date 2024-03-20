import torch
from gnn_classes import LayerlessEmbedding  # Adjust the import path as necessary

# Updated hparams based on your provided configuration
hparams = {
    "adjacent": False,
    "clustering": "build_edges",  # Ensure this matches a valid function name in utils_torch
    "emb_dim": 8,
    "emb_hidden": 512,
    "endcaps": True,
    "factor": 0.3,
    "in_channels": 12,  # Match this with the dimensionality of your dummy input
    "knn_train": 20,
    "knn_val": 100,
    "layerless": True,
    "layerwise": False,
    "lr": 0.002,
    "margin": 1,
    "n_workers": 1,
    "nb_layer": 6,
    "noise": False,
    "overwrite": True,
    "patience": 5,
    "pt_min": 0,
    "r_train": 1,
    "r_val": 1.0,
    "randomisation": 2,
    "regime": ["rp", "hnm", "ci"],
    "train_split": [8, 1, 1],
    "warmup": 500,
    "weight": 4
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize your model with the updated hparams
model = LayerlessEmbedding(hparams).to(device)

# Create a dummy input tensor matching the expected input dimensions
dummy_input = torch.randn(1, hparams["in_channels"], device=device)  # Batch size of 1

# Attempt a forward pass with the dummy input
try:
    dummy_output = model(dummy_input)
    print("Forward pass successful. Output shape:", dummy_output.shape)
except RuntimeError as e:
    print("Error during forward pass:", e)

