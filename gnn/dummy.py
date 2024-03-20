import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

import os
import tempfile

# Test CUDA, PyTorch compatability on weaker instances of embedding classes

class EnhancedEmbeddingModel(pl.LightningModule):
    def __init__(self, in_channels=10, emb_hidden=32, emb_dim=16, nb_layer=3, use_norm=True, activation_fn=nn.Tanh):
        super().__init__()
        self.use_norm = use_norm
        layers = [nn.Linear(in_channels, emb_hidden)]
        layers.extend([nn.Linear(emb_hidden, emb_hidden) for _ in range(nb_layer - 1)])
        self.layers = nn.ModuleList(layers)
        self.emb_layer = nn.Linear(emb_hidden, emb_dim)
        self.norm = nn.LayerNorm(emb_hidden)
        self.act = activation_fn()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.use_norm:
                x = self.norm(x)
            x = self.act(x)
        x = self.emb_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)

class ComprehensiveInferenceCallback(Callback):
    def __init__(self):
        super().__init__()
        self.output_dir = tempfile.mkdtemp()

    def on_train_end(self, trainer, pl_module):
        print("Comprehensive Inference Started.")
        # Simulate processing multiple batches with more complex logic
        for i in range(3):  # Simulate 3 batches
            dummy_data = torch.randn(5, 10)
            dummy_data = dummy_data.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                inference_output = pl_module(dummy_data)
            transformed_output = self.transform_and_process_data(inference_output, pl_module)
            output_path = os.path.join(self.output_dir, f"comprehensive_inference_batch_{i}.pt")
            torch.save(transformed_output, output_path)
            print(f"Saved comprehensive batch {i} inference output to {output_path}.")
        print("Comprehensive Inference Completed.")

    def transform_and_process_data(self, data, pl_module):
        return data


# Create dummy data
x = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randn(100, 16)  # 100 samples, target embeddings of dim 16

dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=10)

model = EnhancedEmbeddingModel()
callback = ComprehensiveInferenceCallback()

trainer = Trainer(
    max_epochs=2,
    callbacks=[callback],
    gpus=1 if torch.cuda.is_available() else 0,
    limit_train_batches=10 
)

trainer.fit(model, train_loader)

import scipy as sp
import numpy as np

from torch_cluster import radius_graph

device = 'cuda'

def graph_intersection(pred_graph, truth_graph):
    """
    Use sparse representation to compare the predicted graph
    and the truth graph so as to label the edges in the predicted graph
    to be 1 as true and 0 as false.
    """
    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    l1 = pred_graph.cpu().numpy()
    l2 = truth_graph.cpu().numpy()
    e_1 = sp.sparse.coo_matrix((np.ones(l1.shape[1]), l1), shape=(array_size, array_size)).tocsr()
    e_2 = sp.sparse.coo_matrix((np.ones(l2.shape[1]), l2), shape=(array_size, array_size)).tocsr()
    e_intersection = (e_1.multiply(e_2) - ((e_1 - e_2)>0)).tocoo()

    new_pred_graph = torch.from_numpy(np.vstack([e_intersection.row, e_intersection.col])).long().to(device)
    y = e_intersection.data > 0

    return new_pred_graph, y

def test_graph_intersection():
    # Dummy data
    pred_graph = torch.tensor([[0, 1, 2], [2, 3, 0]], device=device)
    truth_graph = torch.tensor([[0, 2, 3], [3, 0, 2]], device=device)
    
    new_pred_graph, y = graph_intersection(pred_graph, truth_graph)
    print(f"New Predicted Graph: {new_pred_graph}")
    print(f"Labels: {y}")

# Call the test function to see if it executes without errors
test_graph_intersection()

# Create dummy tensors of the specified sizes
A = torch.randn(1024, 8, device=device)
B = torch.randn(7837, 8, device=device)

# Perform the matrix multiplication
# Since B needs to be transposed to match the inner dimensions, we use B.T
result = torch.matmul(A, B.T)

print(f"Result shape: {result.shape}")
