import torch.nn as nn

from gpt.transformer_layer import TransformerLayer
from gpt import CONFIGS

NUM_EMBEDDINGS = CONFIGS['num_embeddings']
EMBEDDING_DIM = CONFIGS['embedding_dim']
N_TRANSFORMER_LAYERS = CONFIGS['n_transformer_layers']


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(NUM_EMBEDDINGS, EMBEDDING_DIM)
        self.transformer_layers = nn.ModuleList([TransformerLayer()] * N_TRANSFORMER_LAYERS)
        self.readout_layer = nn.Linear(EMBEDDING_DIM, NUM_EMBEDDINGS)

    def forward(self, x):
        # x -> (batch_size, seq_length)
        x = self.embeddings(x)
        # x -> (batch_size, seq_length, embedding_dim)
        for transformer_layer in self.transormer_layers:
            x = transformer_layer(x)
        # x -> (batch_size, seq_length, emd_dim)

        return self.readout_layer(x)  # -> (batch_size, seq_length, num_embeddings)
