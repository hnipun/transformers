import torch.nn as nn

from gpt.transformer_layer import TransformerLayer


class Transformer(nn.Module):
    def __init__(self, num_embeddings: int,
                 embedding_dim: int,
                 n_transformer_layers: int,
                 num_attention_heads: int,
                 num_weights: int):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.transformer_layers = nn.ModuleList([TransformerLayer(embedding_dim=embedding_dim,
                                                                  num_attention_heads=num_attention_heads,
                                                                  num_weights=num_weights
                                                                  )] * n_transformer_layers)
        self.readout_layer = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, x):
        # x -> (batch_size, seq_length)
        x = self.embeddings(x)
        # x -> (batch_size, seq_length, embedding_dim)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        # x -> (batch_size, seq_length, embedding_dim)

        return self.readout_layer(x)  # -> (batch_size, seq_length, num_embeddings)
