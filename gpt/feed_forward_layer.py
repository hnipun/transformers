import torch.nn as nn


class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        hidden_size = embedding_dim * 4

        self.norm_layer = nn.LayerNorm(embedding_dim)
        self.linear_layer1 = nn.Linear(embedding_dim, hidden_size)
        self.activation_layer = nn.ReLU()
        self.linear_layer2 = nn.Linear(hidden_size, embedding_dim)

    def forward(self, x):
        x_norm = self.norm_layer(x)

        output = self.linear_layer1(x_norm)
        output = self.activation_layer(output)
        output = self.linear_layer2(output)

        return output + x  # x or x_norm ?
