import torch.nn as nn

from gpt import CONFIGS

EMBEDDING_DIM = CONFIGS['embedding_dim']


class FeedForwardLayer(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_size = EMBEDDING_DIM * 4

        self.norm_layer = nn.LayerNorm(EMBEDDING_DIM)
        self.linear_layer1 = nn.Linear(EMBEDDING_DIM, hidden_size)
        self.activation_layer = nn.ReLU()
        self.linear_layer2 = nn.Linear(hidden_size, EMBEDDING_DIM)

    def forward(self, x):
        x_norm = self.norm_layer(x)

        output = self.linear_layer1(x_norm)
        output = self.activation_layer(output)
        output = self.linear_layer2(output)

        return output + x  # x or x_norm ?
