import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_attention_heads: int, num_weights: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.num_weights = num_weights

        self.norm_layer = nn.LayerNorm(self.embedding_dim)

        self.key_linear = nn.Linear(self.embedding_dim, self.num_attention_heads * num_weights)
        self.query_linear = nn.Linear(self.embedding_dim, self.num_attention_heads * num_weights)
        self.value_linear = nn.Linear(self.embedding_dim, self.num_attention_heads * num_weights)

        self.softmax = nn.Softmax(dim=2)  # dim should be 1 ?
        self.linear_layer = nn.Linear(self.num_attention_heads * num_weights, self.embedding_dim)

    def forward(self, x):
        batch_size, seq_length = x.shape[0], x.shape[1]
        # x -> (batch_size, seq_length, embedding_dim)
        x_norm = self.norm_layer(x)  # pre-norm layer
        # x_norm -> (batch_size, seq_length, embedding_dim)
        kx = self.key_linear(x_norm)
        # kx -> (batch_size, seq_length, num_attention_heads * num_attention_features)
        kx = kx.view(batch_size, seq_length, self.num_attention_heads, self.num_weights)
        # kx -> (batch_size, seq_length, num_attention_heads, num_attention_features)

        qx = self.query_linear(x_norm)
        qx = qx.view(batch_size, seq_length, self.num_attention_heads, self.num_weights)

        vx = self.value_linear(x_norm)
        vx = vx.view(batch_size, seq_length, self.num_attention_heads, self.num_weights)

        score = torch.einsum('bihd,bjhd ->bijh', qx, kx)
        # score -> (batch_size, seq_length, seq_length, num_attention_heads)
        score = score / (self.num_weights ** 0.5)  # reduce the variance, thus less spikes in probs, better gradients

        mask = self._get_mask(seq_length).to(x.device)
        # mask -> (1, seq_length, seq_length, 1)
        score = score.masked_fill(mask, float("-inf"))
        # score -> (batch_size, seq_length, seq_length, num_attention_heads)

        probs = self.softmax(score)
        # score -> (batch_size, seq_length, seq_length, num_attention_heads)

        output = torch.einsum('bijh,bjhd ->bihd', probs, vx)
        # output -> (batch_size, seq_length, num_attention_heads, num_attention_features)
        output = output.reshape(batch_size, seq_length, self.num_attention_heads * self.num_weights)  # concat
        # output -> (batch_size, seq_length, num_attention_heads * num_attention_features)
        output = self.linear_layer(output)
        # output -> (batch_size, seq_length, embedding_dim)

        return output + x  # x or x_norm ?

    @staticmethod
    def _get_mask(seq_length: int):  # should be batch_size, seq_length, seq_length, num_attention_heads
        # 1, seq_length, seq_length, 1 is enough

        res = torch.ones(seq_length, seq_length, dtype=bool)
        res = torch.triu(res, diagonal=1)

        res = torch.unsqueeze(res, 0)
        res = torch.unsqueeze(res, -1)

        return res
