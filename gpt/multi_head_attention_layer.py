import torch
import torch.nn as nn

from gpt import CONFIGS

EMBEDDING_DIM = CONFIGS['embedding_dim']
NUM_ATTENTION_HEADS = CONFIGS['num_attention_heads']
NUM_WEIGHTS = CONFIGS['num_weights']


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm_layer = nn.LayerNorm(EMBEDDING_DIM)

        self.key_linear = nn.Linear(EMBEDDING_DIM, NUM_ATTENTION_HEADS * NUM_WEIGHTS)
        self.query_linear = nn.Linear(EMBEDDING_DIM, NUM_ATTENTION_HEADS * NUM_WEIGHTS)
        self.value_linear = nn.Linear(EMBEDDING_DIM, NUM_ATTENTION_HEADS * NUM_WEIGHTS)

        self.softmax = nn.Softmax(dim=2)  # dim should be 1 ?
        self.linear_layer = nn.Linear(NUM_ATTENTION_HEADS * NUM_WEIGHTS, EMBEDDING_DIM)

    def forward(self, x):
        batch_size, seq_length = x.shape[0], x.shape[1]
        # x -> (batch_size, seq_length, embedding_dim)
        x_norm = self.norm_layer(x)  # pre-norm layer
        # x_norm -> (batch_size, seq_length, embedding_dim)
        kx = self.key_linear(x_norm)
        # kx -> (batch_size, seq_length, num_attention_heads * num_attention_features)
        kx = kx.view(batch_size, seq_length, NUM_ATTENTION_HEADS, NUM_WEIGHTS)
        # kx -> (batch_size, seq_length, num_attention_heads, num_attention_features)

        qx = self.query_linear(x_norm)
        qx = qx.view(batch_size, seq_length, NUM_ATTENTION_HEADS, NUM_WEIGHTS)

        vx = self.value_linear(x_norm)
        vx = vx.view(batch_size, seq_length, NUM_ATTENTION_HEADS, NUM_WEIGHTS)

        score = torch.einsum('bihd,bjhd ->bijh', qx, kx)
        # score -> (batch_size, seq_length, seq_length, num_attention_heads)
        score = score / (NUM_WEIGHTS ** 0.5)

        mask = self._get_mask(seq_length).to(x.device)
        # mask -> (1, seq_length, seq_length, 1)
        score = score.masked_fill(mask, float("-inf"))
        # score -> (batch_size, seq_length, seq_length, num_attention_heads)

        probs = self.softmax(score)
        # score -> (batch_size, seq_length, seq_length, num_attention_heads)

        output = torch.einsum('bijh,bjhd ->bihd', probs, vx)
        # output -> (batch_size, seq_length, num_attention_heads, num_attention_features)
        output = output.reshape(batch_size, seq_length, NUM_ATTENTION_HEADS * NUM_WEIGHTS)  # concat
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
