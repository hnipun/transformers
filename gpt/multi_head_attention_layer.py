import torch
import torch.nn as nn

from gpt import CONFIGS

BATCH_SIZE = CONFIGS['batch_size']
NUM_EMBEDDINGS = CONFIGS['num_embeddings']
EMBEDDING_DIM = CONFIGS['embedding_dim']
N_TRANSFORMER_LAYERS = CONFIGS['n_transformer_layers']
NUM_ATTENTION_HEADS = CONFIGS['num_attention_heads']
NUM_ATTENTION_FEATURES = CONFIGS['num_attention_features']


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm_layer = nn.LayerNorm(EMBEDDING_DIM)

        self.key_linear = nn.Linear(EMBEDDING_DIM, NUM_ATTENTION_HEADS * NUM_ATTENTION_FEATURES)
        self.query_linear = nn.Linear(EMBEDDING_DIM, NUM_ATTENTION_HEADS * NUM_ATTENTION_FEATURES)
        self.value_linear = nn.Linear(EMBEDDING_DIM, NUM_ATTENTION_HEADS * NUM_ATTENTION_FEATURES)

        self.softmax = nn.Softmax(dim=2)
        self.linear_layer = nn.Linear(NUM_ATTENTION_HEADS * NUM_ATTENTION_FEATURES, EMBEDDING_DIM)

    def forward(self, x):
        seq_length = x.shape[1]

        x_norm = self.norm_layer(x)  # pre-norm layer

        kx = self.key_linear(x_norm)
        kx = kx.view(BATCH_SIZE, seq_length, NUM_ATTENTION_HEADS, NUM_ATTENTION_FEATURES)

        qx = self.query_linear(x_norm)
        qx = qx.view(BATCH_SIZE, seq_length, NUM_ATTENTION_HEADS, NUM_ATTENTION_FEATURES)

        vx = self.value_linear(x_norm)
        vx = vx.view(BATCH_SIZE, seq_length, NUM_ATTENTION_HEADS, NUM_ATTENTION_FEATURES)

        score = torch.einsum('bihd,bjhd ->bijh', qx, kx)
        score = score / (NUM_ATTENTION_FEATURES ** 0.5)

        mask = self._get_mask(seq_length).to(x.device)
        score = score.masked_fill(mask, float("-inf"))

        probs = self.softmax(score)

        output = torch.einsum('bijh,bjhd ->bihd', probs, vx)
        output = output.reshape(BATCH_SIZE, seq_length, NUM_ATTENTION_HEADS * NUM_ATTENTION_FEATURES)
        output = self.linear_layer(output)

        return output + x  # x +  x

    @staticmethod
    def _get_mask(seq_length: int):  # should be batch_size, seq_length, seq_length, heads
        # 1, seq_length, seq_length, 1 is enough

        res = torch.ones(seq_length, seq_length, dtype=bool)
        res = torch.triu(res, diagonal=1)

        res = torch.unsqueeze(res, 0)
        res = torch.unsqueeze(res, -1)

        return res
