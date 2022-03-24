import torch.nn as nn

from gpt.multi_head_attention_layer import MultiHeadAttentionLayer
from gpt.feed_forward_layer import FeedForwardLayer


class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.multi_head_attention_layer = MultiHeadAttentionLayer()
        self.feed_forward_layer = FeedForwardLayer()

    def forward(self, x):
        # x -> (batch_size, seq_length, embedding_dim)
        x = self.multi_head_attention_layer(x)
        # x -> (batch_size, seq_length, embedding_dim)

        return self.feed_forward_layer(x)
