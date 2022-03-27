import torch.nn as nn

from gpt.multi_head_attention_layer import MultiHeadAttentionLayer
from gpt.feed_forward_layer import FeedForwardLayer


class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_attention_heads: int, num_weights: int):
        super().__init__()

        self.multi_head_attention_layer = MultiHeadAttentionLayer(embedding_dim=embedding_dim,
                                                                  num_attention_heads=num_attention_heads,
                                                                  num_weights=num_weights
                                                                  )
        self.feed_forward_layer = FeedForwardLayer(embedding_dim=embedding_dim)

    def forward(self, x):
        # x -> (batch_size, seq_length, embedding_dim)
        x = self.multi_head_attention_layer(x)
        # x -> (batch_size, seq_length, embedding_dim)

        return self.feed_forward_layer(x)
