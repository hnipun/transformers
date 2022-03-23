import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.multi_head_attention = MultiHeadAttentionLayer()
        self.feed_forward_layer = FeedForwardLayer()

    def forward(self, x):  # (batch_size, seq_length, emd_dim)
        x = self.multi_head_attention(x)
        # logger.log(f'multihead attention layer {x.size()}')
        return self.feed_forward_layer(x)
