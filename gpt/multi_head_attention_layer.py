import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm_layer = nn.LayerNorm(EMBEDDING_DIM)

        self.key_linear = nn.Linear(EMBEDDING_DIM, NUM_ATTENTION_HEADS * NUM_FEATURES)
        self.query_linear = nn.Linear(EMBEDDING_DIM, NUM_ATTENTION_HEADS * NUM_FEATURES)
        self.value_linear = nn.Linear(EMBEDDING_DIM, NUM_ATTENTION_HEADS * NUM_FEATURES)

        self.softmax = nn.Softmax(dim=2)
        self.linear_layer = nn.Linear(NUM_ATTENTION_HEADS * NUM_FEATURES, EMBEDDING_DIM)

    def forward(self, x):
        x_residual = x
        x = self.norm_layer(x)
        seq_length = x.shape[1]
        kx = self.key_linear(x)
        # logger.log(f'kx layer {kx.size()}')
        kx = kx.view(BATCH_SIZE, seq_length, NUM_ATTENTION_HEADS, NUM_FEATURES)
        # logger.log(f'kx layer {kx.size()}')

        qx = self.query_linear(x)
        qx = kx.view(BATCH_SIZE, seq_length, NUM_ATTENTION_HEADS, NUM_FEATURES)

        vx = self.value_linear(x)
        vx = kx.view(BATCH_SIZE, seq_length, NUM_ATTENTION_HEADS, NUM_FEATURES)

        score = torch.einsum('bihd,bjhd ->bijh', qx, kx)
        score = score / (NUM_FEATURES) ** 0.5

        # logger.log(f'score layer {score.size()}')

        mask = get_mask(seq_length).to(x.device)
        score = score.masked_fill(mask, float("-inf"))

        probs = self.softmax(score)

        # logger.log(f'probs layer {probs.size()}')

        output = torch.einsum('bijh,bjhd ->bihd', probs, vx)
        # logger.log(f'output layer {output.size()}')
        output = output.reshape(BATCH_SIZE, seq_length, NUM_ATTENTION_HEADS * NUM_FEATURES)
        # logger.log(f'output layer {output.size()}')
        output = self.linear_layer(output)
        # logger.log(f'output layer {output.size()}')

        # logger.log(f'kx layer {kx.size()}')

        return x + x_residual
