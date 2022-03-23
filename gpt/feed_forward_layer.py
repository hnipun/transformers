import torch.nn as nn


class FeedForwardLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # self.norm_layer = nn.LayerNorm(EMBEDDING_DIM)
        # self.linear_layer1 = nn.Linear(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_DIM / 2)
        # self.activation_layer = nn.ReLU()
        # self.linear_layer2 = nn.Linear(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_DIM)

    def forward(self, x):
        # x_residual = x
        #
        # x = self.norm_layer(x)
        # x = self.linear_layer1(x)
        # x = self.activation_layer(x)
        # x = self.linear_layer2(x)
        #
        # return x + x_residual
        return x
