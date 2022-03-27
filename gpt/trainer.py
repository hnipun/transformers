import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from labml import monit
from labml import tracker, experiment, logger

from data_loader.dataset import TextDataset
from gpt import CONFIGS
from gpt.transformer import Transformer

SEED = CONFIGS['seed']
DEVICE = torch.device('cuda')

BATCH_SIZE = CONFIGS['batch_size']
SEQ_LENGTH = CONFIGS['seq_length']
NUM_EMBEDDINGS = CONFIGS['num_embeddings']

EMBEDDING_DIM = CONFIGS['embedding_dim']
N_TRANSFORMER_LAYERS = CONFIGS['n_transformer_layers']
NUM_ATTENTION_HEADS = CONFIGS['num_attention_heads']
NUM_WEIGHTS = CONFIGS['num_weights']

NUM_EPOCHS = CONFIGS['num_epochs']
LEARNING_RATE = CONFIGS['learning_rate']
TRAIN_LOG_INTERVAL = CONFIGS['train_log_interval']

train_data_loader = DataLoader(TextDataset(), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
model = Transformer(num_embeddings=NUM_EMBEDDINGS,
                    embedding_dim=EMBEDDING_DIM,
                    n_transformer_layers=N_TRANSFORMER_LAYERS,
                    num_attention_heads=NUM_ATTENTION_HEADS,
                    num_weights=NUM_WEIGHTS
                    ).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train():
    for batch_idx, train_data in monit.enum('train', train_data_loader):
        train_data = train_data.to(DEVICE)

        optimizer.zero_grad()
        out = model(train_data[:, :-1])
        out = out.to(DEVICE)

        loss = F.cross_entropy(out.view(BATCH_SIZE * SEQ_LENGTH, NUM_EMBEDDINGS),
                               train_data[:, 1:].reshape(BATCH_SIZE * SEQ_LENGTH))
        loss.backward()
        optimizer.step()

        tracker.add_global_step()
        tracker.add({'loss.train': loss})

        if batch_idx % TRAIN_LOG_INTERVAL == 0:
            tracker.save()


def main():
    torch.manual_seed(SEED)

    experiment.create(name='gpt')
    experiment.configs(CONFIGS)
    experiment.add_pytorch_models(dict(model=model))

    with experiment.start():
        for epoch in range(1, NUM_EPOCHS + 1):
            train()
            logger.log()


if __name__ == '__main__':
    main()
