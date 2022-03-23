import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from labml import monit

from data_loader.dataset import TextDataset
from gpt import CONFIGS
from gpt.transformer import Transformer

DEVICE = torch.device('cuda')
BATCH_SIZE = CONFIGS['batch_size']
NUM_EPOCHS = CONFIGS['num_epochs']
SEQ_LENGTH = CONFIGS['seq_length']
LEARNING_RATE = CONFIGS['learning_rate']
SHUFFLE_TRAIN_DATA = CONFIGS['shuffle_train_data']

train_data_loader = DataLoader(TextDataset(), batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN_DATA)
model = Transformer().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train():
    for epoch in monit.loop(NUM_EPOCHS):
        for train_data in monit.iterate('train', train_data_loader):
            train_data = train_data.to(DEVICE)

            if train_data.size()[0] < 64:  # remove this
                continue

            optimizer.zero_grad()
            out = model(train_data[:, :-1])
            out = out.to(DEVICE)

            loss = F.cross_entropy(out.view(BATCH_SIZE * SEQ_LENGTH, 65),
                                   train_data[:, 1:].reshape(BATCH_SIZE * SEQ_LENGTH))
            loss.backward()
            optimizer.step()
