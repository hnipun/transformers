import torch
from torch.utils.data import Dataset
import labml.utils.download
from labml import lab

SEQ_LENGTH = 128


class TextDataset(Dataset):
    def __init__(self, seq_length: int = SEQ_LENGTH):
        self.seq_length = seq_length

        labml.utils.download.download_file(
            'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
            lab.get_data_path() / 'tiny_shakespeare.txt',
        )
        with open(lab.get_data_path() / 'tiny_shakespeare.txt', "r") as f:
            self.text = f.read()

        self.text = list(self.text)
        self.vocab = {t: ids for ids, t in enumerate(set(self.text))}
        self.text_tensor = torch.tensor(self._to_vocab(self.text))

    def _to_vocab(self, text):
        return [self.vocab[t] for t in text]

    def __len__(self):
        return len(self.text) // self.seq_length

    def __getitem__(self, idx):
        return self.text_tensor[self.seq_length * idx:self.seq_length * (idx + 1)]
