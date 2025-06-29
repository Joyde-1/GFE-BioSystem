import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


class GaitEmbeddingExtractionDataset(Dataset):
    def __init__(self, dataset, transform=None, flatten=True):
        """
        Parameters
        ----------
        dataset : dict
            {'keypoints_sequences': list/array (N, L, 17, 2),
             'labels'            : list/array (N,) }
            Le sequenze devono essere già normalizzate e
            portate alla lunghezza fissa L (es. 32).
        transform : callable or None
            Pipeline di data-augmentation da applicare **solo in training**.
        flatten : bool
            Se True restituisce shape (L, 34) per il modello fully-connected;
            altrimenti mantiene (L, 17, 2) (utile per ST-GCN, ecc.).
        """
        self.seq   = dataset['keypoints_sequences']
        self.labels = dataset['labels']
        self.transform = transform
        self.flatten = flatten

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq_np   = np.asarray(self.seq[idx], dtype=np.float32)   # (L, J, 2)
        label_np = self.labels[idx]

        seq = torch.from_numpy(seq_np)                           # Tensor

        # ---- data-augmentation (solo se definita) --------------
        if self.transform is not None:
            seq = self.transform(seq).float()

        # ---- eventuale flatten ---------------------------------
        if self.flatten:                                         # (L,34)
            seq = seq.view(seq.shape[0], -1)

        item = {
            'keypoints_sequence': seq,                           # float32
            'label'            : torch.tensor(label_np, dtype=torch.long)
        }
        return item