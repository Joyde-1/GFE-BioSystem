import torch
import numpy as np
from torch.utils.data import Dataset


class GaitEmbeddingExtractionDataset(Dataset):
    def __init__(self, dataset):
        self.keypoints_sequences = dataset['keypoints_sequences']
        self.labels = dataset['labels']

    def __len__(self):
        return len(self.keypoints_sequences)

    def __getitem__(self, index):
        np_keypoints_sequence = self.keypoints_sequences[index]           # numpy (T,17,2)
        np_label = self.labels[index]
    
        # Convert sequences to float32
        np_keypoints_sequence = np.array(np_keypoints_sequence, dtype=np.float32)  # Convert sequences to float32

        # Convert sequence to torch tensor
        torch_sequence = torch.from_numpy(np_keypoints_sequence).float()

        # Convert labels to torch tensor
        torch_label = torch.tensor(np_label, dtype=torch.long)

        item = {
            'keypoints_sequence': torch_sequence,
            'label': torch_label
        }

        return item