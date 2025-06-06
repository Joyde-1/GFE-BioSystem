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

        T, F = np_keypoints_sequence.shape

        if F == 17 * 3:
            np_keypoints_sequence = np_keypoints_sequence.reshape(T, 17, 3).astype(np.float32)  # Convert to (T, 17, 3)
        else:
            raise ValueError(f"Unexpected number of features: {F}. Expected 34 or 51.")

        # Convert sequences to float32
        np_keypoints_sequence = np.array(np_keypoints_sequence, dtype=np.float32)  # Convert sequences to float32

        # Convert sequence to torch tensor
        torch_sequence = torch.from_numpy(np_keypoints_sequence)

        torch_sequence = torch_sequence.permute(2, 0, 1).contiguous()  # now (C, T, 17)

        # Convert labels to torch tensor
        torch_label = torch.tensor(np_label - 1, dtype=torch.long)

        item = {
            'keypoints_sequence': torch_sequence,
            'label': torch_label
        }

        return item