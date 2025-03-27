import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FaceDetectionDataset(Dataset):

    def __init__(self, dataset, image_size, mean, std):
        """
        Initializes the GestureRecognitionDataset object by setting the sequences and labels
        from the input dataset.

        Parameters
        ----------
        dataset : dict
            The dataset containing the gesture sequence samples and their corresponding labels.
        """

        self.images = dataset['images']
        self.bounding_boxes = dataset['bounding_boxes']

        # convert PIL image to torch tensor
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.Resampling.LANCZOS),    # Oppure lasciare interpolation di default
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """

        return len(self.images) 

    def __getitem__(self, index):
        """
        Retrieves a sample and its label from the dataset at the specified index.

        Parameters
        ----------
        index : int
            The index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing the sequence and label of the sample.
            The keys are 'sequence' for the gesture data and 'label' for the corresponding label.
        """

        image = self.images[index]
        bounding_box = self.bounding_boxes[index]

        bounding_box = np.array(bounding_box)

        torch_image = self.transform(image)

        # Convert annotation to torch tensor
        torch_bounding_box = torch.tensor(bounding_box, dtype=torch.float32)

        item = {
            'image': torch_image,
            'bounding_box': torch_bounding_box
        }

        return item