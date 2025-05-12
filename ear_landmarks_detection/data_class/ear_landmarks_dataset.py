import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class EarLandmarksDataset(Dataset):

    def __init__(self, dataset, image_size, mean, std):
        """
        Initializes the EarDetectionDataset object by setting the images and landmarks
        from the input dataset.

        Parameters
        ----------
        dataset : dict
            The dataset containing the gesture sequence samples and their corresponding labels.
        """

        self.images = dataset['images']
        self.landmarks_list = dataset['landmarks_list']

        # convert PIL image to torch tensor
        if mean == None and std == None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=Image.Resampling.LANCZOS),    # Oppure lasciare interpolation di default
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=Image.Resampling.LANCZOS),    # Oppure lasciare interpolation di default
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
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
            A dictionary containing the image and landmarks of the sample.
            The keys are 'image' for the gesture data and 'landmarks' for the corresponding label.
        """

        image = self.images[index]
        landmarks = self.landmarks_list[index]

        landmarks = np.array(landmarks)

        torch_image = self.transform(image)

        # Convert annotation to torch tensor
        torch_landmarks = torch.tensor(landmarks, dtype=torch.float32)

        item = {
            'image': torch_image,
            'landmarks': torch_landmarks
        }

        return item