import cv2
from tqdm import tqdm
import os
import shutil
import random
from PIL import Image
from sklearn.model_selection import train_test_split


class PrepareData():
    """
    Prepares detection data for deep learning model.

    Attributes
    ----------
    data_paths : str
        Base directory where detection data is stored.
    data : Dict
        This dictionary holds the processed data. It contains two keys: 'images' and 'labels'.
    """

    def __init__(self, yolo_detection_config, biometric_trait):
        """
        Initializes the PrepareData instance

        Parameters
        ----------
        data_paths : str
            The path to the base directory containing database
        """
        
        self.yolo_detection_config = yolo_detection_config
        self.biometric_trait = biometric_trait

    def _get_image_names(self):
        images_path = []
        images_file = []

        data = {
            'image_names': [],
            'subjects': []
        }

        images_path = os.path.join(self.yolo_detection_config.data_dir, self.biometric_trait)

        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Directory {images_path} not found.")

        images_file = [f for f in os.listdir(images_path) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.png')]

        # tqdm is used to show the progress barr
        for image_file in tqdm(images_file, desc=f"Loading {self.biometric_trait} image names", unit="file"):
            if image_file.endswith(".bmp") or image_file.endswith('.jpg') or image_file.endswith('.png'):
                # Carica l'immagine
                image_path = os.path.join(images_path, image_file)

                image_name = os.path.basename(os.path.splitext(image_path)[0])

                data['image_names'].append(image_name)

                # Extract the subject number from the image name
                data['subjects'].append(image_name.split('_')[0])

        return images_path, images_file, data
    
    def _data_splitting(self, data):
        """
        Splits the training dataset into training and validation sets based on the configuration.

        Parameters
        ----------
        train_set : dict
            Dictionary containing 'sequences' and 'labels' as keys.
        config : dict
            Configuration object with parameters for splitting, including 'val_size' which defines the proportion of the validation set.

        Returns
        -------
        tuple of dict
            Returns two dictionaries, the first containing the training data and labels, and the second containing the validation data and labels.
        """

        train_set_image_names, test_set_image_names, train_set_subjects, _ = train_test_split(data['image_names'], data['subjects'], test_size=self.yolo_detection_config.data.test_size, random_state=42, shuffle=True, stratify=data['subjects'])
        train_set_image_names, val_set_image_names, _, _ = train_test_split(train_set_image_names, train_set_subjects, test_size=self.yolo_detection_config.data.val_size, random_state=42, shuffle=True, stratify=train_set_subjects)

        return train_set_image_names, val_set_image_names, test_set_image_names
    
    def _data_augmentatation(self, image):
        if random.random() < 0.5:
            alpha = 1.03
            beta = 3
        else:
            alpha = 0.97
            beta = -3
        image = cv2.convertScaleAbs(image.copy(), alpha=alpha, beta=beta)
        return image

    def _load_and_save_images_and_bounding_boxes(self, images_path, images_file, train_set_image_names, val_set_image_names):
        # Percorsi di salvataggio
        for subset in ['train', 'val', 'test']:
            subset_path = os.path.join(self.yolo_detection_config.save_data_splitted_path, "splitted database", self.biometric_trait, subset)

            os.makedirs(subset_path, exist_ok=True)
    
            os.makedirs(os.path.join(subset_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(subset_path, "labels"), exist_ok=True)

        for image_file in tqdm(images_file, desc=f"Loading {self.biometric_trait} image files", unit="file"):
            if image_file.endswith(".bmp") or image_file.endswith('.jpg') or image_file.endswith('.png'):
                # Carica l'immagine
                image_path = os.path.join(images_path, image_file)

                image_name = os.path.basename(os.path.splitext(image_path)[0])

                bounding_box_path = os.path.join(self.yolo_detection_config.bounding_boxes_dir, self.biometric_trait, image_name + '.txt')

                # image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                if image is None:
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                if image_name in train_set_image_names or image_name in val_set_image_names:
                    image = self._data_augmentatation(image.copy())

                # Converti da BGR (OpenCV) a RGB (Pillow)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = Image.fromarray(image).resize((self.yolo_detection_config.data.image_size, self.yolo_detection_config.data.image_size))

                # Determina il subset (train, val, test)
                if image_name in train_set_image_names:
                    subset = 'train'
                elif image_name in val_set_image_names:
                    subset = 'val'
                else:
                    subset = 'test'

                # Salva l'immagine e la bounding box nella directory corrispondente
                subset_image_path = os.path.join(self.yolo_detection_config.save_data_splitted_path, "splitted database", self.biometric_trait, subset, "images", f"{image_name}.bmp")
                subset_bounding_box_path = os.path.join(self.yolo_detection_config.save_data_splitted_path, "splitted database", self.biometric_trait, subset, "labels", f"{image_name}.txt")
                
                # Salva l'immagine nella directory corrispondente
                image.save(subset_image_path)

                # Salva la bounding box nel file corrispondente
                shutil.copy2(bounding_box_path, subset_bounding_box_path)
    
    # Funzione per caricare le immagini
    def prepare_data(self):
        images_path, images_file, data = self._get_image_names()

        train_set_image_names, val_set_image_names, test_set_image_names = self._data_splitting(data)
        
        self._load_and_save_images_and_bounding_boxes(images_path, images_file, train_set_image_names, val_set_image_names)