import cv2
from tqdm import tqdm
import os
import shutil
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


class PrepareData():
    """
    Prepares face detection data for deep learning model.

    Attributes
    ----------
    data_paths : str
        Base directory where face detection data is stored.
    data : Dict
        This dictionary holds the processed data. It contains two keys: 'images' and 'labels'.
    """

    def __init__(self, face_detection_cnn_config):
        """
        Initializes the PrepareData instance

        Parameters
        ----------
        data_paths : str
            The path to the base directory containing database
        """
        
        self.face_detection_cnn_config = face_detection_cnn_config

    def _get_image_names(self):
        images_path = []
        images_file = []
        
        data = {
            'image_names': [],
            'subjects': []
        }

        images_path = os.path.join(self.face_detection_cnn_config.data_dir, "face")

        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Directory {images_path} not found.")

        images_file = [f for f in os.listdir(images_path) if f.endswith('.bmp')]

        # tqdm is used to show the progress barr
        for image_file in tqdm(images_file, desc="Loading face image names", unit="file"):
            if image_file.endswith(".bmp"):
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

        train_set_image_names, test_set_image_names, train_set_subjects, _ = train_test_split(data['image_names'], data['subjects'], test_size=self.face_detection_cnn_config.data.test_size, random_state=42, shuffle=True, stratify=data['subjects'])
        train_set_image_names, val_set_image_names, _, _ = train_test_split(train_set_image_names, train_set_subjects, test_size=self.face_detection_cnn_config.data.val_size, random_state=42, shuffle=True, stratify=train_set_subjects)

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
        
    def _load_images_and_annotations(self, images_path, images_file, train_set_image_names, val_set_image_names):
        train_set = {
            'images': [],
            'bounding_boxes': []
        }

        val_set = {
            'images': [],
            'bounding_boxes': []
        }

        test_set = {
            'images': [],
            'bounding_boxes': []
        }

        # Percorsi di salvataggio
        for subset in ['train', 'val', 'test']:
            subset_path = os.path.join(self.face_detection_cnn_config.save_data_splitted_path, "splitted database", "face", subset)

            os.makedirs(subset_path, exist_ok=True)
    
            os.makedirs(os.path.join(subset_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(subset_path, "labels"), exist_ok=True)

        bounding_boxes_path = os.path.join(self.face_detection_cnn_config.bounding_boxes_dir, "face")

        if not os.path.exists(bounding_boxes_path):
            raise FileNotFoundError(f"Directory {bounding_boxes_path} not found.")

        for image_file in tqdm(images_file, desc=f"Loading {images_path}", unit="file"):
            if image_file.endswith(".bmp"):
                # Carica l'immagine
                image_path = os.path.join(images_path, image_file)

                image_name = os.path.basename(os.path.splitext(image_path)[0])

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                # image = Image.open(image_path).convert("RGB")
                    
                if image is None:
                    raise FileNotFoundError(f"Immagine non trovata: {image_path}")
                
                if image_name in train_set_image_names or image_name in val_set_image_names:
                    image = self._data_augmentatation(image.copy())

                # Converti da BGR (OpenCV) a RGB (Pillow)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = Image.fromarray(image).resize((self.face_detection_cnn_config.data.image_size, self.face_detection_cnn_config.data.image_size))

                bounding_box_path = os.path.join(bounding_boxes_path, image_name + '.txt')

                bounding_box_name = os.path.basename(os.path.splitext(bounding_box_path)[0])

                # Read the file and extract the annotations
                with open(bounding_box_path, 'r') as file:
                    line = file.readline()
                    bounding_box = list(map(float, line.strip().split()[-4:]))

                    if bounding_box_name in train_set_image_names:
                        train_set['images'].append(image)
                        train_set['bounding_boxes'].append(bounding_box)
                        subset = 'train'
                    elif bounding_box_name in val_set_image_names:
                        val_set['images'].append(image)
                        val_set['bounding_boxes'].append(bounding_box)
                        subset = 'val'
                    else:
                        test_set['images'].append(image)
                        test_set['bounding_boxes'].append(bounding_box)
                        subset = 'test'

                # Salva l'immagine nella directory corrispondente
                if self.face_detection_cnn_config.save_image.save_loaded_image:
                    # Salva l'immagine e la bounding box nella directory corrispondente
                    subset_image_path = os.path.join(self.face_detection_cnn_config.save_data_splitted_path, "splitted database", "face", subset, "images", f"{image_name}.bmp")
                    subset_bounding_box_path = os.path.join(self.face_detection_cnn_config.save_data_splitted_path, "splitted database", "face", subset, "labels", f"{image_name}.txt")

                    # Salva l'immagine nella directory corrispondente
                    image.save(subset_image_path)

                    # Salva la bounding box nel file corrispondente
                    shutil.copy2(bounding_box_path, subset_bounding_box_path)

        return train_set, val_set, test_set
    
    def _calculate_mean_and_std(self, images):
        """
        Calcola la media e la deviazione standard per ogni canale (R, G, B) di un dataset di immagini.

        Parameters
        ----------
        images : list of PIL.Image
            Lista di immagini PIL caricate.

        Returns
        -------
        mean : list
            Lista delle medie per ciascun canale [R_mean, G_mean, B_mean].
        std : list
            Lista delle deviazioni standard per ciascun canale [R_std, G_std, B_std].
        """
        num_images = len(images)
        if num_images == 0:
            raise ValueError("La lista delle immagini è vuota.")

        # Inizializza array accumulatore per somma dei pixel
        mean = np.zeros(3)  # Tre canali: R, G, B
        std = np.zeros(3)

        for image in images:
            # Converti immagine in array NumPy
            np_image = np.array(image) / 255.0  # Normalizza in [0, 1]
            # Calcola la somma dei pixel per canale
            mean += np_image.mean(axis=(0, 1))  # Media per canale
            std += np_image.std(axis=(0, 1))   # Std dev per canale

        mean /= num_images
        std /= num_images

        return mean.tolist(), std.tolist()
    
    def load_data(self):
        images_path, images_file, data = self._get_image_names()

        train_set_image_names, val_set_image_names, test_set_image_names = self._data_splitting(data)

        train_set, val_set, test_set = self._load_images_and_annotations(images_path, images_file, train_set_image_names, val_set_image_names)

        # Calcola media e deviazione standard sulle immagini del training set
        mean, std = self._calculate_mean_and_std(train_set['images'])
        print(f"Media (R, G, B): {mean}")
        print(f"Deviazione standard (R, G, B): {std} \n\n")

        return train_set, val_set, test_set, mean, std