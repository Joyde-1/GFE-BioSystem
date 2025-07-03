import cv2
from tqdm import tqdm
import os
import json
import shutil
import random
import numpy as np
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

    def __init__(self, ear_landmarks_detection_config):
        """
        Initializes the PrepareData instance

        Parameters
        ----------
        data_paths : str
            The path to the base directory containing database
        """
        
        self._ear_landmarks_detection_config = ear_landmarks_detection_config

    def _get_image_names(self):
        images_path = []
        images_file = []

        data = {
            'image_names': [],
            'subjects': []
        }

        images_path = os.path.join(self._ear_landmarks_detection_config.data_dir, self._ear_landmarks_detection_config.biometric_trait)

        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Directory {images_path} not found.")

        images_file = [f for f in os.listdir(images_path) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.png')]

        # tqdm is used to show the progress barr
        for image_file in tqdm(images_file, desc=f"Loading {self._ear_landmarks_detection_config.biometric_trait} image names", unit="file"):
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
            Dictionary containing 'images' and 'landmarks' as keys.
        config : dict
            Configuration object with parameters for splitting, including 'val_size' which defines the proportion of the validation set.

        Returns
        -------
        tuple of dict
            Returns two dictionaries, the first containing the training data and landmarks, and the second containing the validation data and landmarks.
        """

        train_set_image_names, test_set_image_names, train_set_subjects, _ = train_test_split(data['image_names'], data['subjects'], test_size=self._ear_landmarks_detection_config.data.test_size, random_state=42, shuffle=True, stratify=data['subjects'])
        train_set_image_names, val_set_image_names, _, _ = train_test_split(train_set_image_names, train_set_subjects, test_size=self._ear_landmarks_detection_config.data.val_size, random_state=42, shuffle=True, stratify=train_set_subjects)

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
    
    def _read_landmarks_json(self, landmarks_path, image_name):
        # Carica il file LabelMe
        with open(landmarks_path, 'r') as f:
            data = json.load(f)
        
        # Estrai le informazioni dell'immagine
        width = data['imageWidth']
        height = data['imageHeight']
        
        # Definisci l'ordine dei keypoint che vuoi utilizzare
        # In questo esempio consideriamo quattro keypoint: "top", "bottom", "outer", "inner"
        keypoints_order = ['top', 'bottom', 'outer', 'inner']
        keypoints = []
        
        # Per ciascun keypoint, cerca la corrispondente annotazione in LabelMe
        for kp in keypoints_order:
            point = None
            for shape in data['shapes']:
                if shape['label'] == kp:
                    # Presupponiamo che ogni shape contenga un solo punto
                    point = shape['points'][0]
                    break
            if point is None:
                raise ValueError(f"Annotazione mancante per il keypoint: {kp} and {image_name}")
            x, y = point

            # Normalize x and y coords
            x_norm = x / width
            y_norm = y / height

            keypoints.append(x_norm)
            keypoints.append(y_norm)
        
        return keypoints
    
    def _remap_landmarks(self, original_landmarks, x_min, y_min, bounding_box_w, bounding_box_h, orig_image_w, orig_image_h):
        remapped_landmarks = []

        # Processa ogni coppia (x, y)
        for i in range(0, len(original_landmarks), 2):
            x = original_landmarks[i]
            y = original_landmarks[i + 1]

            x = (x - x_min) / bounding_box_w
            y = (y - y_min) / bounding_box_h

            remapped_landmarks.extend([x, y])

        return remapped_landmarks

    def _crop_image(self, image, x_min, y_min, x_max, y_max):
        # Ottieni le dimensioni dell'immagine
        image_h, image_w, _ = image.shape

        x_min = int(x_min * image_w)
        y_min = int(y_min * image_h)
        x_max = int(x_max * image_w)
        y_max = int(y_max * image_h)

        # Ritaglia la bounding box dall'immagine originale
        image = image[y_min:y_max, x_min:x_max]

        return image

    def _convert_image_and_landmarks(self, image, landmarks, bounding_box_path):
        # Read the file and extract the annotations
        with open(bounding_box_path, 'r') as file:
            line = file.readline()
            bounding_box = list(map(float, line.strip().split()[-4:]))

        x_center, y_center, bounding_box_w, bounding_box_h = bounding_box
        x_min = x_center - bounding_box_w / 2
        y_min = y_center - bounding_box_h / 2
        x_max = x_center + bounding_box_w / 2
        y_max = y_center + bounding_box_h / 2

        # Get original image dimensions
        orig_image_h, orig_image_w, _ = image.shape

        # Crop the image using normalized coordinates
        image = self._crop_image(image, x_min, y_min, x_max, y_max)

        # Remap landmarks using pixel coordinates (since landmarks from _read_landmarks_json are in pixel coordinates)
        landmarks = self._remap_landmarks(landmarks, x_min, y_min, bounding_box_w, bounding_box_h, orig_image_h, orig_image_w)

        image = cv2.resize(image, (self._ear_landmarks_detection_config.data.image_size, self._ear_landmarks_detection_config.data.image_size), interpolation=cv2.INTER_AREA)

        return image, landmarks
        
    def _load_images_and_landmarks(self, images_path, images_file, train_set_image_names, val_set_image_names):
        train_set = {
            'images': [],
            'landmarks_list': []
        }

        val_set = {
            'images': [],
            'landmarks_list': []
        }

        test_set = {
            'images': [],
            'landmarks_list': []
        }

        # Percorsi di salvataggio
        for subset in ['train', 'val', 'test']:
            subset_path = os.path.join(self._ear_landmarks_detection_config.save_data_splitted_path, "splitted_ear_landmarks_database", self._ear_landmarks_detection_config.biometric_trait, subset)

            os.makedirs(subset_path, exist_ok=True)
    
            os.makedirs(os.path.join(subset_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(subset_path, "labels"), exist_ok=True)

        landmarks_list_path = os.path.join(self._ear_landmarks_detection_config.landmarks_dir, self._ear_landmarks_detection_config.biometric_trait)
        bouding_boxes_path = os.path.join(self._ear_landmarks_detection_config.bounding_boxes_dir, self._ear_landmarks_detection_config.biometric_trait)

        if not os.path.exists(landmarks_list_path):
            raise FileNotFoundError(f"Directory {landmarks_list_path} not found.")
        
        if not os.path.exists(bouding_boxes_path):
            raise FileNotFoundError(f"Directory {bouding_boxes_path} not found.")

        for image_file in tqdm(images_file, desc=f"Loading {self._ear_landmarks_detection_config.biometric_trait} image files", unit="file"):
            if image_file.endswith(".bmp") or image_file.endswith('.jpg') or image_file.endswith('.png'):
                # Carica l'immagine
                image_path = os.path.join(images_path, image_file)

                image_name = os.path.basename(os.path.splitext(image_path)[0])

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                # image = Image.open(image_path).convert("RGB")
                    
                if image is None:
                    raise FileNotFoundError(f"Immagine non trovata: {image_path}")
                
                if image_name in train_set_image_names or image_name in val_set_image_names:
                    image = self._data_augmentatation(image.copy())

                landmarks_path = os.path.join(landmarks_list_path, image_name + '.json')

                # Read the file and extract the landmarks
                landmarks = self._read_landmarks_json(landmarks_path, image_name)

                bounding_box_path = os.path.join(bouding_boxes_path, image_name + '.txt')

                image, landmarks = self._convert_image_and_landmarks(image.copy(), landmarks, bounding_box_path)

                # Converti da BGR (OpenCV) a RGB (Pillow)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = Image.fromarray(image)

                if image_name in train_set_image_names:
                    train_set['images'].append(image)
                    train_set['landmarks_list'].append(landmarks)
                    subset = 'train'
                elif image_name in val_set_image_names:
                    val_set['images'].append(image)
                    val_set['landmarks_list'].append(landmarks)
                    subset = 'val'
                else:
                    test_set['images'].append(image)
                    test_set['landmarks_list'].append(landmarks)
                    subset = 'test'

                # Salva l'immagine e i landmarks nella directory corrispondente
                if self._ear_landmarks_detection_config.save_image.save_loaded_data:
                    subset_image_path = os.path.join(self._ear_landmarks_detection_config.save_data_splitted_path, "splitted_ear_landmarks_database", self._ear_landmarks_detection_config.biometric_trait, subset, "images", f"{image_name}.bmp")
                    subset_landmarks_path = os.path.join(self._ear_landmarks_detection_config.save_data_splitted_path, "splitted_ear_landmarks_database", self._ear_landmarks_detection_config.biometric_trait, subset, "labels", f"{image_name}.json")

                    # Salva l'immagine nella directory corrispondente
                    image.save(subset_image_path)

                    # Salva i landmarks nel file corrispondente
                    shutil.copy2(landmarks_path, subset_landmarks_path)

        return train_set, val_set, test_set
    
    def load_data(self):
        images_path, images_file, data = self._get_image_names()

        train_set_image_names, val_set_image_names, test_set_image_names = self._data_splitting(data)

        train_set, val_set, test_set = self._load_images_and_landmarks(images_path, images_file, train_set_image_names, val_set_image_names)

        return train_set, val_set, test_set
    
    def calculate_mean_and_std(self, images):
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