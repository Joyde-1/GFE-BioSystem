import cv2
from tqdm import tqdm
import os


class LoadData:

    def __init__(self, config, biometric_trait):
        """
        Initializes the PrepareData instance

        Parameters
        ----------
        data_paths : str
            The path to the base directory containing database
        """
        
        self.config = config
        self.biometric_trait = biometric_trait
    
    # Funzione per caricare le immagini
    def load_face_images(self):
        
        images = []

        image_names = []

        image_paths = []
            
        path = os.path.join(self.config.data_dir, self.biometric_trait)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory {path} not found.")
        
        image_files = [f for f in os.listdir(path) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.png')]

        # Ordinamento in base al primo e secondo numero nel nome:
        image_files = sorted(
            image_files,
            key=lambda x: (
                int(x.split('_')[0]),                          # primo numero
                int(x.split('_')[1].split('.')[0])               # secondo numero (rimuovo l'estensione)
            )
        )

        for image_file in tqdm(image_files, desc=f"Loading {self.biometric_trait} images"):
            image_path = os.path.join(path, image_file)

            # Carica l'immagine in scala di grigi
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if image is None:
                raise FileNotFoundError(f"Image {image_path} not found.")

            image_name = os.path.basename(os.path.splitext(image_path)[0])

            images.append(image)

            image_names.append(image_name)

            image_paths.append(image_path)
            
        return images, image_names, image_paths