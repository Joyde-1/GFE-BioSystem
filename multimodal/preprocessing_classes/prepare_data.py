import cv2
from tqdm import tqdm
import os


class PrepareData:

    sides = ['dx', 'sx']

    def __init__(self, multimodal_config):
        """
        Initializes the PrepareData instance

        Parameters
        ----------
        multimodal_config :
            Contains the path to the base directory containing database
        """
        
        self.multimodal_config = multimodal_config
    
    def load_face_images(self):
        images = []

        image_names = []

        image_paths = []
            
        face_path = os.path.join(self.multimodal_config.data_dir, "face")

        if not os.path.exists(face_path):
            raise FileNotFoundError(f"Directory {face_path} not found.")
        
        image_files = [f for f in os.listdir(face_path) if f.endswith('.bmp')]

        # Ordinamento in base al primo e secondo numero nel nome:
        image_files = sorted(
            image_files,
            key=lambda x: (
                int(x.split('_')[0]),                          # first number
                int(x.split('_')[1].split('.')[0])               # second number (remove extension)
            )
        )

        for image_file in tqdm(image_files, desc=f"Loading face images"):
            image_path = os.path.join(face_path, image_file)

            # Load gray scale image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if image is None:
                raise FileNotFoundError(f"Image {image_path} not found.")

            image_name = os.path.basename(os.path.splitext(image_path)[0])

            images.append(image)

            image_names.append(image_name)

            image_paths.append(image_path)
            
        return images, image_names, image_paths
    
    def load_ears_images(self):
        images = {
            'dx': [],
            'sx': []
        }

        image_names = {
            'dx': [],
            'sx': []
        }

        image_paths = {
            'dx': [],
            'sx': []
        }
            
        for side in self.eye_sides:
            side_path = os.path.join(self.multimodal_config.data_dir, "ear", side)
            if os.path.exists(side_path):
                image_files = [f for f in os.listdir(side_path) if f.endswith('.bmp')]

                # Sort by first and second number in the name:
                image_files = sorted(
                    image_files,
                    key=lambda x: (
                        int(x.split('_')[0]),                          # first number
                        int(x.split('_')[1].split('.')[0])               # second number (remove extension)
                    )
                )

                print(f"\n\nFiles in {side} folder: {image_files}\n\n")

                for image_file in tqdm(image_files, desc=f"Loading images in {side}"):
                    print(f"\n\nLoading image: {image_file}\n\n")
                    image_path = os.path.join(side_path, image_file)

                    # Load gray scale image
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        raise FileNotFoundError(f"Image not found: {image_path}")

                    image_name = os.path.basename(os.path.splitext(image_path)[0])

                    images[side].append(image)
                    image_names[side].append(image_name)
                    image_paths[side].append(image_path)
            
        return images, image_names, image_paths