# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import torch
from torchvision import transforms
from PIL import Image

try:
    from face.face_detection_cnn.utils import load_config, select_device, select_model
except ModuleNotFoundError:
    from face_detection_cnn.utils import load_config, select_device, select_model


class CNN:
    def __init__(self, face_config):
        self.face_config = face_config

    def _load_model(self, face_detection_cnn_config):
        # Load model weights
        self.model.load_state_dict(torch.load(f"{face_detection_cnn_config.training.checkpoints_dir}/{face_detection_cnn_config.training.model_name}.pt"))
    
    def _prepare_predict_process(self):
        # Load configuration
        face_detection_cnn_config = load_config('face/face_detection_cnn/config/face_detection_cnn_config.yaml')

        # Set device
        self.device = select_device(face_detection_cnn_config)

        # Select model
        self.model = select_model(face_detection_cnn_config)

        self.model.to(self.device)

        # Load model weights
        self._load_model(face_detection_cnn_config)

        # convert PIL image to torch tensor (prima c'era cv2.INTER_AREA)
        self.transform = transforms.Compose([
            transforms.Resize((face_detection_cnn_config.data.image_size, face_detection_cnn_config.data.image_size), interpolation=Image.Resampling.LANCZOS),
            transforms.ToTensor()
        ])

    def predict_face_bounding_box(self, image):
        self._prepare_predict_process()

        height, width, _ = image.shape

        predicted_image = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        # Converti l'immagine in tensore e normalizza
        image_tensor = self.transform(image).unsqueeze(0).to(self.device, dtype=torch.float32)

        self.model.eval()  # Imposta il modello in modalità di valutazione

        # Effettua la predizione
        with torch.no_grad():
            predicted_bounding_box = self.model(image_tensor)

        # Rimuove la dimensione del batch e converte in numpy
        predicted_bounding_box = predicted_bounding_box.squeeze(0).cpu().numpy()

        x_center, y_center, box_width, box_height = map(float, predicted_bounding_box)

        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2
        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2

        predicted_bounding_box = [x_min, y_min, x_max, y_max]

        # Calcola le coordinate delle predicted bounding box in pixel
        x_min = int(x_min * width)
        y_min = int(y_min * height)
        x_max = int(x_max * width)
        y_max = int(y_max * height)

        # Disegna la buonding box predetta sull'immagine
        cv2.rectangle(predicted_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

        if self.face_config.show_images.detected_face_bounding_box:
            cv2.imshow("Predicted face bounding box image", predicted_image)
            cv2.moveWindow("Predicted face bounding box image", 0, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return predicted_image, predicted_bounding_box