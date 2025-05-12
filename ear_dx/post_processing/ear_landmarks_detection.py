# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import cv2
import torch
from torchvision import transforms
from PIL import Image

from ear_landmarks_detection.ear_landmarks_detection_utils import load_config, select_device, select_model


class EarLandmarksDetection:
    def __init__(self, ear_config):
        self._ear_config = ear_config

    def _load_model(self):
        model_path = os.path.join(f"{self._ear_landmarks_detection_config.training.checkpoints_dir}{self._ear_landmarks_detection_config.biometric_trait}_landmarks_detection", f"{self._ear_landmarks_detection_config.training.model_name}.pt")
        # Load model weights
        self.model.load_state_dict(torch.load(model_path))

    def _prepare_predict_process(self):
        # Load configuration
        self._ear_landmarks_detection_config = load_config('ear_landmarks_detection/config/ear_landmarks_detection_config.yaml')

        # Set device
        self.device = select_device(self._ear_landmarks_detection_config)

        # Select model
        self.model = select_model(self._ear_landmarks_detection_config)

        self.model.to(self.device)

        # Load model weights
        self._load_model()

        # convert PIL image to torch tensor (prima c'era cv2.INTER_AREA)
        self.transform = transforms.Compose([
            transforms.Resize((self._ear_landmarks_detection_config.data.image_size, self._ear_landmarks_detection_config.data.image_size), interpolation=Image.Resampling.LANCZOS),
            transforms.ToTensor()
        ])

    def _prepare_image(self, image, bounding_box):
        x_min, y_min, x_max, y_max = bounding_box
    
        # Ritaglia la bounding box dall'immagine originale
        image = image[y_min:y_max, x_min:x_max]
        
        image = cv2.resize(image, (self._ear_landmarks_detection_config.data.image_size, self._ear_landmarks_detection_config.data.image_size), interpolation=cv2.INTER_AREA)

        return image

    def predict_ear_landmarks(self, image, bounding_box):
        self._prepare_predict_process()

        orig_image = image.copy()

        orig_height, orig_width, _ = image.shape

        ear_image = self._prepare_image(image.copy(), bounding_box)

        height, width, _ = ear_image.shape

        predicted_image = ear_image.copy()

        ear_image = cv2.cvtColor(ear_image, cv2.COLOR_BGR2RGB)

        ear_image = Image.fromarray(ear_image)

        # Converti l'immagine in tensore e normalizza
        image_tensor = self.transform(ear_image).unsqueeze(0).to(self.device, dtype=torch.float32)

        self.model.eval()  # Imposta il modello in modalità di valutazione

        # Effettua la predizione
        with torch.no_grad():
            prediction_landmarks = self.model(image_tensor)

        # Rimuove la dimensione del batch e converte in numpy
        prediction_landmarks = prediction_landmarks.squeeze(0).cpu().numpy()

        x_top_pred, y_top_pred, x_bottom_pred, y_bottom_pred, x_outer_pred, y_outer_pred, x_inner_pred, y_inner_pred = map(float, prediction_landmarks)

        norm_predicted_landmarks = [x_top_pred, y_top_pred, x_bottom_pred, y_bottom_pred, x_outer_pred, y_outer_pred, x_inner_pred, y_inner_pred]

        # Calcola le coordinate dei reference landmarks in pixel rispetto a orig_image
        x_top_pred *= orig_width
        y_top_pred *= orig_height
        x_bottom_pred *= orig_width
        y_bottom_pred *= orig_height
        x_outer_pred *= orig_width
        y_outer_pred *= orig_height
        x_inner_pred *= orig_width
        y_inner_pred *= orig_height

        orig_predicted_landmarks = [
            int(x_top_pred), int(y_top_pred), 
            int(x_bottom_pred), int(y_bottom_pred), 
            int(x_outer_pred), int(y_outer_pred), 
            int(x_inner_pred), int(y_inner_pred)
        ]
        
        # Disegna i landmarks predetti sulla orig_image
        cv2.circle(orig_image, (int(x_top_pred), int(y_top_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(orig_image, (int(x_bottom_pred), int(y_bottom_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(orig_image, (int(x_outer_pred), int(y_outer_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(orig_image, (int(x_inner_pred), int(y_inner_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        
        # Calcola le coordinate dei reference landmarks in pixel rispetto a ear_image
        x_top_pred *= width
        y_top_pred *= height
        x_bottom_pred *= width
        y_bottom_pred *= height
        x_outer_pred *= width
        y_outer_pred *= height
        x_inner_pred *= width
        y_inner_pred *= height

        predicted_landmarks = [
            int(x_top_pred), int(y_top_pred), 
            int(x_bottom_pred), int(y_bottom_pred), 
            int(x_outer_pred), int(y_outer_pred), 
            int(x_inner_pred), int(y_inner_pred)
        ]

        # Disegna i landmarks predetti sulla ear_image
        cv2.circle(predicted_image, (int(x_top_pred), int(y_top_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(predicted_image, (int(x_bottom_pred), int(y_bottom_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(predicted_image, (int(x_outer_pred), int(y_outer_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(predicted_image, (int(x_inner_pred), int(y_inner_pred)), radius=3, color=(0, 0, 255), thickness=-1)

        if self._ear_config.show_images.detected_ear_landmarks:
            cv2.imshow("Predicted ear landmarks orig_image", orig_image)
            cv2.moveWindow("Predicted ear landmarks orig_image", 0, 0)
            
            cv2.imshow("Predicted ear landmarks ear_image", predicted_image)
            cv2.moveWindow("Predicted ear landmarks ear_image", 400, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return norm_predicted_landmarks, orig_image, orig_predicted_landmarks, predicted_image, predicted_landmarks