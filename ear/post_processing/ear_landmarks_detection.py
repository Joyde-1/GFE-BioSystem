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
        model_path = os.path.join(f"{self._ear_landmarks_detection_config.training.checkpoints_dir}{self._ear_config.biometric_trait}_landmarks_detection", f"{self._ear_landmarks_detection_config.training.model_name}.pt")
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

        pred_height, pred_width, _ = image.shape
        
        image = cv2.resize(image, (self._ear_landmarks_detection_config.data.image_size, self._ear_landmarks_detection_config.data.image_size), interpolation=cv2.INTER_AREA)

        return image, pred_height, pred_width

    def predict_ear_landmarks(self, image, bounding_box):
        self._prepare_predict_process()

        orig_image = image.copy()
        orig_height, orig_width, _ = image.shape

        # Prepara l'immagine ritagliata e ridimensionata per la predizione
        ear_image, bbox_height, bbox_width = self._prepare_image(image.copy(), bounding_box)
        model_height, model_width, _ = ear_image.shape

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

        # Estrai le coordinate dei landmarks predetti (normalizzate tra 0 e 1)
        x_top_pred, y_top_pred, x_bottom_pred, y_bottom_pred, x_outer_pred, y_outer_pred, x_inner_pred, y_inner_pred = map(float, prediction_landmarks)

        # Salva i landmarks normalizzati (output del modello)
        norm_predicted_landmarks = [x_top_pred, y_top_pred, x_bottom_pred, y_bottom_pred, x_outer_pred, y_outer_pred, x_inner_pred, y_inner_pred]

        # Estrai le coordinate della bounding box
        x_min, y_min, x_max, y_max = bounding_box
        
        # REMAPPING DEI LANDMARKS NELL'IMMAGINE ORIGINALE
        
        # 1. Converti i landmarks normalizzati in coordinate pixel nell'immagine del modello (model_width x model_height)
        x_top_model = x_top_pred * model_width
        y_top_model = y_top_pred * model_height
        x_bottom_model = x_bottom_pred * model_width
        y_bottom_model = y_bottom_pred * model_height
        x_outer_model = x_outer_pred * model_width
        y_outer_model = y_outer_pred * model_height
        x_inner_model = x_inner_pred * model_width
        y_inner_model = y_inner_pred * model_height
        
        # 2. Scala le coordinate dall'immagine del modello alla bounding box originale
        scale_x = bbox_width / model_width
        scale_y = bbox_height / model_height
        
        x_top_bbox = x_top_model * scale_x
        y_top_bbox = y_top_model * scale_y
        x_bottom_bbox = x_bottom_model * scale_x
        y_bottom_bbox = y_bottom_model * scale_y
        x_outer_bbox = x_outer_model * scale_x
        y_outer_bbox = y_outer_model * scale_y
        x_inner_bbox = x_inner_model * scale_x
        y_inner_bbox = y_inner_model * scale_y
        
        # 3. Aggiungi l'offset della bounding box per ottenere le coordinate nell'immagine originale
        x_top_orig = x_min + x_top_bbox
        y_top_orig = y_min + y_top_bbox
        x_bottom_orig = x_min + x_bottom_bbox
        y_bottom_orig = y_min + y_bottom_bbox
        x_outer_orig = x_min + x_outer_bbox
        y_outer_orig = y_min + y_outer_bbox
        x_inner_orig = x_min + x_inner_bbox
        y_inner_orig = y_min + y_inner_bbox
        
        # Crea la lista dei landmarks predetti nell'immagine originale
        orig_predicted_landmarks = [
            int(x_top_orig), int(y_top_orig), 
            int(x_bottom_orig), int(y_bottom_orig), 
            int(x_outer_orig), int(y_outer_orig), 
            int(x_inner_orig), int(y_inner_orig)
        ]
        
        # Disegna i landmarks predetti sull'immagine originale
        cv2.circle(orig_image, (int(x_top_orig), int(y_top_orig)), radius=6, color=(0, 255, 0), thickness=-1)
        cv2.circle(orig_image, (int(x_bottom_orig), int(y_bottom_orig)), radius=6, color=(0, 255, 0), thickness=-1)
        cv2.circle(orig_image, (int(x_outer_orig), int(y_outer_orig)), radius=6, color=(0, 255, 0), thickness=-1)
        cv2.circle(orig_image, (int(x_inner_orig), int(y_inner_orig)), radius=6, color=(0, 255, 0), thickness=-1)
        
        # Disegna i landmarks predetti sull'immagine ritagliata
        cv2.circle(predicted_image, (int(x_top_model), int(y_top_model)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(predicted_image, (int(x_bottom_model), int(y_bottom_model)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(predicted_image, (int(x_outer_model), int(y_outer_model)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(predicted_image, (int(x_inner_model), int(y_inner_model)), radius=3, color=(0, 0, 255), thickness=-1)
        
        # Crea la lista dei landmarks predetti nell'immagine ritagliata
        predicted_landmarks = [
            int(x_top_model), int(y_top_model), 
            int(x_bottom_model), int(y_bottom_model), 
            int(x_outer_model), int(y_outer_model), 
            int(x_inner_model), int(y_inner_model)
        ]

        if self._ear_config.show_images.detected_ear_landmarks:
            cv2.imshow("Predicted ear landmarks orig_image", orig_image)
            cv2.moveWindow("Predicted ear landmarks orig_image", 0, 0)
            
            cv2.imshow("Predicted ear landmarks ear_image", predicted_image)
            cv2.moveWindow("Predicted ear landmarks ear_image", 400, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return norm_predicted_landmarks, orig_image, orig_predicted_landmarks, predicted_image, predicted_landmarks