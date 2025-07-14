# Standard library imports
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import torch
from ultralytics import YOLO

try:
    from yolo_detection.yolo_utils import load_config, select_device
except ModuleNotFoundError:
    # Fallback to relative import
    sys.path.append('/Users/giovanni/Desktop/Tesi di Laurea/GFE-BioSystem')
    from yolo_detection.yolo_utils import load_config, select_device


class Yolo:
    def __init__(self, config, biometric_trait):
        self._config = config
        self.biometric_trait = biometric_trait
        self._prepare_predict_process()

    def _load_model(self, yolo_detection_config):
        # Load model weights
        self.model = YOLO(f"{yolo_detection_config.training.checkpoints_dir}/{yolo_detection_config.training.model_name}.pt")

    def _prepare_predict_process(self):
        # Load configuration
        yolo_detection_config = load_config(f'yolo_detection/config/{self.biometric_trait}_yolo_detection_config.yaml')

        # Set device
        self.device = select_device(yolo_detection_config)

        # Load model
        self._load_model(yolo_detection_config)

    def predict_bounding_box(self, image_path):
        # Effettua la predizione
        prediction = self.model.predict(
            source=image_path,
            conf=0.25,
            iou=0.6,
            imgsz=self._config.detection.image_size,
            device=self.device,
            retina_masks=False,
            # Visualization params:
            show=False,
            save=False,
            save_crop=False,    # True sul main (forse)
            show_labels=True,   # False sul main       
            show_conf=True,     # False sul main
            show_boxes=True,    # False sul main
            line_width=2,
            verbose=False
        )
        
        # Ottieni l'immagine annotata (bounding box, etichette, ecc.)
        predicted_image = prediction[0].plot()

        if self._config.show_images.detected_bounding_box:
            cv2.imshow(f"Predicted {self.biometric_trait} bounding box image", predicted_image)
            cv2.moveWindow(f"Predicted {self.biometric_trait} bounding box image", 0, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Trova l'indice del valore massimo di confidenza
        max_conf_index = torch.argmax(prediction[0].boxes.conf)

        # Estrai la bounding box corrispondente
        best_xyxyn = prediction[0].boxes.xyxyn[max_conf_index]

        # print("Confidenza più alta:", prediction[0].boxes.conf[max_conf_index].item())
        # print("Bounding box corrispondente:", best_xyxyn)

        return predicted_image, best_xyxyn.tolist()