# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

try:
    from face.face_detection_yolo.utils import load_config, select_device
except:
    from ear_detection_yolo.utils import load_config, select_device


class Yolo:
    def __init__(self, face_config):
        self.face_config = face_config
        self._prepare_predict_process()

    def _load_model(self, face_detection_yolo_config):
        # Load model weights
        self.model = YOLO(f"{face_detection_yolo_config.training.checkpoints_dir}/{face_detection_yolo_config.training.model_name}.pt")

    def _prepare_predict_process(self):
        # Load configuration
        face_detection_yolo_config = load_config('face/face_detection_yolo/config/face_detection_yolo_config.yaml')

        # Set device
        self.device = select_device(face_detection_yolo_config)

        # Load model
        self._load_model(face_detection_yolo_config)

    def predict_face_bounding_box(self, image_path):
        # Effettua la predizione
        prediction = self.model.predict(
            source=image_path,
            conf=0.25,
            iou=0.6,
            imgsz=self.face_config.face_detection.image_size,
            device=self.device,
            retina_masks=False,
            # Visualization params:
            show=False,
            save=False,
            save_crop=False,    # True sul main (forse)
            show_labels=True,   # False sul main       
            show_conf=True,     # False sul main
            show_boxes=True,    # False sul main
            line_width=2
        )
        
        # Ottieni l'immagine annotata (bounding box, etichette, ecc.)
        predicted_image = prediction[0].plot()

        # print("PRED BOXES:", prediction[0].boxes, "\n\n")
        # print("PRED BOXES ORIGIN SHAPE:", prediction[0].boxes.orig_shape, "\n\n")
        # print("PRED BOXES CONF:", prediction[0].boxes.conf, "\n\n")
        # print("PRED BOXES CLS:", prediction[0].boxes.cls, "\n\n")
        # print("PRED BOXES XYXY:", prediction[0].boxes.xyxy, "\n\n")
        # print("PRED BOXES XYXYN:", prediction[0].boxes.xyxyn, "\n\n")
        # print("PRED BOXES XYWH:", prediction[0].boxes.xywh, "\n\n")
        # print("PRED BOXES XYWHN:", prediction[0].boxes.xywhn, "\n\n")

        if self.face_config.show_images.detected_face_bounding_box:
            cv2.imshow("Predicted face bounding box image", predicted_image)
            cv2.moveWindow("Predicted face bounding box image", 0, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Trova l'indice del valore massimo di confidenza
        max_conf_index = torch.argmax(prediction[0].boxes.conf)

        # Estrai la bounding box corrispondente
        best_xyxyn = prediction[0].boxes.xyxyn[max_conf_index]

        print("Confidenza più alta:", prediction[0].boxes.conf[max_conf_index].item())
        print("Bounding box corrispondente:", best_xyxyn)

        return predicted_image, best_xyxyn.tolist()