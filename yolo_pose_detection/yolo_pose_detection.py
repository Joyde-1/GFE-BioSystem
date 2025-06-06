# Standard library imports
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

try:
    from yolo_pose_detection.yolo_pose_utils import load_config, select_device
except ModuleNotFoundError:
    # Fallback to relative import
    sys.path.append('/Users/giovanni/Desktop/Tesi di Laurea/GFE-BioSystem')
    from yolo_pose_detection.yolo_pose_utils import load_config, select_device


class YoloPose:
    def __init__(self, config, biometric_trait):
        self._config = config
        self.biometric_trait = biometric_trait
        self._prepare_predict_process()

    def _load_model(self, yolo_pose_detection_config):
        # Load model weights
        self.model = YOLO(f"{yolo_pose_detection_config.training.checkpoints_dir}/{yolo_pose_detection_config.training.model_name}.pt")

    def _prepare_predict_process(self):
        # Load configuration
        yolo_pose_detection_config = load_config(f'yolo_pose_detection/config/{self.biometric_trait}_yolo_pose_detection_config.yaml')

        # Set device
        self.device = select_device(yolo_pose_detection_config)

        # Load model
        self._load_model(yolo_pose_detection_config)

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
            # kpt_radius=3,
            # kpt_line=2
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

        if self._config.show_images.detected_keypoints:
            cv2.imshow(f"Predicted {self.biometric_trait} keypoints image", predicted_image)
            cv2.moveWindow(f"Predicted {self.biometric_trait} keypoints image", 0, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        keypoints = []

        keypoints_coords = prediction[0].keypoints.xy[0].cpu().numpy().tolist()  # Convert keypoints to numpy array
        keypoints_scores = prediction[0].keypoints.conf[0].cpu().numpy().tolist()  # Get keypoints confidence scores
        print("Keypoints:", keypoints_coords)
        print("Keypoints confidence scores:", keypoints_scores)

        for keypoint_coords, keypoint_score in zip(keypoints_coords, keypoints_scores):
            # print(f"Keypoint: {kp}, Score: {kp_score}")
            keypoints.append(keypoint_coords[0])
            keypoints.append(keypoint_coords[1])
            keypoints.append(keypoint_score)
        
        # # Trova l'indice del valore massimo di confidenza
        # max_conf_index = torch.argmax(prediction[0].boxes.conf)

        # # Estrai la bounding box corrispondente
        # best_xyxyn = prediction[0].boxes.xyxyn[max_conf_index]

        # print("Confidenza più alta:", prediction[0].boxes.conf[max_conf_index].item())
        # print("Bounding box corrispondente:", best_xyxyn)

        return keypoints, predicted_image