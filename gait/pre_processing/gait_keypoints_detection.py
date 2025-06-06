# Standard library imports
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import os
from mmengine.config import Config
from mmpose.apis import init_model, inference_topdown

try:
    from gait_keypoints_detection.gait_keypoints_detection_utils import load_config, select_device
except ModuleNotFoundError:
    # Fallback to relative import
    sys.path.append('/Users/giovanni/Desktop/Tesi di Laurea/GFE-BioSystem')
    from gait_keypoints_detection.gait_keypoints_detection_utils import load_config, select_device


class GaitKeypointsDetection:
    def __init__(self, gait_config):
        self._gait_config = gait_config
        self._prepare_predict_process()

    def _load_model(self, gait_keypoints_detection_config):
        train_config_file = '/Users/giovanni/Desktop/Tesi di Laurea/GFE-BioSystem/gait_keypoints_detection/config/train_gait_keypoints_detection_model.py'

        checkpoint_path = os.path.join(gait_keypoints_detection_config.training.checkpoints_dir, 'epoch_100.pth')

        # Load model weights
        self.model = init_model(
            train_config_file, checkpoint_path, device=self.device,  
            cfg_options=dict(device=self.device)  # se serve iniettare via cfg‐options
        )

    def _prepare_predict_process(self):
        # Load configuration
        gait_keypoints_detection_config = load_config('gait_keypoints_detection/config/gait_keypoints_detection_config.yaml')

        # Set device
        self.device = select_device(gait_keypoints_detection_config)

        self._load_model(gait_keypoints_detection_config)

    # def draw_keypoints(self, frame, keypoints, keypoint_scores, radius=3, color=(0, 255, 0), thickness=-1):
    #     """
    #     Disegna ogni keypoint su img (x,y,v).
    #     """
    #     for (x, y), v in zip(keypoints, keypoint_scores):
    #         if v > 0:
    #             cv2.circle(frame, (int(x), int(y)), radius, color, thickness)
    #     return frame

    def _draw_keypoints(self, frame, keypoints):
        """
        Visualizza i keypoints OpenPose sul frame.
        
        Parameters
        ----------
        keypoints_data : dict
            Dizionario contenente i keypoints in formato COCO
        frame : numpy.ndarray
            Immagine del frame su cui visualizzare i keypoints
        """

        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # Definisci colori diversi per diverse parti del corpo
        colors = {
            'head': (0, 255, 255),    # Giallo
            'torso': (0, 0, 255),     # Rosso
            'right_arm': (255, 0, 0),  # Blu
            'left_arm': (0, 255, 0),   # Verde
            'right_leg': (255, 0, 255), # Magenta
            'left_leg': (255, 255, 0)   # Ciano
        }

        custom_connections = [
            [0, 1, 'head'], [0, 2, 'head'], [1, 2, 'head'], [1, 3, 'head'], [2, 4, 'head'], [3, 5, 'head'], [4, 6, 'head'],
            [5, 6, 'torso'], [5, 11, 'torso'], [6, 12, 'torso'], [11, 12, 'torso'],
            [5, 7, 'left_arm'], [7, 9, 'left_arm'],
            [6, 8, 'right_arm'], [8, 10, 'right_arm'],
            [11, 13, 'left_leg'], [13, 15, 'left_leg'],
            [12, 14, 'right_leg'], [14, 16, 'right_leg']
        ]

        # Disegna i keypoints
        num_keypoints = len(keypoints) // 3
        for i in range(num_keypoints):
            x = int(keypoints[3 * i])
            y = int(keypoints[3 * i + 1])
            v = keypoints[3 * i + 2]  # visibilità
            
            if v > 0:  # Se il keypoint è visibile
                # Disegna un cerchio sul keypoint
                cv2.circle(frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
                
                # Se sono disponibili i nomi, li disegna accanto al punto
                if i < len(keypoint_names):
                    cv2.putText(frame, keypoint_names[i], (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Disegna le connessioni personalizzate
        for conn in custom_connections:
            idx1, idx2, part = conn
            
            # Verifica che entrambi i keypoints siano validi
            if (idx1 * 3 + 2 < len(keypoints) and idx2 * 3 + 2 < len(keypoints) and
                keypoints[idx1 * 3 + 2] > 0 and keypoints[idx2 * 3 + 2] > 0):
                
                pt1 = (int(keypoints[idx1 * 3]), int(keypoints[idx1 * 3 + 1]))
                pt2 = (int(keypoints[idx2 * 3]), int(keypoints[idx2 * 3 + 1]))
                
                # Disegna una linea tra i due keypoints con il colore della parte del corpo
                cv2.line(frame, pt1, pt2, colors[part], 2)

        return frame

    def detect_keypoints(self, frame):
        h, w, _ = frame.shape

        # Top‐down: usa il bbox che copre tutta l’immagine letterbox (256×192)
        # Se le tue immagini sono già letterbox 256×192, puoi fare:
        bbox = np.array([[0, 0, w, h]], dtype=np.float32)

        # --- 4) Inferenza ---
        pose_results = inference_topdown(
            self.model,
            frame,
            bboxes=bbox,
            bbox_format='xyxy'
        )
        # pose_results è una lista di dict, per noi [0]['keypoints'] è Nx3

        # Estrai il primo (e unico) set di keypoints
        keypoints_coords = pose_results[0].pred_instances.keypoints[0]  # numpy array (17, 2)
        keypoint_scores = pose_results[0].pred_instances.keypoint_scores[0]  # numpy array (17)

        keypoints_xyv = []
        # print("KEYPOINTS: ", kps.tolist())

        for keypoint_coords, keypoint_score in zip(keypoints_coords.tolist(), keypoint_scores.tolist()):
            # print(f"Keypoint: {kp}, Score: {kp_score}")
            keypoints_xyv.append(keypoint_coords[0])
            keypoints_xyv.append(keypoint_coords[1])
            keypoints_xyv.append(keypoint_score)

        # print(f"Keypoints detection results: {keypoints}")
        # print(f"Keypoints score detection results: {keypoint_scores}")

        # --- 5) Visualizzazione e salvataggio ---
        # frame_with_detected_keypoints = self.draw_keypoints(frame.copy(), keypoints_xyv, radius=4, color=(0, 255, 0), thickness=-1)
        frame_with_detected_keypoints = self._draw_keypoints(frame.copy(), keypoints_xyv)

        if self._gait_config.show_images.detected_keypoints:
            cv2.imshow(f"Detected gait keypoints frame", frame_with_detected_keypoints)
            cv2.moveWindow(f"Detected gait keypoints frame", 0, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"Detected gait keypoints: {keypoints_xyv}")
        
        return keypoints_xyv, frame_with_detected_keypoints