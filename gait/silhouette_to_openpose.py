# -*- coding: utf-8 -*-
"""
Script per estrarre keypoints da immagini di silhouette di andatura (dataset OU-ISIR MVLP-Pose)
utilizzando il modello OpenPose via OpenCV DNN e salvando in formato JSON compatibile OpenPose.

Prerequisiti:
1. Scaricare i file del modello OpenPose COCO (pose_deploy_linevec.prototxt e pose_iter_440000.caffemodel) da:
   https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models/pose/coco
2. Organizzare la struttura cartelle:
   - models/
     - pose_deploy_linevec.prototxt
     - pose_iter_440000.caffemodel
   - input_images/   (immagini .png della silhouette)
   - output/         (verranno salvati i JSON e le immagini con keypoints visivi)

Uso:
> python extract_gait_keypoints_openpose.py
"""
import os
import sys
import glob
import json
import cv2
import numpy as np

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_config, browse_path, path_extractor, save_image, load_checkpoint, save_checkpoint


# -----------------------
# FUNZIONI DI PREPROCESSING
# -----------------------
def preprocess_silhouette(img):
    # Partiamo da BGR o grayscale
    if len(img.shape) < 3 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Rimuoviamo artefatti con blur + threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    # Manteniamo silhouette su sfondo neutro (127)
    bg = np.full_like(img, 127)
    fg = cv2.bitwise_and(img, img, mask=mask)
    return cv2.bitwise_or(fg, bg)


if __name__ == '__main__':
    gait_config = load_config('config/gait_config.yaml')

    if gait_config.browse_path:
        gait_config.data_dir = browse_path('Select the database folder')
        gait_config.save_path = browse_path('Select the folder where images and plots will be saved')

    # -----------------------
    # CONFIGURAZIONE MODELLI
    # -----------------------
    protoFile  = os.path.join(gait_config.models_dir, "pose_deploy_linevec.prototxt")
    weightsFile = os.path.join(gait_config.models_dir, "pose_iter_440000.caffemodel")

    # Numero di punti chiave nel modello COCO (18)
    nPoints = 18
    # Soglia minima di confidenza per considerare un punto valido
    threshold = 0.1

    # Carica rete
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # try:
    #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    #     print("[INFO] Utilizzo GPU CUDA per DNN")
    # except Exception:
    #     print("[INFO] CUDA non disponibile, si usa CPU")

    # Cartelle input/output
    os.makedirs(gait_config.save_dir, exist_ok=True)

    # Parametri di inferenza
    inWidth  = 656   # più alto per silhouette pulite
    inHeight = 368
    scale    = 1.0/255
    mean     = (0, 0, 0)
    swapRB   = False
    crop     = False

    # -----------------------
    # ELABORAZIONE IMMAGINI
    # -----------------------
    for img_path in sorted(glob.glob(os.path.join(gait_config.data_dir, '*.png'))):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARN] Impossibile caricare {img_path}")
            continue

        # Preprocess silhouette
        proc_img = preprocess_silhouette(frame)

        # Blob e inferenza
        blob = cv2.dnn.blobFromImage(proc_img, scale, (inWidth, inHeight), mean, swapRB, crop)
        net.setInput(blob)
        output = net.forward()

        H, W = output.shape[2], output.shape[3]

        # Estrazione keypoints
        keypoints = []
        for i in range(nPoints):
            probMap = output[0, i, :, :]
            _, prob, _, point = cv2.minMaxLoc(probMap)
            x = (frame.shape[1] * point[0]) / W
            y = (frame.shape[0] * point[1]) / H
            if prob > threshold:
                keypoints.append((int(x), int(y), float(prob)))
            else:
                keypoints.append((0, 0, 0.0))

        # Costruzione JSON stile OpenPose
        pose2d = []
        for x, y, c in keypoints:
            pose2d.extend([x, y, c])
        json_data = {
            "version": 1.2,
            "people": [{
                "pose_keypoints_2d": pose2d,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }]
        }

        # Salvataggio
        base = os.path.basename(img_path).rsplit('.',1)[0]
        json_path = os.path.join(gait_config.save_dir, f"{base}_keypoints.json")
        vis_path  = os.path.join(gait_config.save_dir, f"{base}_vis.png")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Visualizzazione keypoints
        vis = frame.copy()
        for idx in range(nPoints):
            x, y, c = keypoints[idx]
            if c > 0:
                cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
        cv2.imwrite(vis_path, vis)

        print(f"[INFO] Processata {base}: -> {json_path}, {vis_path}")
