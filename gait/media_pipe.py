# -*- coding: utf-8 -*-
"""
Script per estrarre keypoints da immagini di silhouette di andatura
utilizzando MediaPipe Pose e salvando in formato JSON compatibile OpenPose.

Funzionalità:
1. Ritaglio automatico della silhouette per migliorare la precisione
2. Estrazione dei keypoints con MediaPipe Pose
3. Confronto con ground truth (se disponibile)
4. Salvataggio in formato JSON compatibile con OpenPose
"""
import os
import sys
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from pathlib import Path

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_config, browse_path, path_extractor, save_image, load_checkpoint, save_checkpoint

# Inizializzazione di MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Mappatura da MediaPipe a OpenPose (COCO format)
# MediaPipe ha 33 punti, OpenPose COCO ha 18 punti
# Questa mappatura associa i punti MediaPipe ai punti OpenPose
MEDIAPIPE_TO_OPENPOSE = {
    # OpenPose: MediaPipe
    0: 0,    # Nose: nose
    1: 11,   # Neck: left shoulder
    2: 12,   # RShoulder: right shoulder
    3: 14,   # RElbow: right elbow
    4: 16,   # RWrist: right wrist
    5: 11,   # LShoulder: left shoulder
    6: 13,   # LElbow: left elbow
    7: 15,   # LWrist: left wrist
    8: 24,   # MidHip: pelvis
    9: 26,   # RHip: right hip
    10: 28,  # RKnee: right knee
    11: 30,  # RAnkle: right ankle
    12: 25,  # LHip: left hip
    13: 27,  # LKnee: left knee
    14: 29,  # LAnkle: left ankle
    15: 2,   # REye: right eye
    16: 5,   # LEye: left eye
    17: 9,   # REar: right ear
}

# Definizione delle connessioni tra i keypoints per la visualizzazione
# Basato sul modello COCO a 18 punti
POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
]

# Colori per la visualizzazione
COLORS = [
    (0, 100, 255), (0, 100, 255), (0, 255, 255), (0, 100, 255), (0, 255, 255), (0, 100, 255),
    (0, 255, 0), (255, 200, 100), (255, 0, 255), (0, 255, 0), (255, 200, 100), (255, 0, 255),
    (0, 0, 255), (255, 0, 0), (200, 200, 0), (255, 0, 0), (200, 200, 0)
]

# -----------------------
# FUNZIONI DI PREPROCESSING
# -----------------------
def preprocess_silhouette(img):
    """Preprocessa l'immagine della silhouette per migliorare il rilevamento"""
    # Partiamo da BGR o grayscale
    if len(img.shape) < 3 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Rimuoviamo artefatti con blur + threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    # Aumenta il contrasto della silhouette
    img_enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=10)
    
    # Manteniamo silhouette su sfondo neutro (127 invece di 0)
    # Questo aiuta il modello a rilevare meglio i contorni
    bg = np.full_like(img_enhanced, 127)
    fg = cv2.bitwise_and(img_enhanced, img_enhanced, mask=mask)
    result = cv2.bitwise_or(fg, bg, mask=cv2.bitwise_not(mask))
    
    # Crea una versione colorata della silhouette per migliorare il rilevamento
    colored_silhouette = np.zeros_like(img_enhanced)
    colored_silhouette[:,:,0] = mask  # Canale blu
    colored_silhouette[:,:,1] = mask  # Canale verde
    colored_silhouette[:,:,2] = mask  # Canale rosso
    
    return colored_silhouette, mask

def auto_crop_silhouette(img, mask, padding=20):
    """Ritaglia automaticamente la silhouette con un padding"""
    # Trova i contorni della silhouette
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img, (0, 0, img.shape[1], img.shape[0])  # Nessun contorno trovato
    
    # Prendi il contorno più grande (dovrebbe essere la silhouette)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Aggiungi padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2*padding)
    h = min(img.shape[0] - y, h + 2*padding)
    
    # Ritaglia l'immagine
    cropped = img[y:y+h, x:x+w]
    return cropped, (x, y, x+w, y+h)

def load_ground_truth(json_path):
    """Carica i keypoints di ground truth da un file JSON"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'people' in data and len(data['people']) > 0:
            keypoints = data['people'][0]['pose_keypoints_2d']
            # Converti la lista piatta in una lista di tuple (x, y, conf)
            return [(keypoints[i], keypoints[i+1], keypoints[i+2]) 
                    for i in range(0, len(keypoints), 3)]
        return None
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None

def draw_keypoints(img, keypoints, color=(0, 255, 0), radius=4):
    """Disegna i keypoints sull'immagine"""
    vis = img.copy()
    for x, y, conf in keypoints:
        if conf > 0:
            cv2.circle(vis, (int(x), int(y)), radius, color, -1)
    
    # Disegna le connessioni tra i keypoints
    for pair_idx, pair in enumerate(POSE_PAIRS):
        partA = pair[0]
        partB = pair[1]
        
        if partA < len(keypoints) and partB < len(keypoints) and keypoints[partA][2] > 0 and keypoints[partB][2] > 0:
            cv2.line(vis, 
                    (int(keypoints[partA][0]), int(keypoints[partA][1])),
                    (int(keypoints[partB][0]), int(keypoints[partB][1])),
                    COLORS[pair_idx], 2)
    
    return vis

def map_keypoints_to_original(keypoints, crop_box):
    """Mappa i keypoints dall'immagine ritagliata all'immagine originale"""
    x_min, y_min, _, _ = crop_box
    mapped = []
    
    for x, y, conf in keypoints:
        if conf > 0:
            mapped.append((x + x_min, y + y_min, conf))
        else:
            mapped.append((0, 0, 0))
    
    return mapped

def compare_keypoints(img, pred_keypoints, gt_keypoints):
    """Crea un'immagine di confronto tra keypoints predetti e ground truth"""
    vis = img.copy()
    
    # Disegna i keypoints predetti in verde
    for x, y, conf in pred_keypoints:
        if conf > 0:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
    
    # Disegna i keypoints ground truth in blu
    for x, y, conf in gt_keypoints:
        if conf > 0:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
    
    # Disegna le connessioni per entrambi i set
    for pair_idx, pair in enumerate(POSE_PAIRS):
        partA = pair[0]
        partB = pair[1]
        
        # Connessioni predette (verde)
        if partA < len(pred_keypoints) and partB < len(pred_keypoints) and pred_keypoints[partA][2] > 0 and pred_keypoints[partB][2] > 0:
            cv2.line(vis, 
                    (int(pred_keypoints[partA][0]), int(pred_keypoints[partA][1])),
                    (int(pred_keypoints[partB][0]), int(pred_keypoints[partB][1])),
                    (0, 255, 0), 2)
        
        # Connessioni ground truth (blu)
        if partA < len(gt_keypoints) and partB < len(gt_keypoints) and gt_keypoints[partA][2] > 0 and gt_keypoints[partB][2] > 0:
            cv2.line(vis, 
                    (int(gt_keypoints[partA][0]), int(gt_keypoints[partA][1])),
                    (int(gt_keypoints[partB][0]), int(gt_keypoints[partB][1])),
                    (0, 0, 255), 2)
    
    return vis

def mediapipe_to_openpose(mp_results, image_shape):
    """Converte i risultati di MediaPipe nel formato OpenPose"""
    h, w = image_shape[:2]
    openpose_keypoints = [(0, 0, 0)] * 18  # Inizializza con 18 keypoints a zero
    
    if not mp_results.pose_landmarks:
        return openpose_keypoints
    
    # Estrai i landmark di MediaPipe
    landmarks = mp_results.pose_landmarks.landmark
    
    # Converti i landmark di MediaPipe nel formato OpenPose
    for op_idx, mp_idx in MEDIAPIPE_TO_OPENPOSE.items():
        if mp_idx < len(landmarks):
            landmark = landmarks[mp_idx]
            # Converti le coordinate normalizzate in pixel
            x = landmark.x * w
            y = landmark.y * h
            # Usa la visibilità come confidenza
            conf = landmark.visibility
            
            # Filtra i punti con bassa confidenza
            if conf > 0.3:
                openpose_keypoints[op_idx] = (x, y, conf)
    
    return openpose_keypoints

if __name__ == '__main__':
    gait_config = load_config('config/gait_config.yaml')

    if gait_config.browse_path:
        gait_config.data_dir = browse_path('Select the database folder')
        gait_config.save_path = browse_path('Select the folder where images and plots will be saved')

    # Cartelle input/output
    os.makedirs(gait_config.save_dir, exist_ok=True)
    
    # Cartella per i ground truth (se disponibili)
    gt_dir = os.path.join(gait_config.data_dir, 'ground_truth')
    has_ground_truth = os.path.exists(gt_dir)

    # -----------------------
    # ELABORAZIONE IMMAGINI
    # -----------------------
    # Inizializza MediaPipe Pose con parametri ottimizzati per silhouette
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,  # Usa il modello più complesso per maggiore precisione
        enable_segmentation=True,  # Abilita la segmentazione per migliorare il rilevamento
        min_detection_confidence=0.3,  # Abbassa la soglia per catturare più keypoints
        min_tracking_confidence=0.3
    ) as pose:
        
        for img_path in sorted(glob.glob(os.path.join(gait_config.data_dir, '*.png'))):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[WARN] Impossibile caricare {img_path}")
                continue

            # Preprocess silhouette
            proc_img, mask = preprocess_silhouette(frame)
            
            # Auto-crop della silhouette con padding maggiore
            cropped_img, crop_box = auto_crop_silhouette(proc_img, mask, padding=50)
            
            # Converti in RGB per MediaPipe
            rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            
            # Esegui il rilevamento con MediaPipe
            results = pose.process(rgb_img)
            
            # Converti i risultati di MediaPipe nel formato OpenPose
            keypoints = mediapipe_to_openpose(results, rgb_img.shape)
            
            # Mappa i keypoints all'immagine originale
            orig_keypoints = map_keypoints_to_original(keypoints, crop_box)
            
            # Costruzione JSON stile OpenPose
            pose2d = []
            for x, y, c in orig_keypoints:
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
            vis_path = os.path.join(gait_config.save_dir, f"{base}_vis.png")
            cropped_path = os.path.join(gait_config.save_dir, f"{base}_cropped.png")
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            # Visualizzazione keypoints sull'immagine ritagliata
            vis_cropped = draw_keypoints(cropped_img, keypoints)
            cv2.imwrite(cropped_path, vis_cropped)
            
            # Visualizzazione keypoints sull'immagine originale
            vis_orig = draw_keypoints(frame, orig_keypoints)
            
            # Carica ground truth se disponibile
            gt_json_path = os.path.join(gt_dir, f"{base}.json") if has_ground_truth else None
            gt_keypoints = load_ground_truth(gt_json_path)
            
            if gt_keypoints:
                # Crea visualizzazione di confronto
                compare_path = os.path.join(gait_config.save_dir, f"{base}_compare.png")
                compare_img = compare_keypoints(frame, orig_keypoints, gt_keypoints)
                cv2.imwrite(compare_path, compare_img)
                
                # Calcola metriche di valutazione (distanza media tra keypoints corrispondenti)
                valid_pairs = [(i, gt_keypoints[i]) for i, kp in enumerate(orig_keypoints) 
                              if kp[2] > 0 and gt_keypoints[i][2] > 0]
                
                if valid_pairs:
                    distances = [np.sqrt((orig_keypoints[i][0] - gt[0])**2 + 
                                        (orig_keypoints[i][1] - gt[1])**2) 
                                for i, gt in valid_pairs]
                    avg_dist = np.mean(distances)
                    print(f"[INFO] {base}: Distanza media tra keypoints: {avg_dist:.2f} pixel")
                
                print(f"[INFO] Processata {base}: -> {json_path}, {vis_path}, {cropped_path}, {compare_path}")
            else:
                cv2.imwrite(vis_path, vis_orig)
                print(f"[INFO] Processata {base}: -> {json_path}, {vis_path}, {cropped_path}")
            
            # Stampa informazioni sui keypoints rilevati
            valid_keypoints = sum(1 for _, _, conf in orig_keypoints if conf > 0)
            print(f"[INFO] {base}: Rilevati {valid_keypoints}/18 keypoints validi")
            
            # Visualizza i risultati
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("Immagine Originale")
            
            plt.subplot(2, 2, 2)
            plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            plt.title("Silhouette Ritagliata")
            
            plt.subplot(2, 2, 3)
            plt.imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
            plt.title("Keypoints Rilevati")
            
            if gt_keypoints:
                plt.subplot(2, 2, 4)
                plt.imshow(cv2.cvtColor(compare_img, cv2.COLOR_BGR2RGB))
                plt.title("Confronto con Ground Truth")
            
            plt.tight_layout()
            plt.savefig(os.path.join(gait_config.save_dir, f"{base}_summary.png"))
            
            # Visualizza i risultati se richiesto
            plt.show()
            plt.close()

    print("[INFO] Elaborazione completata!")