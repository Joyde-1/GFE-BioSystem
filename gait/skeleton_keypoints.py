# -*- coding: utf-8 -*-
"""
Script per estrarre keypoints da immagini di silhouette di andatura
utilizzando tecniche di scheletrizzazione e morfologia matematica.

Funzionalità:
1. Ritaglio automatico della silhouette
2. Estrazione dello scheletro dalla silhouette
3. Identificazione dei keypoints dallo scheletro
4. Confronto con ground truth (se disponibile)
5. Salvataggio in formato JSON compatibile con OpenPose
"""
import os
import sys
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
from pathlib import Path

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_config, browse_path, path_extractor, save_image, load_checkpoint, save_checkpoint

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
    
    return img, mask

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
    
    # Ritaglia l'immagine e la maschera
    cropped_img = img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    return cropped_img, cropped_mask, (x, y, x+w, y+h)

def extract_skeleton(mask):
    """Estrae lo scheletro dalla maschera della silhouette"""
    # Converti in formato binario per skeletonize
    binary = mask > 0
    
    # Applica skeletonize
    skeleton = skeletonize(binary)
    
    # Converti in formato uint8 per OpenCV
    skeleton_img = np.uint8(skeleton * 255)
    
    return skeleton_img

def find_keypoints_from_skeleton(skeleton, mask, num_keypoints=18):
    """Trova i keypoints dallo scheletro utilizzando punti di interesse anatomici"""
    # Calcola la mappa delle distanze
    dist_transform = distance_transform_edt(mask)
    
    # Trova i punti estremi e le giunzioni nello scheletro
    # Questi sono potenziali keypoints
    kernel = np.ones((3,3), np.uint8)
    skeleton_dilated = cv2.dilate(skeleton, kernel, iterations=1)
    junctions = cv2.bitwise_and(skeleton_dilated, skeleton)
    
    # Trova i punti estremi (endpoints)
    endpoints = np.zeros_like(skeleton)
    for i in range(1, skeleton.shape[0]-1):
        for j in range(1, skeleton.shape[1]-1):
            if skeleton[i, j] > 0:
                neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2]) - skeleton[i, j]
                if neighbors == 1:
                    endpoints[i, j] = 255
    
    # Combina giunzioni e punti estremi
    keypoint_candidates = cv2.bitwise_or(junctions, endpoints)
    
    # Trova i massimi locali nella mappa delle distanze
    # Questi sono i punti più lontani dai bordi della silhouette
    peaks = peak_local_max(dist_transform, min_distance=10, num_peaks=num_keypoints//2)
    
    # Crea un'immagine con i picchi
    peak_img = np.zeros_like(skeleton)
    for peak in peaks:
        peak_img[peak[0], peak[1]] = 255
    
    # Combina tutti i candidati
    all_candidates = cv2.bitwise_or(keypoint_candidates, peak_img)
    
    # Trova le coordinate dei candidati
    y_coords, x_coords = np.where(all_candidates > 0)
    candidates = list(zip(x_coords, y_coords))
    
    # Se abbiamo troppi candidati, seleziona i più significativi
    if len(candidates) > num_keypoints:
        # Ordina per valore nella mappa delle distanze (più alto = più importante)
        candidates = sorted(candidates, key=lambda p: dist_transform[p[1], p[0]], reverse=True)
        candidates = candidates[:num_keypoints]
    
    # Se abbiamo troppo pochi candidati, aggiungi punti dallo scheletro
    if len(candidates) < num_keypoints:
        y_coords, x_coords = np.where(skeleton > 0)
        skeleton_points = list(zip(x_coords, y_coords))
        
        # Ordina per valore nella mappa delle distanze
        skeleton_points = sorted(skeleton_points, key=lambda p: dist_transform[p[1], p[0]], reverse=True)
        
        # Aggiungi punti fino a raggiungere il numero desiderato
        for point in skeleton_points:
            if point not in candidates:
                candidates.append(point)
                if len(candidates) >= num_keypoints:
                    break
    
    # Assegna confidenza 1.0 a tutti i keypoints trovati
    keypoints = [(x, y, 1.0) for x, y in candidates]
    
    # Assicurati di avere esattamente num_keypoints
    while len(keypoints) < num_keypoints:
        keypoints.append((0, 0, 0.0))
    
    # Ordina i keypoints in base alla posizione anatomica (dall'alto verso il basso)
    keypoints = sorted(keypoints, key=lambda kp: kp[1])
    
    # Identifica specifici keypoints anatomici
    h, w = mask.shape
    
    # Trova il punto più alto (testa)
    head_idx = 0
    
    # Trova il punto più basso (piedi)
    feet_idx = len(keypoints) - 1
    
    # Trova il punto centrale (bacino)
    mid_idx = len(keypoints) // 2
    
    # Riorganizza i keypoints per seguire il formato OpenPose
    organized_keypoints = [(0, 0, 0.0)] * num_keypoints
    
    # Assegna i keypoints principali
    if len(keypoints) > 0:
        organized_keypoints[0] = keypoints[head_idx]  # Naso
    
    if len(keypoints) > mid_idx:
        organized_keypoints[1] = keypoints[mid_idx - 2]  # Collo
    
    # Spalle
    if len(keypoints) > 2:
        left_shoulder_idx = min(mid_idx - 1, len(keypoints) - 1)
        right_shoulder_idx = min(mid_idx - 3, len(keypoints) - 1)
        organized_keypoints[2] = keypoints[right_shoulder_idx]  # Spalla destra
        organized_keypoints[5] = keypoints[left_shoulder_idx]   # Spalla sinistra
    
    # Gomiti
    if len(keypoints) > 4:
        left_elbow_idx = min(mid_idx, len(keypoints) - 1)
        right_elbow_idx = min(mid_idx - 4, len(keypoints) - 1)
        organized_keypoints[3] = keypoints[right_elbow_idx]  # Gomito destro
        organized_keypoints[6] = keypoints[left_elbow_idx]   # Gomito sinistro
    
    # Polsi
    if len(keypoints) > 6:
        left_wrist_idx = min(mid_idx + 1, len(keypoints) - 1)
        right_wrist_idx = min(mid_idx - 5, len(keypoints) - 1)
        organized_keypoints[4] = keypoints[right_wrist_idx]  # Polso destro
        organized_keypoints[7] = keypoints[left_wrist_idx]   # Polso sinistro
    
    # Bacino
    if len(keypoints) > mid_idx:
        organized_keypoints[8] = keypoints[mid_idx]  # Bacino
    
    # Anche
    if len(keypoints) > mid_idx + 2:
        organized_keypoints[9] = keypoints[mid_idx + 1]   # Anca destra
        organized_keypoints[12] = keypoints[mid_idx + 2]  # Anca sinistra
    
    # Ginocchia
    if len(keypoints) > mid_idx + 4:
        organized_keypoints[10] = keypoints[mid_idx + 3]  # Ginocchio destro
        organized_keypoints[13] = keypoints[mid_idx + 4]  # Ginocchio sinistro
    
    # Caviglie
    if len(keypoints) > feet_idx - 2:
        organized_keypoints[11] = keypoints[feet_idx - 1]  # Caviglia destra
        organized_keypoints[14] = keypoints[feet_idx]      # Caviglia sinistra
    
    # Occhi e orecchie (se disponibili)
    if head_idx + 1 < len(keypoints):
        organized_keypoints[15] = keypoints[head_idx + 1]  # Occhio destro
    
    if head_idx + 2 < len(keypoints):
        organized_keypoints[16] = keypoints[head_idx + 2]  # Occhio sinistro
    
    if head_idx + 3 < len(keypoints):
        organized_keypoints[17] = keypoints[head_idx + 3]  # Orecchio destro
    
    return organized_keypoints

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
    for img_path in sorted(glob.glob(os.path.join(gait_config.data_dir, '*.png'))):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARN] Impossibile caricare {img_path}")
            continue

        # Preprocess silhouette
        proc_img, mask = preprocess_silhouette(frame)
        
        # Auto-crop della silhouette con padding maggiore
        cropped_img, cropped_mask, crop_box = auto_crop_silhouette(proc_img, mask, padding=50)
        
        # Estrai lo scheletro dalla silhouette
        skeleton = extract_skeleton(cropped_mask)
        
        # Trova i keypoints dallo scheletro
        keypoints = find_keypoints_from_skeleton(skeleton, cropped_mask, num_keypoints=18)
        
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
        skeleton_path = os.path.join(gait_config.save_dir, f"{base}_skeleton.png")
        
        # with open(json_path, 'w') as f:
        #     json.dump(json_data, f, indent=2)
        
        # Visualizzazione keypoints sull'immagine ritagliata
        vis_cropped = draw_keypoints(cropped_img, keypoints)
        cv2.imwrite(cropped_path, vis_cropped)
        
        # Salva lo scheletro
        cv2.imwrite(skeleton_path, skeleton)
        
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
        plt.imshow(skeleton, cmap='gray')
        plt.title("Scheletro Estratto")
        
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