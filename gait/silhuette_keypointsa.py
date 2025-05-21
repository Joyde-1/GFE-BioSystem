# -*- coding: utf-8 -*-
"""
Script per estrarre keypoints da immagini di silhouette di andatura
utilizzando un approccio ibrido di scheletrizzazione e machine learning.

Funzionalità:
1. Ritaglio automatico della silhouette
2. Estrazione dello scheletro e caratteristiche morfologiche
3. Predizione dei keypoints tramite modello di regressione addestrato sui ground truth
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
from skimage.morphology import skeletonize, thin
from skimage.feature import peak_local_max
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

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
        return img, mask, (0, 0, img.shape[1], img.shape[0])  # Nessun contorno trovato
    
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

def extract_contour(mask):
    """Estrae il contorno dalla maschera della silhouette"""
    # Trova i contorni
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crea un'immagine vuota per il contorno
    contour_img = np.zeros_like(mask)
    
    # Disegna il contorno
    cv2.drawContours(contour_img, contours, -1, 255, 1)
    
    return contour_img

def extract_features(skeleton, mask, contour):
    """Estrae caratteristiche morfologiche dalla silhouette"""
    # Calcola la mappa delle distanze
    dist_transform = distance_transform_edt(mask)
    
    # Trova i punti estremi e le giunzioni nello scheletro
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
    
    # Trova i massimi locali nella mappa delle distanze
    peaks = peak_local_max(dist_transform, min_distance=10, num_peaks=20)
    
    # Crea un'immagine con i picchi
    peak_img = np.zeros_like(skeleton)
    for peak in peaks:
        peak_img[peak[0], peak[1]] = 255
    
    # Calcola i momenti della silhouette per trovare il centro di massa
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = mask.shape[1] // 2, mask.shape[0] // 2
    
    # Calcola le caratteristiche di forma
    h, w = mask.shape
    aspect_ratio = float(w) / h
    area = cv2.countNonZero(mask)
    perimeter = cv2.countNonZero(contour)
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
    
    # Restituisci le caratteristiche estratte
    features = {
        'dist_transform': dist_transform,
        'junctions': junctions,
        'endpoints': endpoints,
        'peaks': peak_img,
        'center_mass': (cX, cY),
        'shape_features': {
            'aspect_ratio': aspect_ratio,
            'area': area,
            'perimeter': perimeter,
            'compactness': compactness
        }
    }
    
    return features

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

def load_alphapose_ground_truth(json_path):
    """Carica i keypoints di ground truth da un file JSON in formato AlphaPose"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            # Formato AlphaPose
            keypoints = data[0]['keypoints']
            # Converti la lista piatta in una lista di tuple (x, y, conf)
            return [(keypoints[i], keypoints[i+1], keypoints[i+2]) 
                    for i in range(0, len(keypoints), 3)]
        return None
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None

def create_feature_vector(features, mask_shape):
    """Crea un vettore di caratteristiche per il modello di machine learning"""
    h, w = mask_shape
    
    # Normalizza le coordinate del centro di massa
    cX, cY = features['center_mass']
    cX_norm = cX / w
    cY_norm = cY / h
    
    # Estrai i punti di interesse (junctions, endpoints, peaks)
    junctions_y, junctions_x = np.where(features['junctions'] > 0)
    endpoints_y, endpoints_x = np.where(features['endpoints'] > 0)
    peaks_y, peaks_x = np.where(features['peaks'] > 0)
    
    # Normalizza le coordinate
    junctions_x_norm = junctions_x / w if len(junctions_x) > 0 else []
    junctions_y_norm = junctions_y / h if len(junctions_y) > 0 else []
    endpoints_x_norm = endpoints_x / w if len(endpoints_x) > 0 else []
    endpoints_y_norm = endpoints_y / h if len(endpoints_y) > 0 else []
    peaks_x_norm = peaks_x / w if len(peaks_x) > 0 else []
    peaks_y_norm = peaks_y / h if len(peaks_y) > 0 else []
    
    # Crea un vettore fisso di caratteristiche
    feature_vector = [
        cX_norm, cY_norm,
        features['shape_features']['aspect_ratio'],
        features['shape_features']['compactness']
    ]
    
    # Aggiungi un numero fisso di punti di interesse (max 10 per tipo)
    for points_x, points_y in [(junctions_x_norm, junctions_y_norm), 
                              (endpoints_x_norm, endpoints_y_norm), 
                              (peaks_x_norm, peaks_y_norm)]:
        for i in range(10):
            if i < len(points_x):
                feature_vector.extend([points_x[i], points_y[i]])
            else:
                feature_vector.extend([0, 0])  # Padding
    
    return np.array(feature_vector)

def prepare_training_data(images_dir, gt_dir, format='openpose'):
    """Prepara i dati di addestramento dal dataset di immagini e ground truth"""
    X = []  # Caratteristiche
    Y = []  # Keypoints target
    
    for img_path in sorted(glob.glob(os.path.join(images_dir, '*.png'))):
        base = os.path.basename(img_path).rsplit('.',1)[0]
        
        # Carica l'immagine
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARN] Impossibile caricare {img_path}")
            continue
        
        # Preprocess silhouette
        proc_img, mask = preprocess_silhouette(frame)
        
        # Auto-crop della silhouette
        cropped_img, cropped_mask, crop_box = auto_crop_silhouette(proc_img, mask, padding=50)
        
        # Estrai lo scheletro e il contorno
        skeleton = extract_skeleton(cropped_mask)
        contour = extract_contour(cropped_mask)
        
        # Estrai caratteristiche
        features = extract_features(skeleton, cropped_mask, contour)
        
        # Crea vettore di caratteristiche
        feature_vector = create_feature_vector(features, cropped_mask.shape)
        
        # Carica ground truth
        if format == 'openpose':
            gt_json_path = os.path.join(gt_dir, f"{base}.json")
            gt_keypoints = load_ground_truth(gt_json_path)
        else:  # alphapose
            gt_json_path = os.path.join(gt_dir, f"{base}.json")
            gt_keypoints = load_alphapose_ground_truth(gt_json_path)
        
        if gt_keypoints:
            # Normalizza le coordinate dei keypoints rispetto all'immagine ritagliata
            x_min, y_min, _, _ = crop_box
            normalized_keypoints = []
            
            for x, y, conf in gt_keypoints:
                if conf > 0:
                    # Normalizza le coordinate rispetto all'immagine ritagliata
                    norm_x = (x - x_min) / cropped_mask.shape[1]
                    norm_y = (y - y_min) / cropped_mask.shape[0]
                    normalized_keypoints.extend([norm_x, norm_y, conf])
                else:
                    normalized_keypoints.extend([0, 0, 0])
            
            # Aggiungi ai dati di addestramento
            X.append(feature_vector)
            Y.append(normalized_keypoints)
    
    return np.array(X), np.array(Y)

def train_keypoint_model(X, Y):
    """Addestra un modello di regressione per predire i keypoints"""
    # Dividi in training e validation set
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Crea un modello per ogni keypoint (x, y, conf)
    models = []
    
    # Per ogni tripla (x, y, conf) di ogni keypoint
    for i in range(Y.shape[1]):
        print(f"Addestramento modello per la caratteristica {i+1}/{Y.shape[1]}")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, Y_train[:, i])
        models.append(model)
    
    # Valuta il modello
    mse = 0
    for i, model in enumerate(models):
        pred = model.predict(X_val)
        mse += np.mean((Y_val[:, i] - pred) ** 2)
    
    mse /= len(models)
    print(f"MSE medio: {mse}")
    
    return models

def predict_keypoints(models, feature_vector, mask_shape):
    """Predice i keypoints usando il modello addestrato"""
    # Predici ogni coordinata
    predictions = []
    for model in models:
        pred = model.predict([feature_vector])[0]
        predictions.append(pred)
    
    # Riorganizza le predizioni in formato (x, y, conf)
    h, w = mask_shape
    keypoints = []
    
    for i in range(0, len(predictions), 3):
        if i+2 < len(predictions):
            x_norm = predictions[i]
            y_norm = predictions[i+1]
            conf = predictions[i+2]
            
            # Denormalizza le coordinate
            x = int(x_norm * w)
            y = int(y_norm * h)
            
            keypoints.append((x, y, conf))
    
    return keypoints

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
    
    # Cartella per i ground truth
    gt_dir = gait_config.gt_dir if hasattr(gait_config, 'gt_dir') else os.path.join(gait_config.data_dir, 'ground_truth')
    has_ground_truth = False #os.path.exists(gt_dir)
    
    # Formato dei ground truth (openpose o alphapose)
    gt_format = gait_config.gt_format if hasattr(gait_config, 'gt_format') else 'openpose'
    
    # Percorso del modello addestrato
    model_path = os.path.join(gait_config.save_dir, 'keypoint_models.joblib')
    
    # Verifica se esiste già un modello addestrato
    if os.path.exists(model_path) and not gait_config.retrain_model:
        print("[INFO] Caricamento del modello esistente...")
        models = joblib.load(model_path)
    elif has_ground_truth:
        print("[INFO] Addestramento di un nuovo modello...")
        # Prepara i dati di addestramento
        X, Y = prepare_training_data(gait_config.data_dir, gt_dir, format=gt_format)
        
        if len(X) > 0:
            # Addestra il modello
            models = train_keypoint_model(X, Y)
            
            # Salva il modello
            joblib.dump(models, model_path)
            print(f"[INFO] Modello salvato in {model_path}")
        else:
            print("[ERROR] Nessun dato di addestramento disponibile!")
            sys.exit(1)
    else:
        print("[ERROR] Nessun ground truth disponibile e nessun modello pre-addestrato!")
        sys.exit(1)
    
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
        
        # Auto-crop della silhouette
        cropped_img, cropped_mask, crop_box = auto_crop_silhouette(proc_img, mask, padding=50)
        
        # Estrai lo scheletro e il contorno
        skeleton = extract_skeleton(cropped_mask)
        contour = extract_contour(cropped_mask)
        
        # Estrai caratteristiche
        features = extract_features(skeleton, cropped_mask, contour)
        
        # Crea vettore di caratteristiche
        feature_vector = create_feature_vector(features, cropped_mask.shape)
        
        # Predici i keypoints
        keypoints = predict_keypoints(models, feature_vector, cropped_mask.shape)
        
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
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Visualizzazione keypoints sull'immagine ritagliata
        vis_cropped = draw_keypoints(cropped_img, keypoints)
        cv2.imwrite(cropped_path, vis_cropped)
        
        # Salva lo scheletro
        cv2.imwrite(skeleton_path, skeleton)
        
        # Visualizzazione keypoints sull'immagine originale
        vis_orig = draw_keypoints(frame, orig_keypoints)
        
        # Carica ground truth se disponibile
        if gt_format == 'openpose':
            gt_json_path = os.path.join(gt_dir, f"{base}.json") if has_ground_truth else None
            gt_keypoints = load_ground_truth(gt_json_path)
        else:  # alphapose
            gt_json_path = os.path.join(gt_dir, f"{base}.json") if has_ground_truth else None
            gt_keypoints = load_alphapose_ground_truth(gt_json_path)
        
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
        if hasattr(gait_config, 'show_results') and gait_config.show_results:
            plt.show()
        plt.close()

    print("[INFO] Elaborazione completata!")