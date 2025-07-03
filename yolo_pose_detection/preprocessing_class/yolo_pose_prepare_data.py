import cv2
from tqdm import tqdm
import os
import json
import glob
import shutil
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


class PrepareData():
    """
    Prepares detection data for deep learning model.

    Attributes
    ----------
    data_paths : str
        Base directory where detection data is stored.
    data : Dict
        This dictionary holds the processed data. It contains two keys: 'images' and 'labels'.
    """

    def __init__(self, yolo_pose_detection_config, biometric_trait):
        """
        Initializes the PrepareData instance

        Parameters
        ----------
        data_paths : str
            The path to the base directory containing database
        """
        
        self._yolo_pose_detection_config = yolo_pose_detection_config
        self.biometric_trait = biometric_trait

        # Initialize counters for COCO IDs
        self._next_image_id = 0
        self._next_ann_id = 0

    def _load_silhouette_paths(self):
        """
        Carica i percorsi di tutte le silhouette per i soggetti da 00001 a 00049
        da tutte le cartelle Silhouette_* disponibili.
        
        Returns
        -------
        dict
            Dizionario con soggetti come chiavi e liste di percorsi come valori
        """
        silhouette_data = {
            'frame_paths': [],
            'subjects': [],
            'subject_to_paths': {}
        }
        
        # Cerca tutte le cartelle Silhouette_* nella directory dei dati
        silhouette_dirs = glob.glob(os.path.join(self._yolo_pose_detection_config.frames_dir, "Silhouette_*"))
        
        if not silhouette_dirs:
            raise FileNotFoundError(f"Nessuna directory Silhouette_* trovata in {self._yolo_pose_detection_config.frames_dir}")
        
        # Per ogni cartella Silhouette_*
        for silhouette_dir in silhouette_dirs:
            # print(f"Elaborazione della directory: {silhouette_dir}")
            
            # Per ogni soggetto da 00001 a 00049
            for subject_num in range(1, 50):   #1, 50
                subject_id = f"{subject_num:05d}"  # Formatta come 00001, 00002, ecc.
                
                # Cerca tutte le immagini per questo soggetto nella cartella corrente
                subject_pattern = os.path.join(silhouette_dir, subject_id, "*.png")
                subject_files = glob.glob(subject_pattern)

                # print("Silhouette files found for subject:", subject_id)
                # print(subject_files)
                
                # Se abbiamo trovato file per questo soggetto
                for file_path in subject_files:
                    silhouette_data['frame_paths'].append(file_path)
                    silhouette_data['subjects'].append(subject_id)
                    
                    # Aggiungi al dizionario soggetto -> percorsi
                    if subject_id not in silhouette_data['subject_to_paths']:
                        silhouette_data['subject_to_paths'][subject_id] = []
                    
                    silhouette_data['subject_to_paths'][subject_id].append(file_path)
        
        # Verifica che abbiamo trovato dati
        if not silhouette_data['frame_paths']:
            raise FileNotFoundError(f"Nessuna silhouette trovata per i soggetti da 00001 a 00049")
        
        print(f"Trovate {len(silhouette_data['frame_paths'])} silhouette per {len(silhouette_data['subject_to_paths'])} soggetti")
        print("subjects:", len(silhouette_data['subjects']))
        
        return silhouette_data
    
    def _data_splitting(self, data):
        """
        Divide i dati in set di training, validation e test assicurando che
        ogni soggetto appaia solo in uno dei tre subset.
    
        Parameters
        ----------
        data : dict
            Dizionario contenente 'image_names' e 'subjects' come chiavi.
    
        Returns
        -------
        tuple
            Restituisce tre liste: nomi delle immagini per training, validation e test.
        """
        # Ottieni soggetti unici
        unique_subjects = list(set(data['subjects']))
        
        # Dividi i soggetti in train, validation e test
        train_subjects, test_subjects = train_test_split(
            unique_subjects, 
            test_size=self._yolo_pose_detection_config.data.test_size,
            random_state=42, 
            shuffle=True
        )

        # Dividi i soggetti rimanenti tra validation e test
        train_subjects, val_subjects = train_test_split(
            train_subjects, 
            test_size=self._yolo_pose_detection_config.data.val_size,
            random_state=42, 
            shuffle=True
        )
        
        # Inizializza le liste per i nomi delle immagini
        train_set_frame_paths = []
        val_set_frame_paths = []
        test_set_frame_paths = []
        
        # Assegna ogni immagine al set appropriato in base al soggetto
        for idx, subject in enumerate(data['subjects']):
            frame_path = data['frame_paths'][idx]
            
            if subject in train_subjects:
                train_set_frame_paths.append(frame_path)
            elif subject in val_subjects:
                val_set_frame_paths.append(frame_path)
            else:  # subject in test_subjects
                test_set_frame_paths.append(frame_path)
        
        print(f"Split completato: {len(train_set_frame_paths)} immagini di training, "
              f"{len(val_set_frame_paths)} immagini di validation, "
              f"{len(test_set_frame_paths)} immagini di test")
        print(f"Soggetti: {len(train_subjects)} in training, "
              f"{len(val_subjects)} in validation, "
              f"{len(test_subjects)} in test")
        
        return train_set_frame_paths, val_set_frame_paths, test_set_frame_paths
    
    def _preprocess_silhouette(self, img):
        """Preprocessa l'immagine della silhouette per migliorare il rilevamento"""
        # Partiamo da BGR o grayscale
        if len(img.shape) < 3 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Rimuoviamo artefatti con blur + threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        # Manteniamo silhouette su sfondo neutro (127 invece di 0)
        # Questo aiuta il modello a rilevare meglio i contorni
        bg = np.full_like(img, 127)
        fg = cv2.bitwise_and(img, img, mask=mask)

        # cv2.imshow("_preprocess_silhouette", cv2.bitwise_or(fg, bg))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return cv2.bitwise_or(fg, bg), mask
    
    def _crop_silhouette(self, img, mask, padding=20):
        """Ritaglia automaticamente la silhouette con un padding"""
        # Trova i contorni della silhouette
        #  img, mask = self._preprocess_silhouette(img.copy())

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img, (0, 0, img.shape[1], img.shape[0])  # Nessun contorno trovato
        
        # Prendi il contorno più grande (dovrebbe essere la silhouette)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Aggiungi padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        # Ritaglia l'immagine
        cropped = img[y:y+h, x:x+w]

        # cv2.imshow("_crop_silhouette", cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return cropped, (x, y, x+w, y+h)
    
    def _letterbox(self, img, target_size=(640, 640), color=(127,127,127)):
        """Ridimensiona img preservando aspect ratio, poi pad fino a target_size."""
        # ih, iw = target_size[1], target_size[0]  # nota: target=(W,H)
        ih = self._yolo_pose_detection_config.data.image_height
        iw = self._yolo_pose_detection_config.data.image_width
        h, w = img.shape[:2]
        scale = min(iw / w, ih / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((ih, iw, 3), color, dtype=np.uint8)
        pad_x = (iw - nw) // 2
        pad_y = (ih - nh) // 2
        canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized

        # cv2.imshow("_letterbox", canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return canvas, scale, pad_x, pad_y
    
    def _remap_keypoints(self, keypoints, crop_bbox, scale, pad_x, pad_y):
        """
        keypoints: flat list [x1,y1,c1, x2,y2,c2, ...]
        crop_bbox: (x0,y0,x1,y1) in original frame
        """
        x0, y0, x1, y1 = crop_bbox
        remapped = []
        for x, y, c in zip(keypoints[0::3], keypoints[1::3], keypoints[2::3]):
            # Verifica se il keypoint cade all'interno del bounding box del crop e ha confidenza > 0
            if c > 0 and x >= x0 and x <= x1 and y >= y0 and y <= y1:
                # trasla nella ROI
                xr = x - x0
                yr = y - y0
                # scala e pad
                x_new = xr * scale + pad_x
                y_new = yr * scale + pad_y

                # Clamping per rimanere all'interno del range [0, target_size]
                x_check = max(0.0, min(x_new, self._yolo_pose_detection_config.data.image_width))
                y_check = max(0.0, min(y_new, self._yolo_pose_detection_config.data.image_height))

                if x_check < 0 or x_check > self._yolo_pose_detection_config.data.image_width or \
                    y_check < 0 or y_check > self._yolo_pose_detection_config.data.image_height:
                    print(f"[ERROR] Keypoint fuori dai limiti: ({x_new:.1f}, {y_new:.1f}) "
                            f"in frame {self._yolo_pose_detection_config.data.image_width}x{self._yolo_pose_detection_config.data.image_height}")
                    # Fuori dal crop: rendi c=0
                    remapped.extend([0.0, 0.0, 0.0])
                else:
                    remapped.extend([x_new, y_new, c])
            else:
                # Fuori dal crop o non visibile: rendi c=0
                remapped.extend([0.0, 0.0, 0.0])
        return remapped
    
    def _preprocess_image_and_keypoints(self, frame, keypoints_data, frame_filename):
        # Estrai il nome del file dal percorso
        # frame_filename = os.path.basename(frame_path)
        # frame_name = os.path.splitext(frame_filename)[0]

        # Assign unique numeric COCO IDs
        image_id = self._next_image_id
        ann_id = self._next_ann_id
        self._next_image_id += 1
        self._next_ann_id += 1

        # 1) preprocess + mask
        proc, mask = self._preprocess_silhouette(frame)
        # 2) crop
        cropped, crop_bbox = self._crop_silhouette(proc, mask)
        # 3) letterbox
        letterboxed, scale, pad_x, pad_y = self._letterbox(cropped)

        # Crea l'informazione dell'immagine con le dimensioni target
        image_info = {
            "id": image_id,
            "file_name": frame_filename,
            "width": self._yolo_pose_detection_config.data.image_width,
            "height": self._yolo_pose_detection_config.data.image_height
        }
        
        # Prendi i keypoints della prima persona (assumiamo che ci sia solo una persona per frame)
        person = keypoints_data["people"][0]
        
        # Estrai i keypoints 2D della posa
        pose_keypoints = person.get("pose_keypoints_2d", [])
        
        # Convert 18 OpenPose keypoints to 17 COCO keypoints by dropping 'neck' and reordering
        op_to_coco = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
        mapped_kps = []
        for idx in op_to_coco:
            x, y, v = pose_keypoints[3*idx:3*idx+3]
            mapped_kps.extend([x, y, v])
        pose_keypoints = mapped_kps

        # Verifica che ci siano abbastanza keypoints (17 keypoints * 3 valori = 51)
        if len(pose_keypoints) < 51:
            print(f"Numero insufficiente di keypoints nel frame: {frame_filename}")
            return None, None

        # 4) remap keypoints
        remapped_keypoints = self._remap_keypoints(
            pose_keypoints, crop_bbox, scale, pad_x, pad_y
        )

        # Debug: ensure exactly 17 keypoints after remapping
        if len(remapped_keypoints) != 17 * 3:
            print(f"[ERROR] Remapped keypoints length {len(remapped_keypoints)} "
                  f"!= 51 for frame {frame_filename}")
            return None, None

        # Calcola la bounding box che racchiude tutti i keypoints rimappati
        xs = []
        ys = []
        for i in range(0, len(remapped_keypoints), 3):
            x, y, conf = remapped_keypoints[i:i+3]
            if conf > 0:  # Considera solo i keypoints con confidenza > 0
                xs.append(x)
                ys.append(y)
        
        if not xs or not ys:
            print(f"Nessun keypoint valido nel frame: {frame_filename}")
            return None, None
        
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        if bbox_width > 640:
            print(f"[WARNING] Bounding box WIDTH {bbox_width:.1f} exceeds 640 pixels in frame {frame_filename}")
        if bbox_height > 640:
            print(f"[WARNING] Bounding box HEIGHT {bbox_height:.1f} exceeds 640 pixels in frame {frame_filename}")
        if x_min < 0:
            print(f"[WARNING] Bounding box X_MIN {x_min:.1f} is negative in frame {frame_filename}")
        if y_min < 0:
            print(f"[WARNING] Bounding box Y_MIN {y_min:.1f} is negative in frame {frame_filename}")
        
        # Conta il numero di keypoints validi
        num_keypoints = sum(1 for i in range(0, len(remapped_keypoints), 3) if remapped_keypoints[i+2] > 0)
        
        # Crea l'informazione dell'annotazione
        annotation_info = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": 0,  # Categoria "person"
            "keypoints": remapped_keypoints,
            "num_keypoints": num_keypoints,
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "area": bbox_width * bbox_height,
            "iscrowd": 0
        }
        
        # Crea il dizionario COCO
        coco_dict = {
            "images": [image_info],
            "annotations": [annotation_info],
            "categories": [
                {
                    "id": 0,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ],
                    "skeleton": [
                        [15, 13],  # left_ankle  ←→ left_knee
                        [13, 11],  # left_knee   ←→ left_hip
                        [16, 14],  # right_ankle ←→ right_knee
                        [14, 12],  # right_knee  ←→ right_hip
                        [11, 12],  # left_hip    ←→ right_hip
                        [5, 11],   # left_shoulder ←→ left_hip
                        [6, 12],   # right_shoulder←→ right_hip
                        [5, 6],    # left_shoulder←→ right_shoulder
                        [5, 7],    # left_shoulder←→ left_elbow
                        [7, 9],    # left_elbow  ←→ left_wrist
                        [6, 8],    # right_shoulder←→ right_elbow
                        [8, 10],   # right_elbow ←→ right_wrist
                        [1, 2],    # left_eye    ←→ right_eye
                        [0, 1],    # nose        ←→ left_eye
                        [0, 2],    # nose        ←→ right_eye
                        # [0, 5],    # nose        ←→ left_shoulder
                        # [0, 6]    # nose        ←→ right_shoulder
                        [1, 3],  # left_eye–left_ear  
                        [2, 4],  # right_eye–right_ear  
                        [3, 5],  # left_ear–left_shoulder  
                        [4, 6]  # right_ear–right_shoulder  
                    ]
                }
            ]
        }

        return letterboxed, coco_dict
    
    def _visualize_keypoints(self, keypoints_data, frame):
        """
        Visualizza i keypoints OpenPose sul frame.
        
        Parameters
        ----------
        keypoints_data : dict
            Dizionario contenente i keypoints in formato COCO
        frame : numpy.ndarray
            Immagine del frame su cui visualizzare i keypoints
        """
        # Verifica che ci siano annotazioni
        if "annotations" not in keypoints_data or not keypoints_data["annotations"]:
            print("Nessuna annotazione trovata nei dati dei keypoints")
            return
        
        # Ottieni i keypoints dalla prima annotazione
        annotation = keypoints_data["annotations"][0]
        keypoints = annotation["keypoints"]
        
        # Ottieni i nomi dei keypoints dalla categoria
        keypoint_names = []
        if "categories" in keypoints_data and keypoints_data["categories"]:
            keypoint_names = keypoints_data["categories"][0].get("keypoints", [])
        
        # Crea una copia dell'immagine per disegnare sopra
        image = frame.copy()
        
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
                cv2.circle(image, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
                
                # Se sono disponibili i nomi, li disegna accanto al punto
                if i < len(keypoint_names):
                    cv2.putText(image, keypoint_names[i], (x + 5, y - 5),
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
                cv2.line(image, pt1, pt2, colors[part], 2)
        
        # Mostra l'immagine con i keypoints
        cv2.imshow("Frame con Keypoints", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _coco_to_yolo_txt(self, coco_keypoints):
        """
        Converte una singola annotazione COCO (coco_dict contenente un'immagine e una annotazione)
        in formato YOLO TXT e salva il file .txt nella directory keypoints del subset.
        """
        # Estrai informazioni immagine e annotazione
        image_info = coco_keypoints["images"][0]
        ann_info = coco_keypoints["annotations"][0]

        # Dimensioni target (usate durante il letterbox)
        img_w = image_info["width"]
        img_h = image_info["height"]

        # Bounding box (pixel) e normalizzazione
        x_min, y_min, box_w, box_h = ann_info["bbox"]
        cx = x_min + box_w / 2
        cy = y_min + box_h / 2
        cx_norm = cx / img_w
        cy_norm = cy / img_h
        w_norm = box_w / img_w
        h_norm = box_h / img_h

        # Keypoints flat [x1, y1, v1, ..., x17, y17, v17]
        kpt = ann_info["keypoints"]
        # Prepara lista di coordinate normalizzate [x1, y1, ..., x17, y17]
        kpt_xyv_norm = []
        for i in range(0, len(kpt), 3):
            x_i, y_i, v_i = kpt[i], kpt[i+1], kpt[i+2]
            if v_i > 0:
                x_norm = x_i / img_w
                y_norm = y_i / img_h
                v_norm = 2
            else:
                x_norm = 0.0
                y_norm = 0.0
                v_norm = 0
            kpt_xyv_norm.extend([x_norm, y_norm, v_norm])

        # Componi la riga di testo: classe=0 sempre
        values = [0, cx_norm, cy_norm, w_norm, h_norm] + kpt_xyv_norm
        yolo_keypoints = " ".join([str(v) for v in values])

        if len(values) != 5 + 3 * 17:
            print(f"[SKIP] Frame {image_info['file_name']} ha {len(values)} valori, mi aspetto {5 + 3*17}. Salto.")
            import sys  # oppure return letterboxed, None, e poi non scrivi il file
            sys.exit(1)

        return yolo_keypoints
    
    def _load_frames_and_keypoints(self, subset_frame_paths, subset, return_data=False):
        """
        Carica i file di keypoints per i frame specificati dalla cartella OUMVLP-Pose.
        
        Parameters
        ----------
        subset_frame_paths : list
            Lista dei percorsi dei frame per cui caricare i keypoints
            
        Returns
        -------
        dict
            Dizionario contenente i frame caricati e i relativi keypoints
        """
        subset_data = {
            'frames': [],
            'keypoints': []
        }

        frames_count = 0
        keypoints_count = 0
        
        # Percorso base della cartella dei keypoints
        keypoints_base_dir = os.path.join(self._yolo_pose_detection_config.keypoints_dir, "00")
        
        if not os.path.exists(keypoints_base_dir):
            raise FileNotFoundError(f"Directory dei keypoints {keypoints_base_dir} non trovata.")
        
        # Itera su tutti i percorsi dei frame
        for frame_path in tqdm(subset_frame_paths, desc="Caricamento frame e keypoints", unit="frame"):
            # Estrai il soggetto e il tipo di sequenza dal percorso del frame
            # Esempio: /path/to/Silhouette_00-01/00001/000_00_180_090/0001.png
            parts = frame_path.split(os.sep)
            
            # Verifica che il percorso abbia la struttura attesa
            if len(parts) < 4:
                print(f"Percorso del frame non valido: {frame_path}")
                continue

            # print(f"Percorso del frame: {frame_path}")
            # print(f"Numero di parti: {len(parts)}")
            # print(f"Parti: {parts}")
            
            # Estrai il soggetto (es. 00001)
            subject_id = None
            for part in parts:
                if part.isdigit() and len(part) == 5 and 1 <= int(part) <= 49:
                    subject_id = part
                    # print(f"ID soggetto trovato: {subject_id}")
                    break

            if not subject_id:
                print(f"ID soggetto non trovato nel percorso: {frame_path}")
                continue
                
            # Estrai il nome della sequenza (es. 000_00_180_090)
            sequence_name = None
            for i, part in enumerate(parts):
                if part == subject_id and i+1 < len(parts):
                    sequence_name = parts[i-1].split('_')[1].split('-')[0] + '_' + parts[i-1].split('_')[1].split('-')[1]
                    #sequence_name = parts[i+1]
                    break
            # print(f"Nome sequenza trovato: {sequence_name}")
            
            if not sequence_name:
                print(f"Nome sequenza non trovato nel percorso: {frame_path}")
                continue

            # Estrai il nome del file (es. 0001.png)
            frame_filename = os.path.basename(frame_path)
            frame_name = os.path.splitext(frame_filename)[0]
            
            # Costruisci il percorso del file dei keypoints
            # La struttura attesa è: OUMVLP-Pose/00/00001/000_00_180_090/0001.json
            keypoints_path = os.path.join(keypoints_base_dir, subject_id, sequence_name, f"{frame_name}_keypoints.json")
            # print(f"Percorso keypoints: {keypoints_path}")
            
            # Verifica se esiste il file dei keypoints
            if not os.path.exists(keypoints_path):
                # Se non esiste, salta questo frame
                # print(f"File keypoints non trovato: {keypoints_path}")
                continue
            
            # Carica l'immagine del frame
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"Impossibile caricare il frame: {frame_path}")
                continue
            
            # Carica il file JSON dei keypoints
            try:
                with open(keypoints_path, 'r') as f:
                    keypoints = json.load(f)
                
                # Converti i keypoints da OpenPose a COCO
                frame, keypoints = self._preprocess_image_and_keypoints(frame, keypoints, f"{subject_id}_{sequence_name}_{frame_name}.png")
                
                # Se la conversione ha avuto successo, aggiungi il frame e i keypoints ai risultati
                if keypoints:
                    if return_data:
                        subset_data['frames'].append(frame)
                        subset_data['keypoints'].append(keypoints)
                    else:
                        frames_count += 1
                        keypoints_count += 1
                else:
                    # Se la conversione fallisce, salta questo frame
                    print(f"Conversione fallita per il frame: {frame_path}")
                    continue
                
            except Exception as e:
                print(f"Errore nel caricamento del file keypoints {keypoints_path}: {e}")
                continue

            # if f"{subject_id}_{sequence_name}_{frame_name}" == "00014_270_01_0037" or f"{subject_id}_{sequence_name}_{frame_name}" == "00033_180_00_0028":
            #     # Visualizza i keypoints sul frame
            #     self._visualize_keypoints(keypoints, frame.copy())
                
            # self._visualize_keypoints(keypoints, frame.copy())

            if len(frame.shape) < 3 or frame.shape[2] == 1:
                print(f"[ERROR IMAGE SHAPE] Il frame {frame_filename} ha shape {frame.shape}")
                import sys
                sys.exit(1)
            if frame.ndim != 3 or frame.shape[2] != 3:
                print(f"[ERROR IMAGE CHANNELS] Il frame {frame_filename} ha shape {frame.shape}")
                import sys
                sys.exit(1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = Image.fromarray(frame)#.resize((self._yolo_pose_detection_config.data.image_size, self._yolo_pose_detection_config.data.image_size))

            yolo_keypoints = self._coco_to_yolo_txt(keypoints)

            # Salva l'immagine e i gait_keypoints in formato COCO nella directory corrispondente
            subset_image_path = os.path.join(self._yolo_pose_detection_config.save_data_splitted_path, "splitted_yolo_pose_database", self.biometric_trait, subset, "images", f"{subject_id}_{sequence_name}_{frame_name}.png")
            subset_gait_keypoints_path = os.path.join(self._yolo_pose_detection_config.save_data_splitted_path, "splitted_yolo_pose_database", self.biometric_trait, subset, "labels", f"{subject_id}_{sequence_name}_{frame_name}.txt")
            
            # Salva l'immagine nella directory corrispondente
            frame.save(subset_image_path)

            # Salva i gait_keypoints
            with open(subset_gait_keypoints_path, 'w') as f:
                f.write(yolo_keypoints)

        if return_data:
            print(f" - {subset} subset: caricati {len(subset_data['frames'])} frame con {len(subset_data['keypoints'])} keypoints corrispondenti")
            return subset_data
        else:
            print(f" - {subset} subset: caricati {frames_count} frame con {keypoints_count} keypoints corrispondenti")

    def prepare_data(self):
        # Carica i percorsi delle silhouette
        silhouette_data = self._load_silhouette_paths()
        
        # Esegui lo split dei dati
        train_set_frame_paths, val_set_frame_paths, test_set_frame_paths = self._data_splitting(silhouette_data)

        for subset in ['train', 'val', 'test']:
            subset_path = os.path.join(self._yolo_pose_detection_config.save_data_splitted_path, "splitted_yolo_pose_database", self.biometric_trait, subset)

            os.makedirs(subset_path, exist_ok=True)
    
            os.makedirs(os.path.join(subset_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(subset_path, "labels"), exist_ok=True)
        
        # Carica i keypoints per ciascun subset
        self._load_frames_and_keypoints(train_set_frame_paths, "train")
        self._load_frames_and_keypoints(val_set_frame_paths, "val")
        self._load_frames_and_keypoints(test_set_frame_paths, "test")