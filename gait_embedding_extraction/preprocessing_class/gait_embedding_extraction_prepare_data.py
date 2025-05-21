import cv2
from tqdm import tqdm
import os
import json
import glob
import numpy as np
import pandas as pd
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

    fixed_length = 0

    def __init__(self, gait_keypoints_detection_config):
        """
        Initializes the PrepareData instance

        Parameters
        ----------
        data_paths : str
            The path to the base directory containing database
        """
        
        self._gait_keypoints_detection_config = gait_keypoints_detection_config

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

        lengths = []
        
        # Cerca tutte le cartelle Silhouette_* nella directory dei dati
        silhouette_dirs = glob.glob(os.path.join(self._gait_keypoints_detection_config.frames_dir, "Silhouette_*"))

        # Percorso base della cartella dei keypoints
        keypoints_base_dir = os.path.join(self._gait_keypoints_detection_config.keypoints_sequences_dir, "00")
        
        if not os.path.exists(keypoints_base_dir):
            raise FileNotFoundError(f"Directory dei keypoints {keypoints_base_dir} non trovata.")
        
        if not silhouette_dirs:
            raise FileNotFoundError(f"Nessuna directory Silhouette_* trovata in {self._gait_keypoints_detection_config.frames_dir}")
        
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

                sequence_lenght = 0
                
                # Se abbiamo trovato file per questo soggetto
                for file_path in subject_files:
                    silhouette_data['frame_paths'].append(file_path)
                    silhouette_data['subjects'].append(subject_id)
                    
                    # Aggiungi al dizionario soggetto -> percorsi
                    if subject_id not in silhouette_data['subject_to_paths']:
                        silhouette_data['subject_to_paths'][subject_id] = []
                    
                    silhouette_data['subject_to_paths'][subject_id].append(file_path)

                    
                    # Assume fixed_length should be calculated or given in initialization
                    if PrepareData.fixed_length == 0:
                        parts = file_path.split(os.sep)
                
                        # Verifica che il percorso abbia la struttura attesa
                        if len(parts) < 4:
                            print(f"Percorso del frame non valido: {file_path}")
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
                            print(f"ID soggetto non trovato nel percorso: {file_path}")
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
                            print(f"Nome sequenza non trovato nel percorso: {file_path}")
                            continue
                        
                        # Estrai il nome del file (es. 0001.png)
                        frame_filename = os.path.basename(file_path)
                        frame_name = os.path.splitext(frame_filename)[0]
                        
                        # Costruisci il percorso del file dei keypoints
                        # La struttura attesa è: OUMVLP-Pose/00/00001/000_00_180_090/0001.json
                        keypoints_path = os.path.join(keypoints_base_dir, subject_id, sequence_name, f"{frame_name}_keypoints.json")

                        if not os.path.exists(keypoints_path):
                            continue
                        else:
                            sequence_lenght += 1

                if PrepareData.fixed_length == 0:
                    lengths.append(sequence_lenght)

        # Verifica che abbiamo trovato dati
        if not silhouette_data['frame_paths']:
            raise FileNotFoundError(f"Nessuna silhouette trovata per i soggetti da 00001 a 00049")
        
        print(f"Trovate {len(silhouette_data['frame_paths'])} silhouette per {len(silhouette_data['subject_to_paths'])} soggetti")
        print("subjects:", len(silhouette_data['subjects']))

        if PrepareData.fixed_length == 0:
            pd_lengths = pd.Series(lengths)
                
            description = pd_lengths.describe(percentiles=[.10, .25, .50, .75, .90, .95, .97, .99])

            print("Statistical summary of the number of lines contained in the dataset csv files: \n")
            print(description, "\n")
            print("Standardized number of rows for all samples of the dataset (95% percentile):", int(description["95%"]), "\n\n\n")

            PrepareData.fixed_length = int(description["95%"])
        
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
            test_size=self._gait_keypoints_detection_config.data.test_size,
            random_state=42, 
            shuffle=True
        )

        # Dividi i soggetti rimanenti tra validation e test
        train_subjects, val_subjects = train_test_split(
            train_subjects, 
            test_size=self._gait_keypoints_detection_config.data.val_size,
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
        
        print(f"Split completato: {len(train_set_frame_paths)} frames di training, "
              f"{len(val_set_frame_paths)} frames di validation, "
              f"{len(test_set_frame_paths)} frames di test")
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
        return cropped, (x, y, x+w, y+h)
    
    def _letterbox(self, img, target_size=(256, 192), color=(127,127,127)):
        """Ridimensiona img preservando aspect ratio, poi pad fino a target_size."""
        ih, iw = target_size[1], target_size[0]  # nota: target=(W,H)
        h, w = img.shape[:2]
        scale = min(iw / w, ih / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((ih, iw, 3), color, dtype=np.uint8)
        pad_x = (iw - nw) // 2
        pad_y = (ih - nh) // 2
        canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
        return canvas, scale, pad_x, pad_y

    def _remap_keypoints_for_resize(self, keypoints, original_width, original_height, target_width, target_height):
        """
        Rimappa le coordinate dei keypoints quando un'immagine viene ridimensionata.
        
        Parameters
        ----------
        keypoints : list
            Lista di keypoints nel formato [x1, y1, c1, x2, y2, c2, ...] dove c è la confidenza
        original_width : int
            Larghezza originale dell'immagine
        original_height : int
            Altezza originale dell'immagine
        target_width : int
            Larghezza target dell'immagine ridimensionata
        target_height : int
            Altezza target dell'immagine ridimensionata
            
        Returns
        -------
        list
            Lista di keypoints rimappati alle nuove dimensioni
        """
        # Calcola i fattori di scala
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Crea una nuova lista per i keypoints rimappati
        remapped_keypoints = []
        
        # Processa ogni tripla (x, y, c)
        for i in range(0, len(keypoints), 3):
            x, y, conf = keypoints[i:i+3]
            
            # Applica la scala solo se il keypoint è valido (confidenza > 0)
            if conf > 0:
                x_scaled = x * scale_x
                y_scaled = y * scale_y
                remapped_keypoints.extend([x_scaled, y_scaled, conf])
            else:
                print(f"Keypoint non valido: {x}, {y}, {conf}")
                # Mantieni i keypoints non validi come sono
                remapped_keypoints.extend([x, y, conf])
                # remapped_keypoints.extend([0, 0, 0])
        
        return remapped_keypoints
    
    def _remap_keypoints(self, keypoints, crop_bbox, scale, pad_x, pad_y):
        """
        keypoints: flat list [x1,y1,c1, x2,y2,c2, ...]
        crop_bbox: (x0,y0,x1,y1) in original frame
        """
        x0, y0, x1, y1 = crop_bbox
        remapped = []
        for x, y, c in zip(keypoints[0::3], keypoints[1::3], keypoints[2::3]):
            if c > 0:
                # trasla nella ROI
                xr = x - x0
                yr = y - y0
                # scala e pad
                x_new = xr * scale + pad_x
                y_new = yr * scale + pad_y
                remapped.extend([x_new, y_new, c])
            else:
                remapped.extend([0, 0, 0])
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
        
        # Ottieni le dimensioni dell'immagine originale
        original_height, original_width = frame.shape[:2]

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
            "width": self._gait_keypoints_detection_config.data.image_width,
            "height": self._gait_keypoints_detection_config.data.image_height
        }
        
        # # Verifica la consistenza dei keypoints
        # is_consistent, message = self._verify_keypoints_consistency(keypoints_data)
        
        # if not is_consistent:
        #     print(f"Keypoints non consistenti nel frame {frame_path}: {message}")
        #     return None
        
        # Normalizza i keypoints per garantire una struttura consistente
        # normalized_data = self._normalize_keypoints(keypoints_data)
        
        # # Verifica se ci sono persone rilevate
        # if "people" not in keypoints_data or len(keypoints_data["people"]) == 0:
        #     print(f"Nessuna persona rilevata nel frame: {frame_path}")
        #     return None
        
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
            return None
        
        # # Rimappa i keypoints alle nuove dimensioni
        # remapped_keypoints = self._remap_keypoints_for_resize(
        #     pose_keypoints, 
        #     original_width, 
        #     original_height, 
        #     self._gait_keypoints_detection_config.data.image_width, 
        #     self._gait_keypoints_detection_config.data.image_height
        # )

        # 4) remap keypoints
        remapped_keypoints = self._remap_keypoints(
            pose_keypoints, crop_bbox, scale, pad_x, pad_y
        )

        # Debug: ensure exactly 17 keypoints after remapping
        if len(remapped_keypoints) != 17 * 3:
            print(f"[ERROR] Remapped keypoints length {len(remapped_keypoints)} "
                  f"!= 51 for frame {frame_filename}")
            return None

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
            return None
        
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        # Conta il numero di keypoints validi
        num_keypoints = sum(1 for i in range(0, len(remapped_keypoints), 3) if remapped_keypoints[i+2] > 0)
        
        # Crea l'informazione dell'annotazione
        annotation_info = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,  # Categoria "person"
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
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": [
                        # "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
                        # "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
                        # "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
                        # "left_eye", "right_ear", "left_ear"
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ],
                    "skeleton": [
                        # [1, 2], [2, 3], [3, 4], [4, 5], [2, 6], [6, 7], [7, 8], [2, 9], [9, 10],
                        # [10, 11], [2, 12], [12, 13], [13, 14], [1, 15], [15, 17], [1, 16], [16, 18]
                        # [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                        # [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                        # [0, 14], [14, 16], [0, 15], [15, 17]
                        [0, 1], [0, 2], [1, 3], [2, 4],
                        [5, 6], [6, 8], [5, 7], [7, 9],
                        [6, 10], [5, 11], [11, 13], [6, 12],
                        [12, 14], [11, 13], [13, 15], [12, 14], [14, 16]
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
        
        # Ridimensiona il frame alle dimensioni target
        # frame = cv2.resize(frame, (self._gait_keypoints_detection_config.data.image_size, 
        #                           self._gait_keypoints_detection_config.data.image_size))
        
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
        
        # Definisci connessioni personalizzate per un migliore schema visivo
        # Indici basati su OpenPose: [nose, neck, rshoulder, relbow, rwrist, lshoulder, lelbow, lwrist, 
        #                           rhip, rknee, rankle, lhip, lknee, lankle, reye, leye, rear, lear]
        # custom_connections = [
        #     # Testa
        #     [0, 1, 'head'],     # naso a collo
        #     [14, 0, 'head'],    # occhio destro a naso
        #     [15, 0, 'head'],    # occhio sinistro a naso
        #     [16, 14, 'head'],   # orecchio destro a occhio destro
        #     [17, 15, 'head'],   # orecchio sinistro a occhio sinistro
            
        #     # Torso
        #     [1, 2, 'torso'],    # collo a spalla destra
        #     [1, 5, 'torso'],    # collo a spalla sinistra
        #     [2, 8, 'torso'],    # spalla destra a anca destra
        #     [5, 11, 'torso'],   # spalla sinistra a anca sinistra
        #     # [8, 11, 'torso'],   # anca destra a anca sinistra
            
        #     # Braccio destro
        #     [2, 3, 'right_arm'],  # spalla destra a gomito destro
        #     [3, 4, 'right_arm'],  # gomito destro a polso destro
            
        #     # Braccio sinistro
        #     [5, 6, 'left_arm'],   # spalla sinistra a gomito sinistro
        #     [6, 7, 'left_arm'],   # gomito sinistro a polso sinistro
            
        #     # Gamba destra
        #     [1, 8, 'right_leg'],  # collo a anca destra
        #     [8, 9, 'right_leg'],  # anca destra a ginocchio destro
        #     [9, 10, 'right_leg'], # ginocchio destro a caviglia destra
            
        #     # Gamba sinistra
        #     [1, 11, 'left_leg'],  # collo a anca sinistra
        #     [11, 12, 'left_leg'], # anca sinistra a ginocchio sinistro
        #     [12, 13, 'left_leg']  # ginocchio sinistro a caviglia sinistra
        # ]

        # Updated skeleton connections for COCO-17 keypoints (0-based indices)
        # 0:nose,1:left_eye,2:right_eye,3:left_ear,4:right_ear,
        # 5:left_shoulder,6:right_shoulder,7:left_elbow,8:right_elbow,
        # 9:left_wrist,10:right_wrist,11:left_hip,12:right_hip,
        # 13:left_knee,14:right_knee,15:left_ankle,16:right_ankle
        custom_connections = [
            [0, 1, 'head'], [0, 2, 'head'], [1, 3, 'head'], [2, 4, 'head'],
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
    
    def _load_keypoint_sequences(self, subset_frame_paths, subset):
        """
        Loads keypoint sequences for each (subject, sequence) in the given frame paths.
        Returns sequences (N, fixed_length, 51) and labels (N,)
        """

        sequence_dict = {}
        
        # Percorso base della cartella dei keypoints
        keypoints_base_dir = os.path.join(self._gait_keypoints_detection_config.keypoints_sequences_dir, "00")
        
        if not os.path.exists(keypoints_base_dir):
            raise FileNotFoundError(f"Directory dei keypoints {keypoints_base_dir} non trovata.")
        
        # Itera su tutti i percorsi dei frame
        for frame_path in tqdm(subset_frame_paths, desc=f"Loading keypoints [{subset}]", unit="frame"):
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
            # frame = cv2.imread(frame_path)
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
                if keypoints and "annotations" in keypoints and len(keypoints["annotations"]) > 0:
                    kps = keypoints['annotations'][0]['keypoints']
                    key = (subject_id, sequence_name)
                    sequence_dict.setdefault(key, []).append((frame_name, kps))
                
                # # Se la conversione ha avuto successo, aggiungi il frame e i keypoints ai risultati
                # if keypoints:
                #     subset_data['frames'].append(frame)
                #     subset_data['keypoints'].append(keypoints)
                # else:
                #     # Se la conversione fallisce, salta questo frame
                #     print(f"Conversione fallita per il frame: {frame_path}")
                #     continue
                
            except Exception as e:
                print(f"Errore nel caricamento del file keypoints {keypoints_path}: {e}")
                continue

            # self._visualize_keypoints(keypoints, frame.copy())

            # # Salva l'immagine e i gait_keypoints in formato COCO nella directory corrispondente
            # subset_image_path = os.path.join(self._gait_keypoints_detection_config.save_data_splitted_path, "splitted_gait_keypoints_sequences_database", subset, "frames", f"{subject_id}_{sequence_name}_{frame_name}.png")
            # subset_gait_keypoints_path = os.path.join(self._gait_keypoints_detection_config.save_data_splitted_path, "splitted_gait_keypoints_sequences_database", subset, "keypoints", f"{subject_id}_{sequence_name}_{frame_name}.json")

            # # Salva i gait_keypoints in formato COCO nel file corrispondente
            # with open(subset_gait_keypoints_path, 'w') as f:
            #     json.dump(keypoints, f, indent=4)

        # After processing all frames, build fixed-length sequences
        keypoints_sequences, labels = [], []
        for (subject_id, sequence_name), items in sequence_dict.items():
            # items is list of (frame_name, kps)
            # sort by frame number
            sorted_items = sorted(items, key=lambda x: int(x[0]))
            kps_list = [k for _, k in sorted_items]
            if len(kps_list) >= self.fixed_length:
                window = kps_list[:self.fixed_length]
            else:
                # pad with zero-keypoints
                pad_kps = [[0.0] * len(kps_list[0]) for _ in range(self.fixed_length - len(kps_list))]
                window = kps_list + pad_kps
            keypoints_sequences.append(window)
            labels.append(int(subject_id))
        
        # return np.array(sequences), np.array(labels)

        subset_data = {
            'keypoints_sequences': np.array(keypoints_sequences),
            'labels': np.array(labels)
        }
        
        print(f" - {subset} subset: caricate {len(subset_data['keypoints_sequences'])} sequenze")

        return subset_data
    
    # Funzione per caricare le immagini
    def prepare_data(self):
        # Carica i percorsi delle silhouette
        silhouette_data = self._load_silhouette_paths()
        
        # Esegui lo split dei dati
        train_set_frame_paths, val_set_frame_paths, test_set_frame_paths = self._data_splitting(silhouette_data)

        # Percorsi di salvataggio
        for subset in ['train', 'val', 'test']:
            subset_path = os.path.join(self._gait_keypoints_detection_config.save_data_splitted_path, "splitted_gait_keypoints_database", subset)

            os.makedirs(subset_path, exist_ok=True)
    
            os.makedirs(os.path.join(subset_path, "frames"), exist_ok=True)
            os.makedirs(os.path.join(subset_path, "keypoints"), exist_ok=True)
        
        # Carica i keypoints per ciascun subset
        train_data = self._load_keypoint_sequences(train_set_frame_paths, "train")
        val_data = self._load_keypoint_sequences(val_set_frame_paths, "val")
        test_data = self._load_keypoint_sequences(test_set_frame_paths, "test")
        
        return train_data, val_data, test_data
    
    def reshape_data(self, subset_data):
        """
        Reshapes the sequences in the dataset into a format suitable for machine learning models.

        Returns
        -------
        dict
            The reshaped data dictionary.
        
        Example
        -------
        >>> train_data = PrepareData('/path/to/data', train=True)
        >>> train_set = train_data.load_data(is_ml_model=False)
        >>> train_set = train_data.reshape_data()
        >>> print(train_set['keypoints_sequences'].shape)
        """

        # subset_data['keypoints_sequences'] = subset_data['keypoints_sequences'].to_numpy()

        # Reshape the data
        subset_data['keypoints_sequences'] = subset_data['keypoints_sequences'].reshape(-1, PrepareData.fixed_length, subset_data['keypoints_sequences'].shape[1])

        return subset_data