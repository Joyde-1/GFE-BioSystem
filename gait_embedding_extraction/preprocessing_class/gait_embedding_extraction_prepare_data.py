import cv2
from tqdm import tqdm
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d


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

    def __init__(self, gait_keypoints_detection_config):
        """
        Initializes the PrepareData instance

        Parameters
        ----------
        data_paths : str
            The path to the base directory containing database
        """
        
        self._gait_keypoints_detection_config = gait_keypoints_detection_config
    
    def _visualize_keypoints(self, keypoints, frame):
        """
        Visualizza i keypoints sul frame.
        
        Parameters
        ----------
        keypoints_data : dict
            Dizionario contenente i keypoints
        frame : numpy.ndarray
            Immagine del frame su cui visualizzare i keypoints
        """
        keypoints = np.array(keypoints).reshape(-1).tolist()

        print("keypoints shape:", np.array(keypoints).shape)
        print("keypoints:", keypoints)

        dim_keypoints = 2  # Ogni keypoint ha 3 valori: x, y, visibilità (opzionale)

        # Nomi dei keypoints dalla categoria
        keypoint_names = [
            # "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # Crea una copia dell'immagine per disegnare sopra
        image = frame.copy()

        # Definisci colori diversi per diverse parti del corpo
        colors = {
            # 'head': (0, 255, 255),    # Giallo
            'torso': (0, 0, 255),     # Rosso
            'right_arm': (255, 0, 0),  # Blu
            'left_arm': (0, 255, 0),   # Verde
            'right_leg': (255, 0, 255), # Magenta
            'left_leg': (255, 255, 0)   # Ciano
        }

        # Updated skeleton connections for COCO-17 keypoints (0-based indices)
        # 0:nose,1:left_eye,2:right_eye,3:left_ear,4:right_ear,
        # 5:left_shoulder,6:right_shoulder,7:left_elbow,8:right_elbow,
        # 9:left_wrist,10:right_wrist,11:left_hip,12:right_hip,
        # 13:left_knee,14:right_knee,15:left_ankle,16:right_ankle
        custom_connections = [
            # [0, 1, 'head'], [0, 2, 'head'], [1, 2, 'head'], [1, 3, 'head'], [2, 4, 'head'], [3, 5, 'head'], [4, 6, 'head'],
            [0, 1, 'torso'], [0, 6, 'torso'], [1, 7, 'torso'], [6, 7, 'torso'],
            [0, 2, 'left_arm'], [2, 4, 'left_arm'],
            [1, 3, 'right_arm'], [3, 5, 'right_arm'],
            [6, 8, 'left_leg'], [8, 10, 'left_leg'],
            [7, 9, 'right_leg'], [9, 11, 'right_leg']
        ]
        # custom_connections = [
        #     # [0, 1, 'head'], [0, 2, 'head'], [1, 2, 'head'], [1, 3, 'head'], [2, 4, 'head'], [3, 5, 'head'], [4, 6, 'head'],
        #     [5, 6, 'torso'], [5, 11, 'torso'], [6, 12, 'torso'], [11, 12, 'torso'],
        #     [5, 7, 'left_arm'], [7, 9, 'left_arm'],
        #     [6, 8, 'right_arm'], [8, 10, 'right_arm'],
        #     [11, 13, 'left_leg'], [13, 15, 'left_leg'],
        #     [12, 14, 'right_leg'], [14, 16, 'right_leg']
        # ]
        
        # Disegna i keypoints
        num_keypoints = len(keypoints) // dim_keypoints
        for i in range(num_keypoints):
            x = int(keypoints[dim_keypoints * i])
            y = int(keypoints[dim_keypoints * i + 1])
            # v = keypoints[dim_keypoints * i + 2]  # visibilità
            
            # if v > 0:  # Se il keypoint è visibile
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
            # if (idx1 * dim_keypoints + 2 < len(keypoints) and idx2 * dim_keypoints + 2 < len(keypoints) and
            #     keypoints[idx1 * dim_keypoints + 2] > 0 and keypoints[idx2 * dim_keypoints + 2] > 0):
                
            pt1 = (int(keypoints[idx1 * dim_keypoints]), int(keypoints[idx1 * dim_keypoints + 1]))
            pt2 = (int(keypoints[idx2 * dim_keypoints]), int(keypoints[idx2 * dim_keypoints + 1]))
            
            # Disegna una linea tra i due keypoints con il colore della parte del corpo
            cv2.line(image, pt1, pt2, colors[part], 2)
        
        # Mostra l'immagine con i keypoints
        cv2.imshow("Frame con Keypoints", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _normalize_keypoints(self, keypoints):
        """
        Normalizza key-points 2-D rendendo ogni sequenza invariante a traslazione
        e scala.  Usa il centro-bacino come origine e la lunghezza del torace
        (mediana sulla sequenza) come fattore di scala.

        Parameters
        ----------
        keypoints : np.ndarray
            • shape (T, 17, 2)  o  (17, 2)            –  x, y
            • shape (T, 17, 3)  o  (17, 3)            –  x, y, conf  (la conf viene ignorata)

        Returns
        -------
        np.ndarray
            Stessa shape dell’input (senza conf) dopo normalizzazione.
        """

        # ------------------------------------------------------------------ #
        # 1)  Mantieni solo (x, y) anche se c’è la colonna confidence
        # ------------------------------------------------------------------ #
        if keypoints.shape[-1] == 3:
            keypoints = keypoints[..., :2]

        # ------------------------------------------------------------------ #
        # 2)  Gestisci sia singolo frame che sequenza
        # ------------------------------------------------------------------ #
        if keypoints.ndim == 2:                # (17, 2)
            keypoints = keypoints[np.newaxis]  # (1, 17, 2)
            squeeze = True
        else:
            squeeze = False                    # (T, 17, 2)

        # Copia float32 per sicurezza
        kpts = keypoints.astype(np.float32).copy()

        # Indici COCO (modifica se usi un ordine diverso)
        L_HIP, R_HIP, L_SHO, R_SHO = 11, 12, 5, 6

        # ------------------------------------------------------------------ #
        # 3)  Calcola ancoraggio (mid-hip) e scala frame-per-frame
        # ------------------------------------------------------------------ #
        mid_hip = (kpts[:, L_HIP] + kpts[:, R_HIP]) / 2        # (T, 2)
        mid_sho = (kpts[:, L_SHO] + kpts[:, R_SHO]) / 2        # (T, 2)
        torso_len = np.linalg.norm(mid_sho - mid_hip, axis=1)  # (T,)

        # ------------------------------------------------------------------ #
        # 4)  Fattore di scala della sequenza = mediana delle lunghezze torso
        # ------------------------------------------------------------------ #
        scale_seq = float(np.median(torso_len))
        if scale_seq < 1e-6:                                   # fallback di sicurezza
            # usa diagonale del bbox su tutta la sequenza
            scale_seq = np.linalg.norm(kpts.ptp(axis=(0, 1)))

        # ------------------------------------------------------------------ #
        # 5)  Applica traslazione e scala
        # ------------------------------------------------------------------ #
        kpts_norm = (kpts - mid_hip[:, None, :]) / scale_seq

        return kpts_norm.squeeze(0) if squeeze else kpts_norm

    def _pad_or_truncate_sequence(self, keypoints_sequence):
        """
        Ri-campiona la sequenza di key-points 2-D a una lunghezza fissa usando
        interpolazione lineare sul tempo.

        Parameters
        ----------
        keypoints_sequence : np.ndarray
            shape (T, 17, 2)  oppure (T, 17, 3).
            Se è presente la confidence (3ª colonna) viene scartata.

        Returns
        -------
        np.ndarray
            shape (L, 17, 2) con L = self._gait_keypoints_detection_config.data.fixed_length
        """

        # --------------------------------------------------------------- #
        # 1)  Rimuovi la colonna confidence se presente
        # --------------------------------------------------------------- #
        if keypoints_sequence.shape[-1] == 3:
            keypoints_sequence = keypoints_sequence[..., :2]

        T, J, _ = keypoints_sequence.shape
        L = self._gait_keypoints_detection_config.data.fixed_length

        # --------------------------------------------------------------- #
        # 2)  Se la lunghezza è già corretta, restituisci
        # --------------------------------------------------------------- #
        if T == L:
            return keypoints_sequence.astype(np.float32)

        # --------------------------------------------------------------- #
        # 3)  Interpolazione vettoriale (funziona sia per up che down sample)
        # --------------------------------------------------------------- #
        old_idx = np.linspace(0, 1, T, dtype=np.float32)
        new_idx = np.linspace(0, 1, L, dtype=np.float32)

        #  interp1d sull'intera matrice (T, J*2) → (L, J*2)
        seq_flat = keypoints_sequence.reshape(T, -1)           # (T, 34)
        f = interp1d(old_idx, seq_flat,
                    kind='linear',
                    axis=0,
                    copy=False,
                    assume_sorted=True)
        seq_resampled = f(new_idx).reshape(L, J, 2).astype(np.float32)

        return seq_resampled
    
    def _filter_core_joints(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """Remove unstable facial key‑points and keep 12 body joints.

        Parameters
        ----------
        seq : np.ndarray
            Shape ``(L, 17, 2)`` where *L* is sequence length and 17 are COCO joints.

        Returns
        -------
        np.ndarray
            Shape ``(L, 12, 2)`` retaining only the joints in ``_KEEP_IDX``.
        """
        KEEP_IDX = [
            5, 6,        # shoulders
            7, 8,        # elbows
            9, 10,       # wrists
            11, 12,      # hips
            13, 14,      # knees
            15, 16,      # ankles
        ]
        if keypoints_sequence.ndim != 3 or keypoints_sequence.shape[1] != 17 or keypoints_sequence.shape[2] != 2:
            raise ValueError("Input must have shape (L, 17, 2)")
        # fancy indexing along joint dimension
        return keypoints_sequence[:, KEEP_IDX, :]

    def _load_keypoint_sequences(self):
        """
        Loads keypoint sequences for each (subject, sequence) in the given frame paths.
        """
        sequence_dict = {}

        step = 5

        # Cerca tutte le cartelle Silhouette_* nella directory dei dati
        silhouette_dirs = glob.glob(os.path.join(self._gait_keypoints_detection_config.frames_dir, "Silhouette_*"))

        if not silhouette_dirs:
            raise FileNotFoundError(f"Nessuna directory Silhouette_* trovata in {self._gait_keypoints_detection_config.frames_dir}")
        
        # Percorso base della cartella dei keypoints
        keypoints_base_dir = self._gait_keypoints_detection_config.keypoints_sequences_dir
        
        if not os.path.exists(keypoints_base_dir):
            raise FileNotFoundError(f"Directory dei keypoints {keypoints_base_dir} non trovata.")
        
        # Per ogni cartella Silhouette_*
        for silhouette_dir in tqdm(silhouette_dirs, desc="Loading data", unit="frame"):
            # Estrai l'angolo dalla directory (es. "Silhouette_030-00" o "Silhouette_090-01")
            sequence_name = os.path.basename(silhouette_dir)
            try:
                # Base name example: "Silhouette_030-00" → split on '_' → "030-00" → split on '-' → "030"
                angle_part = sequence_name.split('_')[-1]           # e.g. "030-00"
                angle_str = angle_part.split('-')[0]       # e.g. "030"
                angle = int(angle_str)

                if angle not in self._gait_keypoints_detection_config.data.selected_angles:
                    continue
            except (IndexError, ValueError):
                # Se non si riesce a interpretare l'angolo, salta questa directory
                continue

            check = 0

            # Per ogni soggetto da 00001 a 000num_classes
            for subject_num in range(1, self._gait_keypoints_detection_config.data.num_classes + 1 + step):   #1, 50
                if subject_num == 1 or subject_num == 7 or subject_num == 16 or subject_num == 28:
                    check += 1
                    continue

                subject_id = f"{subject_num:05d}"  # Formatta come 00001, 00002, ecc.

                real_subject_num = subject_num - check
                real_subject_id = f"{real_subject_num:05d}"

                # Cerca tutte le immagini per questo soggetto nella cartella corrente
                subject_sequence_pattern = os.path.join(silhouette_dir, subject_id, "*.png")
                subject_sequence_frames_paths = glob.glob(subject_sequence_pattern)

                # Se abbiamo trovato file per questo soggetto
                for frame_path in subject_sequence_frames_paths:
                    parts = frame_path.split(os.sep)
                    
                    # Verifica che il percorso abbia la struttura attesa
                    if len(parts) < 4:
                        print(f"Percorso del frame non valido: {frame_path}")
                        continue
                    
                    # Estrai il soggetto (es. 00001)
                    subject_id = None
                    for part in parts:
                        if part.isdigit() and len(part) == 5 and 1 <= int(part) <= self._gait_keypoints_detection_config.data.num_classes + step:  #49
                            subject_id = part
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
                    
                    if not sequence_name:
                        print(f"Nome sequenza non trovato nel percorso: {frame_path}")
                        continue

                    # Estrai il nome del file (es. 0001.png)
                    frame_filename = os.path.basename(frame_path)
                    frame_name = os.path.splitext(frame_filename)[0]
                    
                    # Costruisci il percorso del file dei keypoints
                    keypoints_path = os.path.join(keypoints_base_dir, real_subject_id, sequence_name, f"{frame_name}_keypoints.txt")

                    # print("FRAME PATH:", frame_path)
                    # print("KEYPOINTS PATH:", keypoints_path)

                    # print("")
                    
                    # Verifica se esiste il file dei keypoints
                    if not os.path.exists(keypoints_path):
                        # Se non esiste, salta questo frame
                        continue
                    
                    # # Carica l'immagine del frame
                    # frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)

                    # if frame is None:
                    #     print(f"Impossibile caricare il frame: {frame_path}")
                    #     continue
                    
                    # Carica il file TXT dei keypoints
                    try:
                        keypoints = []
                    
                        # Carica il file .txt dei keypoints
                        with open(keypoints_path, 'r') as f:
                            for i, line in enumerate(f):
                                x, y, _ = map(float, line.strip().split(','))
                                keypoints.append([x, y])
                        
                        # Se la conversione ha avuto successo, aggiungi il frame e i keypoints ai risultati
                        if keypoints is not None:
                            key = (real_subject_id, sequence_name)
                            sequence_dict.setdefault(key, []).append((frame_name, keypoints))
                        else:
                            # Se la conversione fallisce, salta questo frame
                            print(f"Conversione fallita per il frame: {frame_path}")
                            continue
                        
                    except Exception as e:
                        print(f"Errore nel caricamento del file keypoints {keypoints_path}: {e}")
                        continue

                    # self._visualize_keypoints(keypoints, frame.copy())

        keypoints_sequences, labels = [], []
        for (subject_id, sequence_name), items in sequence_dict.items():
            if len(items) < 18:
                print(f" - Subject {subject_id}, Sequence {sequence_name}: meno di 18 frames, salto questa sequenza")
                continue
            # items is list of (frame_name, kps)
            # sort by frame number
            sorted_items = sorted(items, key=lambda x: int(x[0]))
            kps_list = [k for _, k in sorted_items]
            kps_list = np.array(kps_list)

            keypoints_sequences.append(kps_list)
            labels.append(int(subject_id) - 1)

            if int(subject_id) - 1 < 0:
                print(f" - Subject {subject_id} ha ID negativo")

        print(f"Caricate {len(keypoints_sequences)} sequenze")

        return keypoints_sequences, labels
    
    def _data_splitting(self, keypoints_sequences, labels):
        # subjects = np.unique(labels)                 # 49 ID
        # train_subj, val_subj = train_test_split(subjects, test_size=0.2, stratify=subjects)
        # train_idx = np.isin(labels, train_subj)
        # val_idx   = np.isin(labels, val_subj)

        # train_set = {k: v[train_idx] for k,v in full_set.items()}
        # val_set   = {k: v[val_idx]   for k,v in full_set.items()}

        # Dividi in train e validation
        train_keypoints_sequences, val_keypoints_sequences, train_labels, val_labels = train_test_split(
            keypoints_sequences, 
            labels,
            test_size=self._gait_keypoints_detection_config.data.val_size,
            random_state=42, 
            shuffle=True,
            stratify=labels
        )

        return train_keypoints_sequences, val_keypoints_sequences, train_labels, val_labels

    def _preprocess_keypoints_sequences(self, subset_keypoints_sequences, subset_labels, subset):
        keypoints_sequences = []
        labels = []

        for keypoints_sequence, label in tqdm(zip(subset_keypoints_sequences, subset_labels), desc=f"Preprocessing {subset} data", unit="sequence"):
            keypoints_sequence = self._normalize_keypoints(keypoints_sequence)
            keypoints_sequence = self._pad_or_truncate_sequence(keypoints_sequence)
            keypoints_sequence = self._filter_core_joints(keypoints_sequence)

            # for keypoints in keypoints_sequence:
            #     self._visualize_keypoints(keypoints, np.zeros((960, 1280, 3), dtype=np.float32))  # Visualizza i keypoints su un'immagine vuota

            keypoints_sequences.append(keypoints_sequence)
            labels.append(label)

        subset_data = {
            'keypoints_sequences': np.array(keypoints_sequences),
            'labels': np.array(labels)
        }

        print(f" - {subset} subset: caricate {len(subset_data['keypoints_sequences'])} sequenze")

        return subset_data

    def prepare_data(self):
        # Carica i keypoints per ciascun subset
        keypoints_sequences, labels = self._load_keypoint_sequences()

        train_keypoints_sequences, val_keypoints_sequences, train_labels, val_labels = self._data_splitting(keypoints_sequences, labels)

        train_data = self._preprocess_keypoints_sequences(train_keypoints_sequences, train_labels, "train")
        val_data = self._preprocess_keypoints_sequences(val_keypoints_sequences, val_labels, "val")
        
        return train_data, val_data