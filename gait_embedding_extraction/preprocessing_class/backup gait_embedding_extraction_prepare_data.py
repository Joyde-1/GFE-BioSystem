import cv2
from tqdm import tqdm
import os
import json
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

# from gait_embedding_extraction.preprocessing_class.gait_embedding_extraction_data_augmentation import GaitDataAugmentation


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
    
    # def _preprocess_keypoints(self, keypoints_data, frame_filename):
    #     # Prendi i keypoints della prima persona (assumiamo che ci sia solo una persona per frame)
    #     person = keypoints_data["people"][0]
        
    #     # Estrai i keypoints 2D della posa
    #     pose_keypoints = person.get("pose_keypoints_2d", [])
        
    #     # Convert 18 OpenPose keypoints to 17 COCO keypoints by dropping 'neck' and reordering
    #     op_to_coco = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    #     keypoints = []
    #     for idx in op_to_coco:
    #         x, y, v = pose_keypoints[3 * idx : 3 * idx + 3]
    #         keypoints.append([x, y, v])

    #     # # Verifica che ci siano abbastanza keypoints (17 keypoints * 3 valori = 51)
    #     # if len(keypoints) < 17:
    #     #     print(f"Numero insufficiente di keypoints nel frame: {frame_filename}")
    #     #     return None, None

    #     return keypoints
    
    def _visualize_keypoints(self, keypoints, frame):
        """
        Visualizza i keypoints OpenPose sul frame.
        
        Parameters
        ----------
        keypoints_data : dict
            Dizionario contenente i keypoints in formato COCO
        frame : numpy.ndarray
            Immagine del frame su cui visualizzare i keypoints
        """
        keypoints = np.array(keypoints).reshape(-1).tolist()

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
            [5, 6, 'torso'], [5, 11, 'torso'], [6, 12, 'torso'], [11, 12, 'torso'],
            [5, 7, 'left_arm'], [7, 9, 'left_arm'],
            [6, 8, 'right_arm'], [8, 10, 'right_arm'],
            [11, 13, 'left_leg'], [13, 15, 'left_leg'],
            [12, 14, 'right_leg'], [14, 16, 'right_leg']
        ]
        
        # Disegna i keypoints
        num_keypoints = len(keypoints) // dim_keypoints
        for i in range(num_keypoints):
            x = int(keypoints[dim_keypoints * i])
            y = int(keypoints[dim_keypoints * i + 1])
            # v = keypoints[num_keypoints * i + 2]  # visibilità
            
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
            # if (idx1 * num_keypoints + 2 < len(keypoints) and idx2 * num_keypoints + 2 < len(keypoints) and
            #     keypoints[idx1 * num_keypoints + 2] > 0 and keypoints[idx2 * num_keypoints + 2] > 0):
                
            pt1 = (int(keypoints[idx1 * num_keypoints]), int(keypoints[idx1 * num_keypoints + 1]))
            pt2 = (int(keypoints[idx2 * num_keypoints]), int(keypoints[idx2 * num_keypoints + 1]))
            
            # Disegna una linea tra i due keypoints con il colore della parte del corpo
            cv2.line(image, pt1, pt2, colors[part], 2)
        
        # Mostra l'immagine con i keypoints
        cv2.imshow("Frame con Keypoints", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def _normalize_keypoints(self, keypoints):
    #     """
    #     Normalizza i keypoints per invarianza alla posizione e scala
    #     Args:
    #         keypoints: array (seq_len, 17, 3) o (17, 3)
    #         method: 'center', 'hip_center', 'bbox'
    #     """
    #     if keypoints.ndim == 2:
    #         keypoints = keypoints[np.newaxis, ...]
    #         squeeze_output = True
    #     else:
    #         squeeze_output = False

    #     normalized = keypoints.copy()

    #     for i in range(len(keypoints)):
    #         frame_kpts = keypoints[i]
    #         valid_mask = frame_kpts[:, 2] > 0.1  # confidence > 0.1

    #         if not np.any(valid_mask):
    #             continue

    #         valid_points = frame_kpts[valid_mask, :2]

    #         if self._gait_keypoints_detection_config.data.normalization_method == 'center':
    #             # Centra rispetto al centroide dei punti validi
    #             center = np.mean(valid_points, axis=0)
    #             normalized[i, :, :2] -= center

    #         elif self._gait_keypoints_detection_config.data.normalization_method == 'hip_center':
    #             # # Centra rispetto al punto medio dei fianchi
    #             # left_hip, right_hip = 11, 12
    #             # if (frame_kpts[left_hip, 2] > 0.1 and frame_kpts[right_hip, 2] > 0.1):
    #             #     hip_center = (frame_kpts[left_hip, :2] + frame_kpts[right_hip, :2]) / 2
    #             #     normalized[i, :, :2] -= hip_center

    #             # Usa centro dei fianchi come riferimento
    #             left_hip, right_hip = 11, 12
    #             if (frame_kpts[left_hip, 2] > 0.1 and frame_kpts[right_hip, 2] > 0.1):
    #                 hip_center = (frame_kpts[left_hip, :2] + frame_kpts[right_hip, :2]) / 2
                    
    #                 # Preserva meglio le informazioni di movimento
    #                 normalized[i, :, :2] = frame_kpts[:, :2] - hip_center
                    
    #                 # Normalizza solo per una scala fissa basata sulla risoluzione dell'immagine
    #                 scale_factor = 100.0  # Scala fissa
    #                 normalized[i, :, :2] /= scale_factor
    #             else:
    #                 # Fallback: usa centroide
    #                 center = np.mean(valid_points, axis=0)
    #                 normalized[i, :, :2] = frame_kpts[:, :2] - center
    #                 normalized[i, :, :2] /= 100.0
    #             # else:
    #             #     # Fallback al centroide
    #             #     center = np.mean(valid_points, axis=0)
    #             #     normalized[i, :, :2] -= center

    #         elif self._gait_keypoints_detection_config.data.normalization_method == 'bbox':
    #             # Normalizza rispetto al bounding box
    #             min_coords = np.min(valid_points, axis=0)
    #             max_coords = np.max(valid_points, axis=0)
    #             bbox_size = max_coords - min_coords
    #             bbox_center = (min_coords + max_coords) / 2

    #             normalized[i, :, :2] -= bbox_center
    #             if np.max(bbox_size) > 0:
    #                 normalized[i, :, :2] /= np.max(bbox_size)

    #     return normalized[0] if squeeze_output else normalized

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

    # def _interpolate_missing_keypoints(self, keypoints):
    #     """
    #     Interpola keypoints mancanti in una sequenza temporale
    #     Args:
    #         keypoints: array (seq_len, 17, 3)
    #         method: 'linear', 'cubic', 'forward_fill'
    #     """
    #     seq_len, num_joints, _ = keypoints.shape
    #     interpolated = keypoints.copy()

    #     for joint_idx in range(num_joints):
    #         for coord_idx in range(2):  # x, y coordinates
    #             # Trova frame validi (confidence > 0.1)
    #             valid_mask = keypoints[:, joint_idx, 2] > 0.1
    #             valid_indices = np.where(valid_mask)[0]

    #             if len(valid_indices) < 2:
    #                 continue

    #             # Interpola tra punti validi
    #             valid_values = keypoints[valid_indices, joint_idx, coord_idx]

    #             if self._gait_keypoints_detection_config.data.interpolation_method == 'linear':
    #                 interpolated[:, joint_idx, coord_idx] = np.interp(
    #                     np.arange(seq_len), valid_indices, valid_values
    #                 )
    #                 # Applica smoothing per ridurre il rumore
    #                 if seq_len > 5:
    #                     from scipy.ndimage import gaussian_filter1d
    #                     interpolated[:, joint_idx, coord_idx] = gaussian_filter1d(
    #                         interpolated[:, joint_idx, coord_idx], sigma=0.5
    #                     )
    #             elif self._gait_keypoints_detection_config.data.interpolation_method == 'forward_fill':
    #                 last_valid = valid_values[0]
    #                 for i in range(seq_len):
    #                     if valid_mask[i]:
    #                         last_valid = keypoints[i, joint_idx, coord_idx]
    #                     interpolated[i, joint_idx, coord_idx] = last_valid

    #     return interpolated

    # def _pad_or_truncate_sequence(self, keypoints_sequence):
    #     """
    #     Adatta sequenze di lunghezza variabile (18-35 frame) a lunghezza fissa
    #     Args:
    #         sequence: array (seq_len, 17, 3)
    #         target_length: lunghezza target (default: 32, compromesso tra 18-35)
    #         method: 'pad', 'truncate', 'interpolate'
    #     Returns:
    #         processed_sequence: array (target_length, 17, 3)
    #     """
    #     keypoints_sequence_len = keypoints_sequence.shape[0]

    #     if keypoints_sequence_len == self._gait_keypoints_detection_config.data.fixed_length:
    #         return keypoints_sequence

    #     elif keypoints_sequence_len < self._gait_keypoints_detection_config.data.fixed_length:
    #         # Sequenza troppo corta: padding
    #         if self._gait_keypoints_detection_config.data.add_frame_method == 'pad_and_truncate':
    #             # Zero padding alla fine
    #             padding = np.zeros((self._gait_keypoints_detection_config.data.fixed_length - keypoints_sequence_len, 17, 3))
    #             return np.concatenate([keypoints_sequence, padding], axis=0)

    #         elif self._gait_keypoints_detection_config.data.add_frame_method == 'reply':
    #             # Per sequenze corte, replica l'ultimo frame
    #             reply_needed = self._gait_keypoints_detection_config.data.fixed_length - keypoints_sequence_len
    #             last_frame = keypoints_sequence[-1:].repeat(reply_needed, axis=0)
    #             return np.concatenate([keypoints_sequence, last_frame], axis=0)

    #         elif self._gait_keypoints_detection_config.data.add_frame_method == 'interpolate':
    #             # Interpolazione temporale per allungare
    #             from scipy.interpolate import interp1d
    #             old_indices = np.linspace(0, 1, keypoints_sequence_len)
    #             new_indices = np.linspace(0, 1, self._gait_keypoints_detection_config.data.fixed_length)

    #             interpolated = np.zeros((self._gait_keypoints_detection_config.data.fixed_length, 17, 3))
    #             for joint_idx in range(17):
    #                 for coord_idx in range(3):
    #                     if np.any(keypoints_sequence[:, joint_idx, coord_idx] != 0):
    #                         f = interp1d(old_indices, keypoints_sequence[:, joint_idx, coord_idx], 
    #                                    kind='linear', fill_value='extrapolate')
    #                         interpolated[:, joint_idx, coord_idx] = f(new_indices)
    #             return interpolated
    #     else:
    #         # Sequenza troppo lunga: truncate
    #         if self._gait_keypoints_detection_config.data.delete_frame_method == 'pad_and_truncate':
    #             # Prendi i primi target_length frame
    #             return keypoints_sequence[:self._gait_keypoints_detection_config.data.fixed_length]
    #         elif self._gait_keypoints_detection_config.data.delete_frame_method == 'downsample':
    #             # Downsample mantenendo informazione temporale
    #             indices = np.linspace(0, keypoints_sequence_len - 1, self._gait_keypoints_detection_config.data.fixed_length, dtype=int)
    #             return keypoints_sequence[indices]

    #     return keypoints_sequence

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
        Returns sequences (N, fixed_length, 51) and labels (N,)
        """

        # sequence_dict = {}

        # for subject_id, subject_dir in enumerate(tqdm(os.listdir(self._gait_keypoints_detection_config.keypoints_sequences_dir), desc="Loading keypoints sequences", unit="subject")):
        #     if subject_dir == '.DS_Store':
        #         continue    
        #     for sequence_dir in os.listdir(os.path.join(self._gait_keypoints_detection_config.keypoints_sequences_dir, subject_dir)):        
        #         if sequence_dir == '.DS_Store':
        #             continue           
        #         angle_str = sequence_dir.split('_')[0]       # e.g. "030"
        #         angle = int(angle_str)

        #         if angle not in self._gait_keypoints_detection_config.data.selected_angles:
        #             continue

        #         for keypoints_path in glob.glob(os.path.join(self._gait_keypoints_detection_config.keypoints_sequences_dir, subject_dir, sequence_dir, "*.txt")):
        #             keypoints = []
                    
        #             # Carica il file .txt dei keypoints
        #             with open(keypoints_path, 'r') as f:
        #                 for line in f:
        #                     x, y, c = map(float, line.strip().split(','))
        #                     keypoints.append([x, y, c])

        #             key = (subject_dir, sequence_dir)
        #             sequence_dict.setdefault(key, []).append((os.path.basename(keypoints_path).split('_')[0], keypoints))




        sequence_dict = {}

        # Cerca tutte le cartelle Silhouette_* nella directory dei dati
        silhouette_dirs = glob.glob(os.path.join(self._gait_keypoints_detection_config.frames_dir, "Silhouette_*"))

        if not silhouette_dirs:
            raise FileNotFoundError(f"Nessuna directory Silhouette_* trovata in {self._gait_keypoints_detection_config.frames_dir}")
        
        # Percorso base della cartella dei keypoints
        # keypoints_base_dir = os.path.join(self._gait_keypoints_detection_config.keypoints_sequences_dir, "00")
        keypoints_base_dir = self._gait_keypoints_detection_config.keypoints_sequences_dir
        
        if not os.path.exists(keypoints_base_dir):
            raise FileNotFoundError(f"Directory dei keypoints {keypoints_base_dir} non trovata.")
        
        # Per ogni cartella Silhouette_*
        for silhouette_dir in tqdm(silhouette_dirs, desc="Loading data", unit="frame"):
            # print(f"Elaborazione della directory: {silhouette_dir}")
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

            # print(f"Elaborazione della directory: {silhouette_dir}")
            
            # Per ogni soggetto da 00001 a 000num_classes
            for subject_num in range(1, self._gait_keypoints_detection_config.data.num_classes + 1):   #1, 50
                subject_id = f"{subject_num:05d}"  # Formatta come 00001, 00002, ecc.

                # Cerca tutte le immagini per questo soggetto nella cartella corrente
                subject_sequence_pattern = os.path.join(silhouette_dir, subject_id, "*.png")
                subject_sequence_frames_paths = glob.glob(subject_sequence_pattern)

                # print("Silhouette files found for subject:", subject_id)
                # print(subject_sequence_frames_paths)

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
                        if part.isdigit() and len(part) == 5 and 1 <= int(part) <= self._gait_keypoints_detection_config.data.num_classes:  #49
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
                    # keypoints_path = os.path.join(keypoints_base_dir, subject_id, sequence_name, f"{frame_name}_keypoints.json")
                    keypoints_path = os.path.join(keypoints_base_dir, subject_id, sequence_name, f"{frame_name}_keypoints.txt")
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
                        # with open(keypoints_path, 'r') as f:
                        #     keypoints = json.load(f)

                        # # Converti i keypoints da OpenPose a COCO
                        # keypoints = self._preprocess_keypoints(keypoints, f"{subject_id}_{sequence_name}_{frame_name}.png")

                        keypoints = []
                    
                        # Carica il file .txt dei keypoints
                        with open(keypoints_path, 'r') as f:
                            for i, line in enumerate(f):
                                # x, y, c = map(float, line.strip().split(','))
                                # keypoints.append([x, y, c])
                                x, y, _ = map(float, line.strip().split(','))
                                keypoints.append([x, y])
                        
                        # Se la conversione ha avuto successo, aggiungi il frame e i keypoints ai risultati
                        if keypoints is not None:
                            key = (subject_id, sequence_name)
                            sequence_dict.setdefault(key, []).append((frame_name, keypoints))
                        else:
                            # Se la conversione fallisce, salta questo frame
                            print(f"Conversione fallita per il frame: {frame_path}")
                            continue
                        
                    except Exception as e:
                        print(f"Errore nel caricamento del file keypoints {keypoints_path}: {e}")
                        continue

                    # print(f"Frame: {subject_id}_{sequence_name}_{frame_name}")
                    # self._visualize_keypoints(keypoints, frame.copy())

        keypoints_sequences, labels = [], []
        for (subject_id, sequence_name), items in sequence_dict.items():
            # print(f" - Subject {subject_id}, Sequence {sequence_name}: {len(items)} frames")
            if len(items) < 18:
                print(f" - Subject {subject_id}, Sequence {sequence_name}: meno di 18 frames, salto questa sequenza")
                continue
            # items is list of (frame_name, kps)
            # sort by frame number
            sorted_items = sorted(items, key=lambda x: int(x[0]))
            kps_list = [k for _, k in sorted_items]
            kps_list = np.array(kps_list)

            keypoints_sequences.append(kps_list)
            labels.append(int(subject_id))

        print(f"Caricate {len(keypoints_sequences)} sequenze")

        return keypoints_sequences, labels
    
    def _data_splitting(self, keypoints_sequences, labels):
        # Dividi in train, validation e test
        train_keypoints_sequences, val_keypoints_sequences, train_labels, val_labels = train_test_split(
        # train_keypoints_sequences, test_keypoints_sequences, train_labels, test_labels = train_test_split(
            keypoints_sequences, 
            labels,
            test_size=self._gait_keypoints_detection_config.data.val_size,
            # test_size=self._gait_keypoints_detection_config.data.test_size,
            random_state=42, 
            shuffle=True,
            stratify=labels
        )

        # train_keypoints_sequences, val_keypoints_sequences, train_labels, val_labels = train_test_split(
        #     train_keypoints_sequences, 
        #     train_labels,
        #     test_size=self._gait_keypoints_detection_config.data.val_size,
        #     random_state=42, 
        #     shuffle=True,
        #     stratify=train_labels
        # )

        return train_keypoints_sequences, val_keypoints_sequences, train_labels, val_labels
        # return train_keypoints_sequences, val_keypoints_sequences, test_keypoints_sequences, train_labels, val_labels, test_labels

    def _preprocess_keypoints_sequences(self, subset_keypoints_sequences, subset_labels, subset):
        keypoints_sequences = []
        labels = []

        # data_augmentation = False

        # if self._gait_keypoints_detection_config.data.data_augmentation:
        #     data_augmentator = GaitDataAugmentation(self._gait_keypoints_detection_config)

        for keypoints_sequence, label in tqdm(zip(subset_keypoints_sequences, subset_labels), desc=f"Preprocessing {subset} data", unit="sequence"):
            # if self._gait_keypoints_detection_config.data.data_augmentation:
            #     if subset == "train":
            #         data_augmentation = True
            #     elif subset == "val" and self._gait_keypoints_detection_config.data.data_augmentation_params.val_augmentation:
            #         data_augmentation = True
            #     elif subset == "test" and self._gait_keypoints_detection_config.data.data_augmentation_params.test_augmentation:
            #         data_augmentation = True
            #     else:
            #         data_augmentation = False

            # if data_augmentation:
            #     augmented_keypoints_sequences = data_augmentator.gait_data_augmentation(keypoints_sequence)
            #     augmented_keypoints_sequences.append(keypoints_sequence)

            #     for augmented_keypoints_sequence in augmented_keypoints_sequences:
            #         # augmented_keypoints_sequence = self._interpolate_missing_keypoints(augmented_keypoints_sequence)
            #         augmented_keypoints_sequence = self._normalize_keypoints(augmented_keypoints_sequence)
            #         augmented_keypoints_sequence = self._pad_or_truncate_sequence(augmented_keypoints_sequence)

            #         keypoints_sequences.append(augmented_keypoints_sequence)
            #         labels.append(label)
            # else:
                # keypoints_sequence = self._interpolate_missing_keypoints(keypoints_sequence)
            keypoints_sequence = self._normalize_keypoints(keypoints_sequence)
            keypoints_sequence = self._pad_or_truncate_sequence(keypoints_sequence)
            keypoints_sequence = self._filter_core_joints(keypoints_sequence)

            keypoints_sequences.append(keypoints_sequence)
            labels.append(label)

        subset_data = {
            'keypoints_sequences': np.array(keypoints_sequences),
            'labels': np.array(labels)
        }

        print(f" - {subset} subset: caricate {len(subset_data['keypoints_sequences'])} sequenze")

        return subset_data

    # Funzione per caricare le immagini
    def prepare_data(self):
        # Carica i keypoints per ciascun subset
        keypoints_sequences, labels = self._load_keypoint_sequences()

        train_keypoints_sequences, val_keypoints_sequences, train_labels, val_labels = self._data_splitting(keypoints_sequences, labels)
        # train_keypoints_sequences, val_keypoints_sequences, test_keypoints_sequences, train_labels, val_labels, test_labels = self._data_splitting(keypoints_sequences, labels)

        train_data = self._preprocess_keypoints_sequences(train_keypoints_sequences, train_labels, "train")
        val_data = self._preprocess_keypoints_sequences(val_keypoints_sequences, val_labels, "val")
        # test_data = self._preprocess_keypoints_sequences(test_keypoints_sequences, test_labels, "test")
        
        return train_data, val_data
        # return train_data, val_data, test_data