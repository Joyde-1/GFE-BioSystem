import cv2
import numpy as np
import os
from scipy.interpolate import interp1d


class GaitPreProcessing:
    def __init__(self, gait_config):
        self._gait_config = gait_config

    def _preprocess_silhouette(self, frame):
        """Preprocessa l'immagine della silhouette per migliorare il rilevamento"""
        # Partiamo da BGR o grayscale
        if len(frame.shape) < 3 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Rimuoviamo artefatti con blur + threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        # Manteniamo silhouette su sfondo neutro (127 invece di 0)
        # Questo aiuta il modello a rilevare meglio i contorni
        bg = np.full_like(frame, 127)
        fg = cv2.bitwise_and(frame, frame, mask=mask)
        return cv2.bitwise_or(fg, bg), mask
    
    def _crop_silhouette(self, frame, mask):
        """Ritaglia automaticamente la silhouette con un padding"""
        # Trova i contorni della silhouette
        #  frame, mask = self._preprocess_silhouette(frame.copy())

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return frame, (0, 0, frame.shape[1], frame.shape[0])  # Nessun contorno trovato
        
        # Prendi il contorno più grande (dovrebbe essere la silhouette)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Aggiungi padding
        x = max(0, x - self._gait_config.pre_processing.padding)
        y = max(0, y - self._gait_config.pre_processing.padding)
        w = min(frame.shape[1] - x, w + 2 * self._gait_config.pre_processing.padding)
        h = min(frame.shape[0] - y, h + 2 * self._gait_config.pre_processing.padding)
        
        # Ritaglia l'immagine
        cropped = frame[y:y + h, x:x + w]
        return cropped, (x, y, x + w, y + h)
    
    def _resize_silhouette(self, frame, color=(127, 127, 127)):
        """Ridimensiona frame preservando aspect ratio, poi pad fino a target_size."""
        ih, iw = self._gait_config.pre_processing.frame_height, self._gait_config.pre_processing.frame_width # target_size[1], target_size[0]  # nota: target=(W,H)
        h, w = frame.shape[:2]
        scale = min(iw / w, ih / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((ih, iw, 3), color, dtype=np.uint8)
        pad_x = (iw - nw) // 2
        pad_y = (ih - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y
    
    def pre_processing_frame(self, frame):
        # 1) preprocess + mask
        proc, mask = self._preprocess_silhouette(frame)
        # 2) crop
        cropped, crop_bbox = self._crop_silhouette(proc, mask)
        # cropped = self._crop_silhouette(proc, mask)
        # 3) resize
        resized, scale, pad_x, pad_y = self._resize_silhouette(cropped)
        # resized = self._resize_silhouette(cropped)

        pre_processing_params = {
            'crop_bbox': crop_bbox,
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y
        }

        # pre_processed_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # pre_processed_frame = Image.fromarray(pre_processed_frame)

        return resized, pre_processing_params
    
    def pre_processing_keypoints(self, keypoints, pre_processing_params):
        """
        Riporta i keypoints dallo spazio (0,S)x(0,S) della patch ritagliata+pad+resized
        allo spazio dell'immagine originale.

        Args:
            keypoints: lista piatta [x1, y1, c1,  x2, y2, c2,  …] con xi, yi in [0, S)
            crop_bbox: tuple (x0, y0, x1, y1) della patch *prima* del pad interno
                    (x0,y0) = angolo in alto a sinistra nella full frame
            scale:      fattore Lp/S  dove Lp = lato quadrato *dopo* il pad interno,
                    S = target_size usato in resize
            pad_x:      numero di pixel che sono stati aggiunti a sinistra (left pad)
            pad_y:      numero di pixel che sono stati aggiunti in alto  (top pad)

        Returns:
            remapped:  nuova lista [x1', y1', c1,  x2', y2', c2,  …] con xi', yi' 
                    calcolati nella full frame originale
        """
        x0, y0, _, _ = pre_processing_params['crop_bbox']
        remapped_keypoints = []
        # itero su triplette (x, y, conf)
        for x_proc, y_proc, c in zip(keypoints[0::3], keypoints[1::3], keypoints[2::3]):
            if c > 0:
                # 1) rimuovo il pad interno
                x_crop = x_proc - pre_processing_params['pad_x']
                y_crop = y_proc - pre_processing_params['pad_y']

                # 1) riporto da [0,S) a [0, Lp)
                x_square = x_crop / pre_processing_params['scale']
                y_square = y_crop / pre_processing_params['scale']

                # 3) riporto alla full frame originale
                x_orig = x_square + x0
                y_orig = y_square + y0

                remapped_keypoints.append([x_orig, y_orig, c])
            else:
                # conf == 0 → punto non rilevato
                remapped_keypoints.append([0, 0, 0])
        return remapped_keypoints
    
    def visualize_keypoints(self, keypoints, frame):
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

        # Nomi dei keypoints dalla categoria
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
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

        # Updated skeleton connections for COCO-17 keypoints (0-based indices)
        # 0:nose,1:left_eye,2:right_eye,3:left_ear,4:right_ear,
        # 5:left_shoulder,6:right_shoulder,7:left_elbow,8:right_elbow,
        # 9:left_wrist,10:right_wrist,11:left_hip,12:right_hip,
        # 13:left_knee,14:right_knee,15:left_ankle,16:right_ankle
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
        if self._gait_config.show_images.visualize_keypoints:
            cv2.imshow("Frame con Keypoints", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image

    def save_keypoints(self, keypoints, subject_id, sequence_name, frame_name):
        # 1) Converti in array NumPy
        # keypoints = np.array(keypoints, dtype=float)    # shape (17, 3)

        save_dir = os.path.join(self._gait_config.save_detected_keypoints_sequences_path, subject_id, sequence_name)

        # Crea la directory se non esiste
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Salva su file di testo
        with open(f'{save_dir}/{frame_name}', 'w') as f:
            for x, y, c in keypoints:
                f.write(f"{x},{y},{c}\n")

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
        L = self._gait_config.pre_processing.fixed_length

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
    
    def pre_processing_keypoints_sequence(self, keypoints_sequence):
        """Preprocessa i keypoints per il modello"""
        # Converti i keypoints in un array numpy
        # keypoints_array = np.array(keypoints, dtype=np.float32)
        
        # Normalizza i keypoints (se necessario)
        # Qui puoi aggiungere la logica di normalizzazione se il tuo modello lo richiede

        np_keypoints_sequence = np.array(keypoints_sequence)

        # pre_processed_keypoints = self._interpolate_missing_keypoints(np_keypoints_sequence)
        pre_processed_keypoints = self._normalize_keypoints(np_keypoints_sequence)
        pre_processed_keypoints = self._pad_or_truncate_sequence(pre_processed_keypoints)
        pre_processed_keypoints = self._filter_core_joints(pre_processed_keypoints)

        return pre_processed_keypoints