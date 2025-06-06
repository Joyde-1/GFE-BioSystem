import cv2
import numpy as np


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
        # return cropped, (x, y, x + w, y + h)
        return cropped
    
    def _resize_silhouette(self, frame, color=(127, 127, 127)):
    # def _resize_silhouette(self, frame, target_size=(256, 192), color=(127, 127, 127)):
        """Ridimensiona frame preservando aspect ratio, poi pad fino a target_size."""
        # target_size = (target_size[1], target_size[0])  # nota: target=(H,W) per OpenCV
        ih, iw = self._gait_config.pre_processing.frame_height, self._gait_config.pre_processing.frame_width # target_size[1], target_size[0]  # nota: target=(W,H)
        h, w = frame.shape[:2]
        scale = min(iw / w, ih / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((ih, iw, 3), color, dtype=np.uint8)
        pad_x = (iw - nw) // 2
        pad_y = (ih - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        # return canvas, scale, pad_x, pad_y
        return canvas
    
    def pre_processing_frame(self, frame):
        # 1) preprocess + mask
        proc, mask = self._preprocess_silhouette(frame)
        # 2) crop
        # cropped, crop_bbox = self._crop_silhouette(proc, mask)
        cropped = self._crop_silhouette(proc, mask)
        # 3) resize
        # resized, scale, pad_x, pad_y = self._resize_silhouette(cropped)
        resized = self._resize_silhouette(cropped)

        # pre_processed_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # pre_processed_frame = Image.fromarray(pre_processed_frame)

        # return pre_processed_frame, resized
        return resized
    
    def pre_processing_keypoints(self, keypoints_sequence):
        """Preprocessa i keypoints per il modello"""
        # Converti i keypoints in un array numpy
        # keypoints_array = np.array(keypoints, dtype=np.float32)
        
        # Normalizza i keypoints (se necessario)
        # Qui puoi aggiungere la logica di normalizzazione se il tuo modello lo richiede

        pre_processed_keypoints = []

        if len(keypoints_sequence) >= self._gait_config.pre_processing.fixed_length:
            window = keypoints_sequence[:self._gait_config.pre_processing.fixed_length]
        else:
            # pad with zero-keypoints
            padded_keypoints = [[0.0] * len(keypoints_sequence[0]) for _ in range(self._gait_config.pre_processing.fixed_length - len(keypoints_sequence))]
            window = keypoints_sequence + padded_keypoints
        pre_processed_keypoints.append(window)

        pre_processed_keypoints = np.array(pre_processed_keypoints)

        return pre_processed_keypoints