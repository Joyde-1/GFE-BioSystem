# import numpy as np
# import random


# class GaitDataAugmentation:
#     def __init__(self, gait_keypoints_detection_config):
#         self._gait_keypoints_detection_config = gait_keypoints_detection_config

#     def _rotate_keypoints_sequence(self, keypoints_sequence, angle_deg):
#         """
#         Ruota in piano tutti i keypoint di `angle_deg` gradi attorno al baricentro dei keypoint.
#         - keypoints_sequence: np.array shape (T, 17, 3)
#         - angle_deg: angolo in gradi (positivo = rotazione anticlockwise)
#         Restituisce un nuovo array (T, 17, 3) con (x, y) ruotati.
#         """
#         # Convertiamo in radianti
#         theta = np.deg2rad(angle_deg)
#         cos_t, sin_t = np.cos(theta), np.sin(theta)

#         # Calcoliamo il baricentro (media dei keypoint visibili su tutti i frame)
#         # Per semplicità, consideriamo media spaziale su tutti i frame e joint, ignorando confidence
#         coords = keypoints_sequence[..., :2]  # (T, 17, 2)
#         centroid = coords.reshape(-1, 2).mean(axis=0)  # (2,)

#         # Matrice di rotazione 2×2
#         R = np.array([[cos_t, -sin_t],
#                     [sin_t,  cos_t]], dtype=np.float32)

#         # Applichiamo la rotazione: x' = R·(x − centroid) + centroid
#         keypoints_sequence_rotated = np.empty_like(keypoints_sequence, dtype=np.float32)
#         for t in range(keypoints_sequence.shape[0]):
#             xy = keypoints_sequence[t, :, :2]  # (17,2)
#             xy_centered = xy - centroid  # (17,2)
#             xy_rot = xy_centered.dot(R.T) + centroid  # (17,2)
#             keypoints_sequence_rotated[t, :, :2] = xy_rot
#             # Confidence rimane invariato
#             keypoints_sequence_rotated[t, :, 2] = keypoints_sequence[t, :, 2]
#         return keypoints_sequence_rotated
    
#     def _scale_keypoints_sequence(self, keypoints_sequence, scale_factor):
#         """
#         Scala le coordinate (x, y) di `scale_factor` attorno al baricentro.
#         - keypoints_sequence: np.array shape (T, 17, 3)
#         - scale_factor: float (es. 0.9, 1.1, ecc.)
#         Restituisce sequenza scalata, mantenendo confidence invariato.
#         """
#         coords = keypoints_sequence[..., :2]  # (T,17,2)
#         centroid = coords.reshape(-1, 2).mean(axis=0)  # (2,)
#         keypoints_sequence_scaled = np.empty_like(keypoints_sequence, dtype=np.float32)
#         for t in range(keypoints_sequence.shape[0]):
#             xy = keypoints_sequence[t, :, :2]  # (17,2)
#             xy_centered = xy - centroid
#             xy_scaled = xy_centered * scale_factor + centroid
#             keypoints_sequence_scaled[t, :, :2] = xy_scaled
#             keypoints_sequence_scaled[t, :, 2] = keypoints_sequence[t, :, 2]
#         return keypoints_sequence_scaled
    
#     def _translate_keypoints_sequence(self, keypoints_sequence, shift_x, shift_y):
#         """
#         Trasla tutti i keypoint di (shift_x, shift_y) in pixel.
#         - shift_x, shift_y: float o int (anche negativi)
#         - keypoints_sequence: np.array (T,17,3)
#         """
#         keypoints_sequence_translated = np.empty_like(keypoints_sequence, dtype=np.float32)
#         keypoints_sequence_translated[..., 0] = keypoints_sequence[..., 0] + shift_x  # x + shift_x
#         keypoints_sequence_translated[..., 1] = keypoints_sequence[..., 1] + shift_y  # y + shift_y
#         keypoints_sequence_translated[..., 2] = keypoints_sequence[..., 2]            # confidence invariato
#         return keypoints_sequence_translated

#     def _horizontal_flip_keypoints_sequence(self, keypoints_sequence, img_width):
#         """
#         Applica horizontal flip: 
#         - x_new = img_width - 1 - x_old  (se coordinate pixel)
#         - scambia keypoints left/right
#         - keypoints_sequence: np.array (T,17,3)
#         - img_width: int (es. 640 pixel) oppure se coordinate normalizzate [0,1], usa 1.0
#         """
#         leftright_pairs = {
#             1: 2,  2: 1,
#             3: 4,  4: 3,
#             5: 6,  6: 5,
#             7: 8,  8: 7,
#             9: 10, 10: 9,
#             11: 12, 12: 11,
#             13: 14, 14: 13,
#             15: 16, 16: 15
#         }
#         keypoints_sequence_flipped = np.empty_like(keypoints_sequence, dtype=np.float32)
#         # 1) Riflessone orizzontale su x
#         keypoints_sequence_flipped[..., 0] = img_width - 1 - keypoints_sequence[..., 0]
#         keypoints_sequence_flipped[..., 1] = keypoints_sequence[..., 1]  # y rimane uguale
#         keypoints_sequence_flipped[..., 2] = keypoints_sequence[..., 2]  # confidence invariato

#         # 2) Scambiamo left/right: creiamo copia temporanea e riallochiamo
#         temp = keypoints_sequence_flipped.copy()
#         T = keypoints_sequence.shape[0]
#         for t in range(T):
#             for i in range(17):
#                 if i in leftright_pairs:
#                     j = leftright_pairs[i]
#                     keypoints_sequence_flipped[t, i, :] = temp[t, j, :]
#                 else:
#                     # keypoints centrali (es. naso, bacino) restano
#                     keypoints_sequence_flipped[t, i, :] = temp[t, i, :]
#         return keypoints_sequence_flipped
    
#     def _add_gaussian_noise(self, keypoints_sequence, sigma=1.0):
#         """
#         Aggiunge rumore gaussiano N(0, sigma^2) a x e y di ciascun keypoint.
#         - keypoints_sequence: np.array (T,17,3)
#         - sigma: deviazione standard in pixel (o unità coordinate)
#         """
#         noise_xy = np.random.normal(loc=0.0, scale=sigma, size=keypoints_sequence[..., :2].shape)
#         keypoints_sequence_noisy = keypoints_sequence.copy().astype(np.float32)
#         keypoints_sequence_noisy[..., :2] += noise_xy
#         # confidence invariato (oppure potresti ridurlo leggermente)
#         return keypoints_sequence_noisy
    
#     def _dropout_joints(self, keypoints_sequence, drop_prob=0.1):
#         """
#         Con probabilità drop_prob imposta confidence=0 (e coordinate a zero) per 
#         simulare keypoint mancanti/occlusi.
#         - keypoints_sequence: np.array (T,17,3)
#         - drop_prob: float tra 0 e 1
#         """
#         keypoints_sequence_drop = keypoints_sequence.copy().astype(np.float32)
#         T = keypoints_sequence.shape[0]
#         for t in range(T):
#             for j in range(17):
#                 if np.random.rand() < drop_prob:
#                     keypoints_sequence_drop[t, j, 2] = 0.0      # confidence -> 0
#                     # keypoints_sequence_drop[t, j, 0:2] = 0.0    # opzionale: azzero le coordinate
#         return keypoints_sequence_drop

#     def gait_data_augmentation(self, keypoints_sequence):
#         """
#         Applica una serie di trasformazioni di data augmentation sui keypoints.
#         - keypoints_sequence: list of lists (T, 51)
#         Restituisce una lista di sequenze augmentate (list of lists).
#         """
#         augmented_keypoints_sequences = []

#         # Lista di operazioni disponibili con parametri casuali
#         augmentation_operations = [
#             lambda keypoints_sequence: self._rotate_keypoints_sequence(
#                 keypoints_sequence.copy(),
#                 np.random.uniform(
#                     self._gait_keypoints_detection_config.data.data_augmentation_params.angle_deg_min,
#                     self._gait_keypoints_detection_config.data.data_augmentation_params.angle_deg_max
#                 )
#             ),
#             lambda keypoints_sequence: self._scale_keypoints_sequence(
#                 keypoints_sequence.copy(),
#                 np.random.uniform(
#                     self._gait_keypoints_detection_config.data.data_augmentation_params.scale_factor_min,
#                     self._gait_keypoints_detection_config.data.data_augmentation_params.scale_factor_max
#                 )
#             ),
#             lambda keypoints_sequence: self._translate_keypoints_sequence(
#                 keypoints_sequence.copy(),
#                 np.random.randint(
#                     self._gait_keypoints_detection_config.data.data_augmentation_params.shift_x_min,
#                     self._gait_keypoints_detection_config.data.data_augmentation_params.shift_x_max + 1
#                 ),
#                 np.random.randint(
#                     self._gait_keypoints_detection_config.data.data_augmentation_params.shift_y_min,
#                     self._gait_keypoints_detection_config.data.data_augmentation_params.shift_y_max + 1
#                 )
#             ),
#             lambda keypoints_sequence: self._horizontal_flip_keypoints_sequence(
#                 keypoints_sequence.copy(),
#                 self._gait_keypoints_detection_config.data.image_width
#             ),
#             lambda keypoints_sequence: self._add_gaussian_noise(
#                 keypoints_sequence.copy(),
#                 self._gait_keypoints_detection_config.data.data_augmentation_params.sigma
#             ),
#             lambda keypoints_sequence: self._dropout_joints(
#                 keypoints_sequence.copy(),
#                 self._gait_keypoints_detection_config.data.data_augmentation_params.drop_prob
#             )
#         ]

#         # # Rotazioni casuali tra -10 e +10 gradi
#         # angle_deg = np.random.uniform(self._gait_keypoints_detection_config.data.data_augmentation_params.angle_deg_min, self._gait_keypoints_detection_config.data.data_augmentation_params.angle_deg_max)
#         # augmented_keypoints_sequences.append(self._rotate_keypoints_sequence(keypoints_sequence, angle_deg))

#         # # Scaling casuale tra 0.9 e 1.1
#         # scale_factor = np.random.uniform(self._gait_keypoints_detection_config.data.data_augmentation_params.scale_factor_min, self._gait_keypoints_detection_config.data.data_augmentation_params.scale_factor_max)
#         # augmented_keypoints_sequences.append(self._scale_keypoints_sequence(keypoints_sequence, scale_factor))

#         # # Traslazione casuale in pixel
#         # shift_x = np.random.randint(self._gait_keypoints_detection_config.data.data_augmentation_params.shift_x_min, self._gait_keypoints_detection_config.data.data_augmentation_params.shift_x_max + 1)
#         # shift_y = np.random.randint(self._gait_keypoints_detection_config.data.data_augmentation_params.shift_y_min, self._gait_keypoints_detection_config.data.data_augmentation_params.shift_y_max + 1)
#         # augmented_keypoints_sequences.append(self._translate_keyoints_sequence(keypoints_sequence, shift_x, shift_y))
        
#         # # Horizontal flip
#         # augmented_keypoints_sequences.append(self._horizontal_flip_keypoints_sequence(keypoints_sequence, self._gait_keypoints_detection_config.data.image_size))
        
#         # # Aggiunta di rumore gaussiano
#         # augmented_keypoints_sequences.append(self._add_gaussian_noise(keypoints_sequence, self._gait_keypoints_detection_config.data.data_augmentation_params.sigma))
        
#         # # Dropout casuale di keypoints
#         # augmented_keypoints_sequences.append(self._dropout_joints(keypoints_sequence, self._gait_keypoints_detection_config.data.data_augmentation_params.drop_prob))

#         # Seleziona 3 operazioni a caso senza ripetizione
#         selected_augmentation_operations = random.sample(augmentation_operations, self._gait_keypoints_detection_config.data.data_augmentation_params.num_augmentation_operations)
        
#         # Applica e raccoglie le sequenze augmentate
#         augmented_keypoints_sequences = [augmentation_operation(keypoints_sequence) for augmentation_operation in selected_augmentation_operations]
#         return augmented_keypoints_sequences







import random
import torch
import numpy as np
from scipy.interpolate import interp1d

class Jitter:
    """Aggiunge rumore gaussiano alle coordinate (x,y)."""
    def __init__(self, sigma=0.005):
        self.sigma = sigma
    def __call__(self, seq):
        # seq: Tensor (L, 17, 2)  nello spazio normalizzato
        return seq + torch.randn_like(seq) * self.sigma


class TimeWarp:
    """
    Allunga/accorcia la sequenza con interpolazione lineare,
    quindi la riporta a lunghezza L_target.
    """
    def __init__(self, min_scale=0.9, max_scale=1.1, L_target=32):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.L = L_target

    def __call__(self, seq):
        # seq è Tensor (L, 17, 2) già normalizzato e fisso?  --> ricampiona lo stesso
        L_old = seq.shape[0]
        factor = random.uniform(self.min_scale, self.max_scale)
        L_new = max(2, int(L_old * factor))

        # vettoriale via numpy, poi torna a tensor
        idx_old = np.linspace(0, 1, L_old, dtype=np.float32)
        idx_new = np.linspace(0, 1, L_new, dtype=np.float32)
        flat = seq.cpu().numpy().reshape(L_old, -1)           # (L_old, 34)
        flat_new = interp1d(idx_old, flat, axis=0)(idx_new)    # (L_new, 34)

        # ricampiona di nuovo a L_target (32) per avere lunghezza fissa
        idx_final = np.linspace(0, 1, self.L, dtype=np.float32)
        flat_final = interp1d(idx_new, flat_new, axis=0)(idx_final)
        return torch.from_numpy(flat_final.reshape(self.L, 17, 2))


class PhaseShift:
    """Shifta l'inizio della sequenza di ±shift_max frame (wrap-around)."""
    def __init__(self, shift_max=2):
        self.shift_max = shift_max
    def __call__(self, seq):
        if self.shift_max == 0: return seq
        k = random.randint(-self.shift_max, self.shift_max)
        if k == 0: return seq
        return torch.roll(seq, shifts=k, dims=0)


def default_augmentation(gait_embedding_extraction_config):
    """Componi la pipeline usata in training."""
    return torch.nn.Sequential(
        Jitter(gait_embedding_extraction_config.data.data_augmentation_params.sigma),
        TimeWarp(gait_embedding_extraction_config.data.data_augmentation_params.min_scale, gait_embedding_extraction_config.data.data_augmentation_params.max_scale, gait_embedding_extraction_config.data.fixed_length),
        PhaseShift(gait_embedding_extraction_config.data.data_augmentation_params.shift_max)
    )