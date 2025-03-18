import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_otsu, gaussian, median, threshold_local
from skimage.morphology import opening, disk, closing


class LBP:   
    def __init__(self, face_config):
        self.face_config = face_config
        
    def extract_lbp_features(self, image):
        """
        Estrae il template biometrico e genera una copia visualizzabile dell'immagine LBP.
        
        Parametri:
            image_bgr: immagine del volto in formato BGR (già ritagliata).
            
        Ritorna:
            template: array numpy (dtype=float32) contenente il template biometrico (vettore concatenato degli istogrammi).
            lbp_visual: immagine LBP normalizzata in scala 0-255 (uint8), pronta per cv2.imshow().
        """
        
        # 1. Calcolo dell'immagine LBP
        lbp_image = local_binary_pattern(
            image, 
            self.face_config.features_extraction.lbp.n_points, 
            self.face_config.features_extraction.lbp.radius, 
            method="uniform"
        )
        
        # 3. Creazione di una copia visualizzabile: normalizza lbp_image in scala 0-255
        lbp_min = lbp_image.min()
        lbp_max = lbp_image.max()
        if lbp_max - lbp_min > 0:
            lbp_visual = ((lbp_image - lbp_min) / (lbp_max - lbp_min) * 255).astype(np.uint8)
        else:
            lbp_visual = np.zeros_like(lbp_image, dtype=np.uint8)
        
        # 4. Suddivisione in griglia per estrarre gli istogrammi locali
        h, w = lbp_image.shape
        block_h = h // self.face_config.features_extraction.lbp.grid_y
        block_w = w // self.face_config.features_extraction.lbp.grid_x
        
        features = []

        n_bins = int(self.face_config.features_extraction.lbp.n_points + 2)

        for i in range(self.face_config.features_extraction.lbp.grid_y):
            for j in range(self.face_config.features_extraction.lbp.grid_x):
                # Definizione dei confini del blocco
                start_y = i * block_h
                end_y = (i + 1) * block_h if i != self.face_config.features_extraction.lbp.grid_y - 1 else h
                start_x = j * block_w
                end_x = (j + 1) * block_w if j != self.face_config.features_extraction.lbp.grid_x - 1 else w
                
                block = lbp_image[start_y:end_y, start_x:end_x]
                
                # Il numero di bin per il metodo "uniform" è n_points + 2
                n_bins = int(self.face_config.features_extraction.lbp.n_points + 2)
                hist, _ = np.histogram(block.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                features.extend(hist.tolist())
        
        # 5. Il template biometrico è la concatenazione di tutti gli istogrammi normalizzati
        template = np.array(features, dtype=np.float32)

        # 6. L2-normalizza il template per una migliore consistenza nel matching
        norm = np.linalg.norm(template) + 1e-6
        template = template / norm

        # Visualizza l'immagine LBP normalizzata
        if self.face_config.show_images.features_extracted_face_image:
            cv2.imshow("LBP face image", lbp_visual)
            cv2.moveWindow("LBP face image", 0, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return template, lbp_visual