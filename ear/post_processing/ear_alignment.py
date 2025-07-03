import cv2
import numpy as np

try:
    from ear.post_processing.ear_landmarks_detection import EarLandmarksDetection
except ModuleNotFoundError:
    try:
        from post_processing.ear_landmarks_detection import EarLandmarksDetection
    except ModuleNotFoundError:
        from ear_landmarks_detection import EarLandmarksDetection


class EarAlignment:
    def __init__(self, ear_config):
        """
        Inizializza la classe con la configurazione per l'allineamento dell'orecchio e 
        inizializza MediaPipe Face Mesh per l'estrazione dei landmark.
        
        :param ear_config: oggetto di configurazione contenente ad es.
            - ear_alignment.width_epsilon_ratio: rapporto per filtrare i landmark dell'orecchio (es. 0.05)
        """
        self._ear_config = ear_config
        self._ear_landmarks_detection = EarLandmarksDetection(ear_config)

    def _get_rotation_matrix(self, image, bounding_box):
        """
        Calcola la matrice di rotazione per allineare l'orecchio in verticale.
        
        Usa i landmark estratti dall'immagine originale (tramite MediaPipe) per selezionare 
        il punto più alto e quello più basso nell'area dell'orecchio (filtrati in base a x).
        L'angolo viene calcolato come:
            angle = arctan2(dy, dx) - 90°
        e il centro della rotazione viene definito come il punto medio tra i due landmark.
        
        :param image: immagine originale in BGR
        :return: matrice di rotazione M, angolo calcolato, centro (tuple in pixel)
        """
        norm_predicted_landmarks, orig_image, orig_predicted_landmarks, predicted_image, predicted_landmarks = self._ear_landmarks_detection.predict_ear_landmarks(image.copy(), bounding_box)
        x_top_pred, y_top_pred, x_bottom_pred, y_bottom_pred, x_outer_pred, y_outer_pred, x_inner_pred, y_inner_pred = orig_predicted_landmarks
        
        dx = x_bottom_pred - x_top_pred
        dy = y_bottom_pred - y_top_pred
        angle = np.degrees(np.arctan2(dy, dx)) - 90

        # Centro definito come il punto medio tra top e bottom
        center_point = ((x_top_pred + x_bottom_pred) / 2, (y_top_pred + y_bottom_pred) / 2)
        M = cv2.getRotationMatrix2D(center_point, angle, 1.0)
        
        return M, angle, (x_top_pred, y_top_pred), (x_bottom_pred, y_bottom_pred), (x_outer_pred, y_outer_pred), (x_inner_pred, y_inner_pred), center_point
    
    def _transform_bbox_cords(self, bounding_box, M, angle, top_point, bottom_point, outer_point, inner_point, center_point):
        # Calcola i nuovi angoli della bounding box ruotata
        x_min, y_min, x_max, y_max = bounding_box
        bbox_points = np.array([
            [x_min, y_min],  # Top-left
            [x_max, y_min],  # Top-right
            [x_min, y_max],  # Bottom-left
            [x_max, y_max]   # Bottom-right
        ], dtype=np.float32)

        # Converti i punti della bounding box in coordinate omogenee
        ones = np.ones((4, 1))
        bbox_points_homogeneous = np.hstack([bbox_points, ones])

        # Trasforma i vertici della bounding box
        rotated_bbox_points = np.dot(M, bbox_points_homogeneous.T).T

        # Trova i limiti del rettangolo ruotato
        x_min_new = max(0, int(np.min(rotated_bbox_points[:, 0])))
        y_min_new = max(0, int(np.min(rotated_bbox_points[:, 1])))
        x_max_new = min(self.img_w, int(np.max(rotated_bbox_points[:, 0])))
        y_max_new = min(self.img_h, int(np.max(rotated_bbox_points[:, 1])))

        # Controlla se l'angolo supera la soglia per applicare il fattore moltiplicativo
        if abs(angle) > self._ear_config.ear_alignment.angle_threshold:
            # print(f"Angolo {angle:.2f} supera la soglia {self._ear_config.ear_alignment.angle_threshold}.")
            box_width = int(self._ear_config.ear_alignment.factor * (x_max_new - x_min_new))  # Riduce la larghezza
            box_height = int(self._ear_config.ear_alignment.factor * (y_max_new - y_min_new))  # Riduce l'altezza
        else:
            # print(f"Angolo {angle:.2f} non supera la soglia {self._ear_config.ear_alignment.angle_threshold}.")
            box_width = x_max_new - x_min_new
            box_height = y_max_new - y_min_new

        # MODIFICA: Rendi la bounding box quadrata prendendo il lato più grande
        max_side = max(box_width, box_height)
        box_width = max_side
        box_height = max_side

        # Calcola i nuovi limiti della bounding box con il fattore applicato
        x_min_final = max(0, x_min_new + (x_max_new - x_min_new - box_width) // 2)
        y_min_final = max(0, y_min_new + (y_max_new - y_min_new - box_height) // 2)
        x_max_final = min(self.img_w, x_min_final + box_width)
        y_max_final = min(self.img_h, y_min_final + box_height)

        # Verifica se la bounding box è uscita dai limiti dell'immagine e aggiusta di conseguenza
        if x_max_final >= self.img_w:
            diff = x_max_final - self.img_w
            x_min_final = max(0, x_min_final - diff)
            x_max_final = self.img_w
        
        if y_max_final >= self.img_h:
            diff = y_max_final - self.img_h
            y_min_final = max(0, y_min_final - diff)
            y_max_final = self.img_h

        # Ricalcola le coordinate di top_point e bottom_point rispetto alla nuova bounding box
        top_point_x_new = top_point[0] - x_min_final
        top_point_y_new = top_point[1] - y_min_final
        top_point_transformed = (top_point_x_new, top_point_y_new)
        
        bottom_point_x_new = bottom_point[0] - x_min_final
        bottom_point_y_new = bottom_point[1] - y_min_final
        bottom_point_transformed = (bottom_point_x_new, bottom_point_y_new)
        
        outer_point_x_new = outer_point[0] - x_min_final
        outer_point_y_new = outer_point[1] - y_min_final
        outer_point_transformed = (outer_point_x_new, outer_point_y_new)
        
        inner_point_x_new = inner_point[0] - x_min_final
        inner_point_y_new = inner_point[1] - y_min_final
        inner_point_transformed = (inner_point_x_new, inner_point_y_new)
        
        center_point_x_new = center_point[0] - x_min_final
        center_point_y_new = center_point[1] - y_min_final
        center_point_transformed = (center_point_x_new, center_point_y_new)

        return x_min_final, y_min_final, x_max_final, y_max_final, top_point_transformed, bottom_point_transformed, outer_point_transformed, inner_point_transformed, center_point_transformed

    def align_ear(self, image, bounding_box):
        self.img_h, self.img_w = image.shape[:2]

        # Calcola la matrice di rotazione basata sui landmark (estratti dall'immagine originale)
        M, angle, top_point, bottom_point, outer_point, inner_point, center_point = self._get_rotation_matrix(image.copy(), bounding_box)
        rotated_image = cv2.warpAffine(image, M, (self.img_w, self.img_h), flags=cv2.INTER_CUBIC)
        
        if self._ear_config.show_images.alignment_ear_image:
            cv2.imshow("Rotated image", rotated_image)
            cv2.moveWindow("Rotated image", 0, 0)

        # Trasforma la bounding box originale secondo la matrice ottenuta
        x_min_final, y_min_final, x_max_final, y_max_final, top_point_transformed, bottom_point_transformed, outer_point_transformed, inner_point_transformed, center_point_transformed = self._transform_bbox_cords(bounding_box, M, angle, top_point, bottom_point, outer_point, inner_point, center_point)

        # MODIFICA: Verifica che la bounding box sia quadrata
        width = x_max_final - x_min_final
        height = y_max_final - y_min_final
        
        if width != height:
            # Prendi il lato più grande
            max_side = max(width, height)
            
            # Ricalcola i limiti per rendere la bounding box quadrata
            center_x = (x_min_final + x_max_final) // 2
            center_y = (y_min_final + y_max_final) // 2
            
            x_min_final = max(0, center_x - max_side // 2)
            y_min_final = max(0, center_y - max_side // 2)
            x_max_final = min(self.img_w, x_min_final + max_side)
            y_max_final = min(self.img_h, y_min_final + max_side)
            
            # Verifica se la bounding box è uscita dai limiti dell'immagine e aggiusta di conseguenza
            if x_max_final >= self.img_w:
                diff = x_max_final - self.img_w
                x_min_final = max(0, x_min_final - diff)
                x_max_final = self.img_w
            
            if y_max_final >= self.img_h:
                diff = y_max_final - self.img_h
                y_min_final = max(0, y_min_final - diff)
                y_max_final = self.img_h

        test_image = rotated_image.copy()
        cv2.rectangle(test_image, (x_min_final, y_min_final), (x_max_final, y_max_final), (0, 255, 0), 2)

        if self._ear_config.show_images.alignment_ear_image:
            cv2.imshow("Bounding box", test_image)
            cv2.moveWindow("Bounding box", 0, 200)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Ritaglia l'orecchio allineato basandosi sulla nuova bounding box
        ear_image_alignment = rotated_image[y_min_final:y_max_final, x_min_final:x_max_final]

        if self._ear_config.show_images.alignment_ear_image:
            cv2.imshow("Alignment ear image", ear_image_alignment)
            cv2.moveWindow("Alignment ear image", 0, 400)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return ear_image_alignment, top_point_transformed, bottom_point_transformed, outer_point_transformed, inner_point_transformed, center_point_transformed