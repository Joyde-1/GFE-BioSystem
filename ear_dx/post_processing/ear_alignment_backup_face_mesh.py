# import cv2
# import numpy as np

# class EarAlignment:
#     def __init__(self, ear_config):
#         """
#         Inizializza la classe con la configurazione per l'allineamento dell'orecchio.
#         """
#         self._ear_config = ear_config

#     def _find_alignment_points(self, ear_image):
#         """
#         Trova i punti di allineamento (punto più in alto e più in basso) nel crop dell'orecchio.
        
#         Se l'immagine è a colori, viene convertita in scala di grigi.
        
#         :param ear_image: immagine (crop) dell'orecchio, in BGR o scala di grigi
#         :return: top_point, bottom_point (tuple di coordinate locali)
#         """
#         # Applica il Canny edge detector usando i parametri dalla configurazione
#         edges = cv2.Canny(
#             ear_image,
#             self._ear_config.ear_alignment.canny_threshold1,
#             self._ear_config.ear_alignment.canny_threshold2
#         )

#         if self._ear_config.show_images.alignment_ear_image:
#             cv2.imshow("Canny edge ear image", edges)
#             cv2.moveWindow("Canny edge ear image", 0, 0)
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()

#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             raise ValueError("Nessun contorno trovato nell'immagine dell'orecchio.")
#         # Seleziona il contorno con l'area massima
#         largest_contour = max(contours, key=cv2.contourArea)
#         # Trova il punto con y minima (alto) e y massima (basso)
#         top_idx = np.argmin(largest_contour[:, :, 1])
#         bottom_idx = np.argmax(largest_contour[:, :, 1])
#         top_point = tuple(largest_contour[top_idx][0])
#         bottom_point = tuple(largest_contour[bottom_idx][0])
#         return top_point, bottom_point

#     def _get_rotation_matrix(self, original_image, ear_image, bbox_abs):
#         """
#         Calcola la matrice di rotazione basata sui landmark estratti dal crop dell'orecchio.
        
#         L'angolo viene calcolato come l'angolo tra il punto più alto e quello più basso,
#         sottraendo 90° per ottenere l'orecchio in posizione verticale.
#         Il centro della rotazione è definito come il centro del crop, tradotto in coordinate
#         globali tramite l'offset della bounding box.
        
#         :param original_image: immagine originale
#         :param ear_image: crop contenente l'orecchio
#         :param bbox_abs: bounding box in coordinate assolute (dizionario con chiavi 'left', 'top', 'right', 'bottom')
#         :return: matrice di rotazione M, angolo calcolato, centro globale
#         """
#         top_point, bottom_point = self._find_alignment_points(ear_image)
#         dx = bottom_point[0] - top_point[0]
#         dy = bottom_point[1] - top_point[1]
#         angle = np.degrees(np.arctan2(dy, dx)) - 90
        
#         # Calcola il centro del crop (locale)
#         ear_center = ((top_point[0] + bottom_point[0]) / 2,
#                       (top_point[1] + bottom_point[1]) / 2)
#         # Ottieni l'offset dalla bbox (in coordinate originali)
#         x1, y1 = bbox_abs['left'], bbox_abs['top']
#         global_center = (ear_center[0] + x1, ear_center[1] + y1)
        
#         M = cv2.getRotationMatrix2D(global_center, angle, 1.0)
#         return M, angle, global_center

#     def _transform_bbox(self, bbox_abs, M, orig_w, orig_h):
#         """
#         Applica la trasformazione (rotazione) alla bounding box.
        
#         :param bbox_abs: bounding box originale (dizionario con chiavi 'left', 'top', 'right', 'bottom')
#         :param M: matrice di rotazione (2x3)
#         :param orig_w: larghezza dell'immagine originale
#         :param orig_h: altezza dell'immagine originale
#         :return: nuova bounding box come tuple (x_min, y_min, x_max, y_max)
#         """
#         x_min, y_min, x_max, y_max = bbox_abs['left'], bbox_abs['top'], bbox_abs['right'], bbox_abs['bottom']
#         corners = np.array([
#             [x_min, y_min],
#             [x_max, y_min],
#             [x_max, y_max],
#             [x_min, y_max]
#         ], dtype=np.float32).reshape(-1, 1, 2)
        
#         transformed_corners = cv2.transform(corners, M).reshape(-1, 2)
#         x_min_new = max(0, int(np.min(transformed_corners[:, 0])))
#         y_min_new = max(0, int(np.min(transformed_corners[:, 1])))
#         x_max_new = min(orig_w, int(np.max(transformed_corners[:, 0])))
#         y_max_new = min(orig_h, int(np.max(transformed_corners[:, 1])))
#         return x_min_new, y_min_new, x_max_new, y_max_new

#     def align_ear(self, original_image, ear_image, bbox):
#         """
#         Allinea l'immagine originale sfruttando i landmark estratti dal crop dell'orecchio.
        
#         La bounding box (bbox) può essere fornita in formato:
#           - Normalizzato: [w_norm, h_norm, x_norm, y_norm]
#           - Assoluto: [x_min, y_min, x_max, y_max] (valori maggiori di 1)
        
#         La funzione:
#           1. Converte la bbox in coordinate assolute se necessario.
#           2. Calcola la matrice di rotazione basata sui landmark dell'orecchio.
#           3. Ruota l'immagine originale.
#           4. Trasforma la bbox e, se l'angolo supera una certa soglia, ridimensiona la bbox applicando un fattore.
#           5. Ritaglia l'immagine ruotata in base alla nuova bbox.
        
#         :param original_image: immagine originale (numpy array)
#         :param ear_image: crop contenente l'orecchio (numpy array)
#         :param bbox: lista di 4 valori che rappresenta la bounding box
#         :return: (new_bbox, cropped_image) dove new_bbox è la bbox finale (x_min, y_min, x_max, y_max)
#         """
#         orig_h, orig_w = original_image.shape[:2]
        
#         # Determina se la bbox è normalizzata oppure assoluta
#         if max(bbox) <= 1:
#             # bbox in formato normalizzato [w_norm, h_norm, x_norm, y_norm]
#             w_norm, h_norm, x_norm, y_norm = bbox
#             bbox_width = w_norm * orig_w
#             bbox_height = h_norm * orig_h
#             center_x = x_norm * orig_w
#             center_y = y_norm * orig_h
#             x1 = int(center_x - bbox_width / 2)
#             y1 = int(center_y - bbox_height / 2)
#             bbox_abs = {
#                 'left': x1,
#                 'top': y1,
#                 'right': int(x1 + bbox_width),
#                 'bottom': int(y1 + bbox_height)
#             }
#         else:
#             # bbox in formato assoluto [x_min, y_min, x_max, y_max]
#             x1, y1, x2, y2 = bbox
#             bbox_abs = {
#                 'left': int(x1),
#                 'top': int(y1),
#                 'right': int(x2),
#                 'bottom': int(y2)
#             }
        
#         # Calcola la matrice di rotazione usando i landmark estratti dal crop
#         M, angle, global_center = self._get_rotation_matrix(original_image, ear_image, bbox_abs)
#         rotated_image = cv2.warpAffine(original_image, M, (orig_w, orig_h), flags=cv2.INTER_CUBIC)
        
#         # Trasforma la bbox originale secondo la matrice ottenuta
#         new_bbox = self._transform_bbox(bbox_abs, M, orig_w, orig_h)
#         x_min_new, y_min_new, x_max_new, y_max_new = new_bbox
        
#         # Se l'angolo supera una soglia, applica un fattore di riduzione simile a quanto fai per il volto
#         if abs(angle) > self._ear_config.ear_alignment.angle_threshold:
#             box_width = int(self._ear_config.ear_alignment.factor * (x_max_new - x_min_new))
#             box_height = int(self._ear_config.ear_alignment.factor * (y_max_new - y_min_new))
#         else:
#             box_width = x_max_new - x_min_new
#             box_height = y_max_new - y_min_new
        
#         # Centra la nuova bounding box
#         x_min_final = max(0, x_min_new + (x_max_new - x_min_new - box_width) // 2)
#         y_min_final = max(0, y_min_new + (y_max_new - y_min_new - box_height) // 2)
#         x_max_final = min(orig_w, x_min_final + box_width)
#         y_max_final = min(orig_h, y_min_final + box_height)
        
#         cropped_image = rotated_image[y_min_final:y_max_final, x_min_final:x_max_final]
        
#         return cropped_image, (x_min_final, y_min_final, x_max_final, y_max_final)



import cv2
import numpy as np
import mediapipe as mp

class EarAlignment:
    def __init__(self, ear_config):
        """
        Inizializza la classe con la configurazione per l'allineamento dell'orecchio e 
        inizializza MediaPipe Face Mesh per l'estrazione dei landmark.
        
        :param ear_config: oggetto di configurazione contenente ad es.
            - ear_alignment.width_epsilon_ratio: rapporto per filtrare i landmark dell'orecchio (es. 0.05)
        """
        self._ear_config = ear_config
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

    def _get_face_landmarks(self, image):
        """
        Esegue il rilevamento dei landmark facciali usando MediaPipe Face Mesh.
        
        :param image: immagine originale in BGR
        :return: lista di coordinate (x, y) in pixel
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            # Fallback: try flipping the image horizontally
            flipped_image = cv2.flip(image, 1)
            rgb_flipped = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_flipped)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = image.shape
                # Adjust x coordinate: x_flipped = w - x
                points = [(w - int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                return points
            else:
                raise ValueError("Nessun landmark facciale rilevato da MediaPipe, anche dopo flip.")
        else:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = image.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            return points
        #     raise ValueError("Nessun landmark facciale rilevato da MediaPipe.")
        # landmarks = results.multi_face_landmarks[0].landmark
        # h, w, _ = image.shape
        # points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        # return points

    def _find_alignment_points(self, image):
        """
        Utilizza i landmark rilevati da MediaPipe per estrarre i punti di allineamento
        dell'orecchio. In particolare, ipotizziamo che, in un'immagine di profilo, l'orecchio
        sia localizzato nella parte destra del volto.
        
        Viene filtrata la lista dei landmark mantenendo quelli con x prossima al valore massimo.
        Da questi si seleziona il punto con y minima (top) e con y massima (bottom).
        
        :param image: immagine originale in BGR
        :return: top_point, bottom_point (tuple di coordinate in pixel)
        """
        points = self._get_face_landmarks(image)
        # Estrai il massimo valore di x (sull'asse orizzontale)
        xs = [p[0] for p in points]
        max_x = max(xs)
        # Usa un epsilon relativo alla larghezza dell'immagine per filtrare i landmark sull'orecchio.
        w = image.shape[1]
        epsilon = int(self._ear_config.ear_alignment.width_epsilon_ratio * w)  # es. 0.05
        ear_points = [p for p in points if p[0] > (max_x - epsilon)]
        if not ear_points:
            raise ValueError("Non sono stati trovati landmark per l'orecchio.")
        top_point = min(ear_points, key=lambda p: p[1])
        bottom_point = max(ear_points, key=lambda p: p[1])
        return top_point, bottom_point

    def _get_rotation_matrix(self, image):
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
        top_point, bottom_point = self._find_alignment_points(image)
        dx = bottom_point[0] - top_point[0]
        dy = bottom_point[1] - top_point[1]
        angle = np.degrees(np.arctan2(dy, dx)) - 90
        # Centro definito come il punto medio tra top e bottom
        center_point = ((top_point[0] + bottom_point[0]) / 2, (top_point[1] + bottom_point[1]) / 2)
        M = cv2.getRotationMatrix2D(center_point, angle, 1.0)
        return M, angle, top_point, bottom_point, center_point

    # def _transform_bbox(self, bbox_abs, M, orig_w, orig_h):
    #     """
    #     Applica la trasformazione (rotazione) alla bounding box.
        
    #     :param bbox_abs: bounding box originale (dizionario con chiavi 'left', 'top', 'right', 'bottom')
    #     :param M: matrice di rotazione (2x3)
    #     :param orig_w: larghezza dell'immagine originale
    #     :param orig_h: altezza dell'immagine originale
    #     :return: nuova bounding box come tuple (x_min, y_min, x_max, y_max)
    #     """
    #     x_min, y_min, x_max, y_max = bbox_abs['left'], bbox_abs['top'], bbox_abs['right'], bbox_abs['bottom']
    #     corners = np.array([
    #         [x_min, y_min],
    #         [x_max, y_min],
    #         [x_max, y_max],
    #         [x_min, y_max]
    #     ], dtype=np.float32).reshape(-1, 1, 2)
        
    #     transformed_corners = cv2.transform(corners, M).reshape(-1, 2)
    #     x_min_new = max(0, int(np.min(transformed_corners[:, 0])))
    #     y_min_new = max(0, int(np.min(transformed_corners[:, 1])))
    #     x_max_new = min(orig_w, int(np.max(transformed_corners[:, 0])))
    #     y_max_new = min(orig_h, int(np.max(transformed_corners[:, 1])))
    #     return x_min_new, y_min_new, x_max_new, y_max_new
    
    def _transform_bbox_cords(self, bounding_box, M, angle, top_point, bottom_point, center_point):
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
            # print(f"Angolo {angle:.2f} supera la soglia {self._ear_config.face_alignment.angle_threshold}.")
            box_width = int(self._ear_config.face_alignment.factor * (x_max_new - x_min_new))  # Riduce la larghezza
            box_height = int(self._ear_config.face_alignment.factor * (y_max_new - y_min_new))  # Riduce l'altezza
        else:
            # print(f"Angolo {angle:.2f} non supera la soglia {self._ear_config.face_alignment.angle_threshold}.")
            box_width = x_max_new - x_min_new
            box_height = y_max_new - y_min_new

        # Calcola i nuovi limiti della bounding box con il fattore applicato
        x_min_final = max(0, x_min_new + (x_max_new - x_min_new - box_width) // 2)
        y_min_final = max(0, y_min_new + (y_max_new - y_min_new - box_height) // 2)
        x_max_final = min(self.img_w, x_min_final + box_width)
        y_max_final = min(self.img_h, y_min_final + box_height)

        # Ricalcola le coordinate di top_point e bottom_point rispetto alla nuova bounding box
        top_point_x_new = top_point[0] - x_min_final
        top_point_y_new = top_point[1] - y_min_final
        top_point_transformed = (top_point_x_new, top_point_y_new)
        bottom_point_x_new = bottom_point[0] - x_min_final
        bottom_point_y_new = bottom_point[1] - y_min_final
        bottom_point_transformed = (bottom_point_x_new, bottom_point_y_new)
        center_point_x_new = center_point[0] - x_min_final
        center_point_y_new = center_point[1] - y_min_final
        center_point_transformed = (center_point_x_new, center_point_y_new)

        return x_min_final, y_min_final, x_max_final, y_max_final, top_point_transformed, bottom_point_transformed, center_point_transformed

    def align_ear(self, image, bounding_box):
        self.img_h, self.img_w = image.shape[:2]

        # # Conversione della bounding box in coordinate assolute
        # if max(bbox) <= 1:
        #     # bbox in formato normalizzato: [w_norm, h_norm, x_norm, y_norm]
        #     w_norm, h_norm, x_norm, y_norm = bbox
        #     bbox_width = w_norm * orig_w
        #     bbox_height = h_norm * orig_h
        #     center_x = x_norm * orig_w
        #     center_y = y_norm * orig_h
        #     x1 = int(center_x - bbox_width / 2)
        #     y1 = int(center_y - bbox_height / 2)
        #     bbox_abs = {'left': x1, 'top': y1, 'right': int(x1 + bbox_width), 'bottom': int(y1 + bbox_height)}
        # else:
        #     x1, y1, x2, y2 = bbox
        #     bbox_abs = {'left': int(x1), 'top': int(y1), 'right': int(x2), 'bottom': int(y2)}

        # Calcola la matrice di rotazione basata sui landmark (estratti dall'immagine originale)
        M, angle, top_point, bottom_point, center_point = self._get_rotation_matrix(image.copy())
        rotated_image = cv2.warpAffine(image, M, (self.img_h, self.img_w), flags=cv2.INTER_CUBIC)
        
        if self._ear_config.show_images.alignment_ear_image:
            cv2.imshow("Rotated image", rotated_image)
            cv2.moveWindow("Rotated image", 0, 0)

        # Trasforma la bounding box originale secondo la matrice ottenuta
        # new_bbox = self._transform_bbox(bbox_abs, M, orig_w, orig_h)
        x_min_final, y_min_final, x_max_final, y_max_final, top_point_transformed, bottom_point_transformed, center_point_transformed = self._transform_bbox_cords(bounding_box, M, angle, top_point, bottom_point, center_point)

        # Ritaglia l'orecchio allineato basandosi sulla nuova bounding box
        ear_image_alignment = rotated_image[y_min_final:y_max_final, x_min_final:x_max_final]

        if self._ear_config.show_images.alignment_ear_image:
            cv2.imshow("Alignment ear image", ear_image_alignment)
            cv2.moveWindow("Alignment ear image", 0, 400)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return ear_image_alignment, top_point_transformed, bottom_point_transformed, center_point_transformed