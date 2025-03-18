import cv2
import dlib
import numpy as np


class FaceAlignment:
    def __init__(self, face_config):
        self.face_config = face_config
        self._load_shape_predictor()

    def _load_shape_predictor(self):
        """
        Inizializza il predittore dei landmark facciali.

        :param predictor_path: percorso al file shape_predictor_68_face_landmarks.dat
        """
        self.predictor = dlib.shape_predictor(self.face_config.predictor_path + '/shape_predictor_68_face_landmarks.dat')

    @staticmethod
    def shape_to_np(shape, dtype="int"):
        """
        Converte l'oggetto dlib shape in un array NumPy di coordinate (x, y).

        :param shape: oggetto restituito dal predittore di landmark di dlib
        :param dtype: tipo di dato per l'array risultante
        :return: array NumPy di dimensione (68, 2)
        """
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _get_rotation_matrix(self, face_image, bounding_box):
        """
        Allinea il volto in un'immagine già croppata.
        Si assume che l'intera immagine sia il volto, quindi il rettangolo di riferimento
        copre l'intera area.

        :param face_image: immagine BGR del volto già croppato
        :param desired_left_eye: tupla (x, y) con le coordinate (in percentuale)
                                 desiderate del centro dell'occhio sinistro nell'immagine allineata
        :param desired_face_width: larghezza desiderata dell'immagine risultante
        :param desired_face_height: altezza desiderata; se None viene usato lo stesso valore di desired_face_width
        :return: immagine del volto allineato oppure None in caso di problemi
        """
        h, w = face_image.shape

        x_min, y_min, x_max, y_max = bounding_box

        # Dal momento che il volto è già croppato, definiamo un rettangolo che copre l'intera immagine
        rect = dlib.rectangle(0, 0, w, h)

        # Estrae i landmark dal volto
        shape = self.predictor(face_image, rect)
        shape_np = self.shape_to_np(shape)

        # Nei modelli a 68 punti di dlib:
        # - gli occhi sono indicizzati da 36 a 41 (occhio destro) e da 42 a 47 (occhio sinistro)
        right_eye_pts = shape_np[36:42]
        left_eye_pts  = shape_np[42:48]

        # Calcola il centro di ciascun occhio
        right_eye_center = right_eye_pts.mean(axis=0).astype("int")
        left_eye_center  = left_eye_pts.mean(axis=0).astype("int")

        # Determina l'angolo tra la linea che congiunge i centri degli occhi e l'orizzontale
        dY = left_eye_center[1] - right_eye_center[1]
        dX = left_eye_center[0] - right_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Calcola la distanza attuale tra gli occhi
        current_eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        # Calcola la distanza desiderata tra gli occhi in base alla posizione voluta
        desired_eye_distance = (1.0 - 2 * self.face_config.face_alignment.desired_left_eye[0]) * w
        # Determina il fattore di scala
        scale = desired_eye_distance / (current_eye_distance + 1e-6)

        # Calcola il centro degli occhi
        eyes_center = (
            x_min + int((left_eye_center[0] + right_eye_center[0]) // 2),
            y_min + int((left_eye_center[1] + right_eye_center[1]) // 2)
        )

        # Ottiene la matrice di rotazione per ruotare e scalare l'immagine
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        return M, angle, eyes_center
    
    def align_face(self, image, face_image, bounding_box):   
        # Ottieni la matrice di rotazione e il centro degli occhi
        M, angle, eyes_center = self._get_rotation_matrix(face_image, bounding_box)

        img_h, img_w = image.shape

        # Applica la rotazione all'intera immagine originale
        rotated_image = cv2.warpAffine(image, M, (img_w, img_h), flags=cv2.INTER_CUBIC)

        if self.face_config.show_images.alignment_face_image:
            cv2.imshow("Rotated image", rotated_image)
            cv2.moveWindow("Rotated image", 0, 0)

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
        x_max_new = min(img_w, int(np.max(rotated_bbox_points[:, 0])))
        y_max_new = min(img_h, int(np.max(rotated_bbox_points[:, 1])))

        # Controlla se l'angolo supera la soglia per applicare il fattore moltiplicativo
        if abs(angle) > self.face_config.face_alignment.angle_threshold:
            # print(f"Angolo {angle:.2f} supera la soglia {self.face_config.face_alignment.angle_threshold}.")
            box_width = int(self.face_config.face_alignment.factor * (x_max_new - x_min_new))  # Riduce la larghezza
            box_height = int(self.face_config.face_alignment.factor * (y_max_new - y_min_new))  # Riduce l'altezza
        else:
            # print(f"Angolo {angle:.2f} non supera la soglia {self.face_config.face_alignment.angle_threshold}.")
            box_width = x_max_new - x_min_new
            box_height = y_max_new - y_min_new

        # Calcola i nuovi limiti della bounding box con il fattore applicato
        x_min_final = max(0, x_min_new + (x_max_new - x_min_new - box_width) // 2)
        y_min_final = max(0, y_min_new + (y_max_new - y_min_new - box_height) // 2)
        x_max_final = min(img_w, x_min_final + box_width)
        y_max_final = min(img_h, y_min_final + box_height)

        # Ricalcola le coordinate di eyes_center rispetto alla nuova bounding box
        eyes_center_x_new = eyes_center[0] - x_min_final
        eyes_center_y_new = eyes_center[1] - y_min_final
        eyes_center_transformed = (eyes_center_x_new, eyes_center_y_new)

        # Ritaglia il volto allineato basandosi sulla nuova bounding box
        face_image_alignment = rotated_image[y_min_final:y_max_final, x_min_final:x_max_final]

        if self.face_config.show_images.alignment_face_image:
            cv2.imshow("Alignment face image", face_image_alignment)
            cv2.moveWindow("Alignment face image", 0, 400)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return face_image_alignment, eyes_center_transformed