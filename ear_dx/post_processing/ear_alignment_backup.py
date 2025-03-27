import cv2
import numpy as np

class EarAlignment:
    def __init__(self, ear_config):
        self.ear_config = ear_config

    def _get_rotation_matrix(self, ear_image, bounding_box):
        """
        Calcola la matrice di rotazione per allineare l'orecchio.
        Utilizza tecniche di analisi dell'immagine per determinare l'orientamento dell'orecchio.

        :param ear_image: immagine in scala di grigi dell'orecchio già ritagliato
        :param bounding_box: coordinate del rettangolo di delimitazione [x_min, y_min, x_max, y_max]
        :return: matrice di rotazione, angolo di rotazione e centro dell'orecchio
        """
        h, w = ear_image.shape
        x_min, y_min, x_max, y_max = bounding_box

        # Calcola il centro dell'orecchio (centro della bounding box)
        ear_center = (
            x_min + (x_max - x_min) // 2,
            y_min + (y_max - y_min) // 2
        )

        # Applica Canny edge detection per trovare i bordi
        edges = cv2.Canny(ear_image, 50, 150)

        # Trova le linee principali nell'orecchio usando la trasformata di Hough
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

        # Se non vengono trovate linee, restituisci una matrice di rotazione identità
        if lines is None or len(lines) == 0:
            return cv2.getRotationMatrix2D(ear_center, 0, 1.0), 0, ear_center

        # Calcola l'angolo medio delle linee trovate
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Converti l'angolo in gradi
            angle_deg = np.degrees(theta) - 90  # Sottrai 90 per ottenere l'angolo rispetto all'orizzontale
            
            # Normalizza l'angolo tra -90 e 90 gradi
            if angle_deg < -90:
                angle_deg += 180
            elif angle_deg > 90:
                angle_deg -= 180
                
            angles.append(angle_deg)

        # Filtra gli angoli outlier usando la mediana
        angles = np.array(angles)
        median_angle = np.median(angles)
        filtered_angles = angles[np.abs(angles - median_angle) < 30]  # Considera solo angoli entro 30 gradi dalla mediana
        
        if len(filtered_angles) > 0:
            angle = np.mean(filtered_angles)
        else:
            angle = median_angle

        # Determina il fattore di scala (default: 1.0, nessun ridimensionamento)
        scale = 1.0

        # Ottiene la matrice di rotazione per ruotare l'immagine
        M = cv2.getRotationMatrix2D(ear_center, angle, scale)

        return M, angle, ear_center
    
    def align_ear(image, point1, point2, output_size=(200, 300)):
        """
        Allinea l'immagine in base a due punti (tipicamente alto e basso dell'orecchio).
        
        Parameters:
            image: immagine originale
            point1: primo punto (x, y)
            point2: secondo punto (x, y)
            output_size: dimensioni finali immagine (width, height)
        Returns:
            Immagine allineata e ridimensionata
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = np.degrees(np.arctan2(dy, dx)) - 90  # -90 per allineare verticalmente

        center = tuple(np.mean([point1, point2], axis=0).astype(int))

        # Matrice di rotazione
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Crop centrato attorno al centro (opzionale)
        x, y = center
        w, h = output_size
        x1 = max(x - w//2, 0)
        y1 = max(y - h//2, 0)
        cropped = rotated[y1:y1+h, x1:x1+w]

        # Resize
        aligned = cv2.resize(cropped, output_size)
        return aligned
    
    def get_alignment_points_from_contour(self, image):
        """
        Estrae i punti di allineamento (alto e basso) dai contorni dell'orecchio.
        
        Parameters:
            image: immagine in scala di grigi
        Returns:
            point1, point2: punto alto e basso per allineamento
        """
        # Canny edge detection
        edges = cv2.Canny(image, 50, 150)

        # Trova contorni
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("Nessun contorno trovato.")

        # Prendi il contorno più grande (area massima)
        largest_contour = max(contours, key=cv2.contourArea)

        # Trova punto più in alto (min y) e più in basso (max y)
        top_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        bottom_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

        return top_point, bottom_point

    def align_ear(self, image, ear_image, bounding_box):
        """
        Allinea l'orecchio nell'immagine utilizzando i punti di contorno.
        
        :param image: immagine originale in scala di grigi
        :param ear_image: immagine dell'orecchio ritagliato in scala di grigi
        :param bounding_box: coordinate del rettangolo di delimitazione [x_min, y_min, x_max, y_max]
        :return: immagine dell'orecchio allineato e centro dell'orecchio trasformato
        """
        # Estrai i punti di allineamento dal contorno dell'orecchio
        try:
            point1, point2 = self.get_alignment_points_from_contour(ear_image)
            
            # Adatta i punti alle coordinate dell'immagine originale
            x_min, y_min, _, _ = bounding_box
            point1 = (point1[0] + x_min, point1[1] + y_min)
            point2 = (point2[0] + x_min, point2[1] + y_min)
            
            # Calcola la differenza tra i punti
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            angle = np.degrees(np.arctan2(dy, dx)) - 90  # -90 per allineare verticalmente

            # Calcola il centro tra i due punti
            center = tuple(np.mean([point1, point2], axis=0).astype(int))

            # Matrice di rotazione
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_h, img_w = image.shape
            rotated_image = cv2.warpAffine(image, M, (img_w, img_h), flags=cv2.INTER_CUBIC)

            if self.ear_config.show_images.alignment_ear_image:
                cv2.imshow("Rotated image", rotated_image)
                cv2.moveWindow("Rotated image", 0, 0)

            # Calcola le dimensioni dell'orecchio
            x_min, y_min, x_max, y_max = bounding_box
            ear_width = x_max - x_min
            ear_height = y_max - y_min
            
            # Ritaglia l'orecchio allineato centrato sul punto medio
            x, y = center
            x1 = max(x - ear_width//2, 0)
            y1 = max(y - ear_height//2, 0)
            x2 = min(x1 + ear_width, img_w)
            y2 = min(y1 + ear_height, img_h)
            
            ear_image_alignment = rotated_image[y1:y2, x1:x2]
            
            # Calcola il centro dell'orecchio rispetto alla nuova bounding box
            ear_center_x_new = center[0] - x1
            ear_center_y_new = center[1] - y1
            ear_center_transformed = (ear_center_x_new, ear_center_y_new)

            if self.ear_config.show_images.alignment_ear_image:
                cv2.imshow("Alignment ear image", ear_image_alignment)
                cv2.moveWindow("Alignment ear image", 0, 400)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return ear_image_alignment, ear_center_transformed
            
        except Exception as e:
            # In caso di errore, utilizza il metodo precedente basato sulla matrice di rotazione
            print(f"Errore nell'allineamento basato sui contorni: {e}. Utilizzo del metodo alternativo.")
            return self._align_ear_fallback(image, ear_image, bounding_box)
    
    def align_ear(self, image, ear_image, bounding_box):
        """
        Metodo di fallback per l'allineamento dell'orecchio utilizzando la matrice di rotazione.
        """
        # Ottieni la matrice di rotazione e il centro dell'orecchio
        M, angle, ear_center = self._get_rotation_matrix(ear_image, bounding_box)

        img_h, img_w = image.shape

        # Applica la rotazione all'intera immagine originale
        rotated_image = cv2.warpAffine(image, M, (img_w, img_h), flags=cv2.INTER_CUBIC)

        if self.ear_config.show_images.alignment_ear_image:
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
        if abs(angle) > self.ear_config.ear_alignment.angle_threshold:
            box_width = int(self.ear_config.ear_alignment.factor * (x_max_new - x_min_new))
            box_height = int(self.ear_config.ear_alignment.factor * (y_max_new - y_min_new))
        else:
            box_width = x_max_new - x_min_new
            box_height = y_max_new - y_min_new

        # Calcola i nuovi limiti della bounding box con il fattore applicato
        x_min_final = max(0, x_min_new + (x_max_new - x_min_new - box_width) // 2)
        y_min_final = max(0, y_min_new + (y_max_new - y_min_new - box_height) // 2)
        x_max_final = min(img_w, x_min_final + box_width)
        y_max_final = min(img_h, y_min_final + box_height)

        # Ricalcola le coordinate del centro dell'orecchio rispetto alla nuova bounding box
        ear_center_x_new = ear_center[0] - x_min_final
        ear_center_y_new = ear_center[1] - y_min_final
        ear_center_transformed = (ear_center_x_new, ear_center_y_new)

        # Ritaglia l'orecchio allineato basandosi sulla nuova bounding box
        ear_image_alignment = rotated_image[y_min_final:y_max_final, x_min_final:x_max_final]

        if self.ear_config.show_images.alignment_ear_image:
            cv2.imshow("Alignment ear image", ear_image_alignment)
            cv2.moveWindow("Alignment ear image", 0, 400)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return ear_image_alignment, ear_center_transformed