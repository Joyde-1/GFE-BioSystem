import cv2
import numpy as np

try:
    from face.post_processing.face_alignment import FaceAlignment
except ModuleNotFoundError:
    try:
        from post_processing.face_alignment import FaceAlignment
    except ModuleNotFoundError:
        from face_alignment import FaceAlignment



class PostProcessing:

    def __init__(self, face_config):
        self.face_config = face_config
        self.face_alignment = FaceAlignment(face_config)

    def _face_alignment(self, image, face_image, bounding_box):
        return self.face_alignment.align_face(image.copy(), face_image.copy(), bounding_box)

    def _bgr_to_gray(self, image):
        # Convert to grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def _cropping_image(self, image, eyes_center):
        """
        Ritaglia un'immagine per renderla quadrata, concentrandosi principalmente sull'altezza.
        Inoltre, aggiusta anche la larghezza.

        :param image: immagine di input (numpy array)
        :return: immagine quadrata ritagliata
        """
        # Ottieni le dimensioni dell'immagine
        height, width = image.shape[:2]

        # Usa il centro degli occhi trasformato per il ritaglio
        eye_center_x, eye_center_y = int(eyes_center[0]), int(eyes_center[1])

        # Riduci la larghezza applicando il fattore moltiplicativo
        new_width = int(width * self.face_config.post_processing.width_factor)

        # Calcola i margini in base a eyes_center, bilanciati a sinistra e destra
        x_min = max(0, eye_center_x - new_width // 2)
        x_max = min(width, eye_center_x + new_width // 2)

        # Bilancia i margini per garantire simmetria
        if x_max - x_min < new_width:
            excess = new_width - (x_max - x_min)
            x_min = max(0, x_min - excess // 2)
            x_max = min(width, x_max + excess // 2)

        # Ritaglia l'immagine in larghezza
        reduced_width_image = image[:, x_min:x_max]

        # Aggiorna le dimensioni dopo la riduzione della larghezza
        height, width = reduced_width_image.shape[:2]

        # Controlla il rapporto altezza/larghezza
        if height / width > self.face_config.post_processing.max_height_ratio:
            # print(f"Rapporto altezza/larghezza troppo grande: {height / width:.2f}")

            # Calcola il margine minimo sopra gli occhi
            min_margin_pixels = int(height * self.face_config.post_processing.min_margin_above_eyes)

            # Calcola la nuova altezza massima
            max_height = int(width * self.face_config.post_processing.bottom_cut_ratio)

            # Verifica che ci sia abbastanza spazio sopra gli occhi
            y_min = max(0, eye_center_y - min_margin_pixels)
            y_max = min(height, y_min + max_height)

            # Ritaglia mantenendo un'immagine rettangolare
            rectangular_image = reduced_width_image[y_min:y_max, :]
            return rectangular_image
        
        # Calcola il margine minimo sopra gli occhi
        min_margin_pixels = int(height * self.face_config.post_processing.min_margin_above_eyes)

        # Verifica che ci sia abbastanza spazio sopra gli occhi
        y_min = max(0, eye_center_y - min_margin_pixels)
        y_max = min(height, y_min + width)  # Assicurati che il ritaglio sia quadrato

        # Se il margine superiore è troppo grande, correggi per evitare di uscire dall'immagine
        if y_max - y_min < width:
            excess = width - (y_max - y_min)
            if y_min - excess // 2 >= 0:
                y_min -= excess // 2
            if y_max + excess // 2 <= height:
                y_max += excess // 2

        # Ritaglia l'immagine per renderla quadrata
        square_image = reduced_width_image[y_min:y_max, :]

        return square_image
    
    def _apply_clahe(self, image):
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=self.face_config.post_processing.clahe.cliplimit, tileGridSize=self.face_config.post_processing.clahe.tilegridsize)
        return clahe.apply(image)
    
    def _resize_image(self, image):
        return cv2.resize(image, (self.face_config.post_processing.image_size, self.face_config.post_processing.image_size), interpolation=cv2.INTER_AREA)    # cv2.INTER_NEAREST

    # def post_processing_image(self, image, bounding_box):

    #     # Ottieni le dimensioni dell'immagine
    #     img_h, img_w, _ = image.shape

    #     x_min = int(bounding_box[0] * img_w)
    #     y_min = int(bounding_box[1] * img_h)
    #     x_max = int(bounding_box[2] * img_w)
    #     y_max = int(bounding_box[3] * img_h)

    #     bounding_box = [x_min, y_min, x_max, y_max]
        
    #     # Ritaglia l'area del volto dall'immagine originale
    #     face_image = image[y_min:y_max, x_min:x_max]

    #     if self.face_config.show_images.post_processed_face_image:
    #         cv2.imshow("Cropped face image", face_image)
    #         cv2.moveWindow("Cropped face image", 0, 0)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
        
    #     # Converti in scala di grigi
    #     image = self._bgr_to_gray(image.copy())
    #     face_image_gray = self._bgr_to_gray(face_image)

    #     if self.face_config.show_images.post_processed_face_image:
    #         cv2.imshow("Gray face image", face_image_gray)
    #         cv2.moveWindow("Gray face image", 400, 0)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    #     face_image_aligmented, eyes_center = self._face_alignment(image, face_image_gray, bounding_box)

    #     if self.face_config.show_images.post_processed_face_image:
    #         cv2.imshow("Alignmented face image", face_image_aligmented)
    #         cv2.moveWindow("Alignmented face image", 800, 0)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #     face_image_cropped = self._cropping_image(face_image_aligmented, eyes_center)

    #     if self.face_config.show_images.post_processed_face_image:
    #         cv2.imshow("Cropped to square face image", face_image_cropped)
    #         cv2.moveWindow("Cropped to square face image", 1200, 0)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #     # Ridimensiona l'immagine a una dimensione fissa
    #     face_image_resized = self._resize_image(face_image_cropped)

    #     if self.face_config.show_images.post_processed_face_image:
    #         cv2.imshow("Resized face image", face_image_resized)
    #         cv2.moveWindow("Resized face image", 0, 400)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #     # Utilizza CLAHE per un miglioramento locale del contrasto
    #     face_image_equalized = self._apply_clahe(face_image_resized)

    #     if self.face_config.show_images.post_processed_face_image:
    #         cv2.imshow("Equalized face image", face_image_equalized)
    #         cv2.moveWindow("Equalized face image", 400, 400)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
        
    #     return face_image_equalized

    def post_processing_image(self, image, bounding_box):

        # Ottieni le dimensioni dell'immagine
        img_h, img_w, _ = image.shape

        x_min = int(bounding_box[0] * img_w)
        y_min = int(bounding_box[1] * img_h)
        x_max = int(bounding_box[2] * img_w)
        y_max = int(bounding_box[3] * img_h)

        bounding_box = [x_min, y_min, x_max, y_max]
        
        # Ritaglia l'area del volto dall'immagine originale
        face_image = image[y_min:y_max, x_min:x_max]

        if self.face_config.show_images.post_processed_face_image:
            cv2.imshow("Cropped face image", face_image)
            cv2.moveWindow("Cropped face image", 0, 0)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        # Converti in scala di grigi
        image = self._bgr_to_gray(image.copy())
        face_image_gray = self._bgr_to_gray(face_image)

        if self.face_config.show_images.post_processed_face_image:
            cv2.imshow("Gray face image", face_image_gray)
            cv2.moveWindow("Gray face image", 400, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Ridimensiona l'immagine a una dimensione fissa
        face_image_resized = self._resize_image(face_image_gray)

        if self.face_config.show_images.post_processed_face_image:
            cv2.imshow("Resized face image", face_image_resized)
            cv2.moveWindow("Resized face image", 0, 400)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return face_image_resized