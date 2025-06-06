import cv2
import numpy as np

try:
    from ear.post_processing.ear_alignment import EarAlignment
except ModuleNotFoundError:
    try:
        from post_processing.ear_alignment import EarAlignment
    except ModuleNotFoundError:
        from ear_alignment import EarAlignment


class EarPostProcessing:

    def __init__(self, ear_config):
        self.ear_config = ear_config
        self.ear_alignment = EarAlignment(ear_config)

    def _ear_alignment(self, image, bounding_box):
        return self.ear_alignment.align_ear(image.copy(), bounding_box)

    def _bgr_to_gray(self, image):
        # Convert to grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # def _get_alignment_points(image):
    #     """
    #     Estrae i punti di allineamento (alto e basso) dai contorni dell'orecchio.
        
    #     Parameters:
    #         image: immagine in scala di grigi
    #     Returns:
    #         point1, point2: punto alto e basso per allineamento
    #     """
    #     # Canny edge detection
    #     edges = cv2.Canny(image, 50, 150)

    #     # Trova contorni
    #     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     if not contours:
    #         raise ValueError("Nessun contorno trovato.")

    #     # Prendi il contorno più grande (area massima)
    #     largest_contour = max(contours, key=cv2.contourArea)

    #     # Trova punto più in alto (min y) e più in basso (max y)
    #     top_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    #     bottom_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

    #     return top_point, bottom_point
    
    def _cropping_image(self, image, ear_center):
        """
        Ritaglia un'immagine per renderla quadrata, concentrandosi principalmente sull'altezza.
        Inoltre, aggiusta anche la larghezza.

        :param image: immagine di input (numpy array)
        :return: immagine quadrata ritagliata
        """
        # Ottieni le dimensioni dell'immagine
        height, width = image.shape[:2]

        # Usa il centro dell'orecchio trasformato per il ritaglio
        eye_center_x, eye_center_y = int(ear_center[0]), int(ear_center[1])

        # Riduci la larghezza applicando il fattore moltiplicativo
        new_width = int(width * self.ear_config.post_processing.width_factor)

        # Calcola i margini in base a ear_center, bilanciati a sinistra e destra
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
        if height / width > self.ear_config.post_processing.max_height_ratio:
            # print(f"Rapporto altezza/larghezza troppo grande: {height / width:.2f}")

            # Calcola il margine minimo sopra l'orecchio
            min_margin_pixels = int(height * self.ear_config.post_processing.min_margin_above_ear)

            # Calcola la nuova altezza massima
            max_height = int(width * self.ear_config.post_processing.bottom_cut_ratio)

            # Verifica che ci sia abbastanza spazio sopra l'orecchio 
            y_min = max(0, eye_center_y - min_margin_pixels)
            y_max = min(height, y_min + max_height)

            # Ritaglia mantenendo un'immagine rettangolare
            rectangular_image = reduced_width_image[y_min:y_max, :]
            return rectangular_image
        
        # Calcola il margine minimo sopra l'orecchio
        min_margin_pixels = int(height * self.ear_config.post_processing.min_margin_above_ear)

        # Verifica che ci sia abbastanza spazio sopra l'orecchio
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
        clahe = cv2.createCLAHE(clipLimit=self.ear_config.post_processing.clahe.cliplimit, tileGridSize=self.ear_config.post_processing.clahe.tilegridsize)
        return clahe.apply(image)
    
    def _resize_image(self, image):
        return cv2.resize(image, (self.ear_config.post_processing.image_size, self.ear_config.post_processing.image_size), interpolation=cv2.INTER_AREA)    # cv2.INTER_NEAREST

    # def post_processing_image(self, image, bounding_box):

    #     # Ottieni le dimensioni dell'immagine
    #     img_h, img_w, _ = image.shape

    #     x_min = int(bounding_box[0] * img_w)
    #     y_min = int(bounding_box[1] * img_h)
    #     x_max = int(bounding_box[2] * img_w)
    #     y_max = int(bounding_box[3] * img_h)

    #     bounding_box = [x_min, y_min, x_max, y_max]
        
    #     # Ritaglia l'area del volto dall'immagine originale
    #     ear_image = image[y_min:y_max, x_min:x_max]

    #     if self.ear_config.show_images.post_processed_ear_image:
    #         cv2.imshow("Cropped ear image", ear_image)
    #         cv2.moveWindow("Cropped ear image", 0, 0)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
        
    #     # Converti in scala di grigi
    #     image = self._bgr_to_gray(image.copy())
    #     ear_image_gray = self._bgr_to_gray(ear_image)

    #     if self.ear_config.show_images.post_processed_ear_image:
    #         cv2.imshow("Gray ear image", ear_image_gray)
    #         cv2.moveWindow("Gray ear image", 400, 0)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    #     ear_image_aligmented, ear_center = self._ear_alignment(image, ear_image_gray, bounding_box)

    #     if self.ear_config.show_images.post_processed_ear_image:
    #         cv2.imshow("Alignmented ear image", ear_image_aligmented)
    #         cv2.moveWindow("Alignmented ear image", 800, 0)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #     ear_image_cropped = self._cropping_image(ear_image_aligmented, ear_center)

    #     if self.ear_config.show_images.post_processed_ear_image:
    #         cv2.imshow("Cropped to square ear image", ear_image_cropped)
    #         cv2.moveWindow("Cropped to square ear image", 1200, 0)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #     # Ridimensiona l'immagine a una dimensione fissa
    #     ear_image_resized = self._resize_image(ear_image_cropped)

    #     if self.ear_config.show_images.post_processed_ear_image:
    #         cv2.imshow("Resized ear image", ear_image_resized)
    #         cv2.moveWindow("Resized ear image", 0, 400)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #     # Utilizza CLAHE per un miglioramento locale del contrasto
    #     ear_image_equalized = self._apply_clahe(ear_image_resized)

    #     if self.ear_config.show_images.post_processed_ear_image:
    #         cv2.imshow("Equalized ear image", ear_image_equalized)
    #         cv2.moveWindow("Equalized ear image", 400, 400)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
        
    #     return ear_image_equalized, ear_image_cropped.shape

    def post_processing_image(self, image, bounding_box):

        # Ottieni le dimensioni dell'immagine
        img_h, img_w, _ = image.shape

        x_min = int(bounding_box[0] * img_w)
        y_min = int(bounding_box[1] * img_h)
        x_max = int(bounding_box[2] * img_w)
        y_max = int(bounding_box[3] * img_h)

        bounding_box = [x_min, y_min, x_max, y_max]
        
        # Ritaglia l'area dell'orecchio dall'immagine originale
        ear_image = image[y_min:y_max, x_min:x_max]

        if self.ear_config.show_images.post_processed_ear_image:
            cv2.imshow("Cropped ear image", ear_image)
            cv2.moveWindow("Cropped ear image", 0, 0)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # Get alignment points
        ear_image_aligmented, top_point, bottom_point, outer_point, inner_point, center_point = self._ear_alignment(image.copy(), bounding_box)

        if self.ear_config.show_images.post_processed_ear_image:
            cv2.imshow("Alignmented ear image", ear_image_aligmented)
            cv2.moveWindow("Alignmented ear image", 400, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Converti in scala di grigi
        # image = self._bgr_to_gray(image.copy())
        ear_image_gray = self._bgr_to_gray(ear_image_aligmented)

        if self.ear_config.show_images.post_processed_ear_image:
            cv2.imshow("Gray ear image", ear_image_gray)
            cv2.moveWindow("Gray ear image", 800, 0)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # Ridimensiona l'immagine a una dimensione fissa
        ear_image_resized = self._resize_image(ear_image_gray)

        if self.ear_config.show_images.post_processed_ear_image:
            cv2.imshow("Resized ear image", ear_image_resized)
            cv2.moveWindow("Resized ear image", 0, 400)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return ear_image_resized, ear_image_gray.shape
    
    def add_padding(self, image, width, height):
        if height - image.shape[0] < 2 and width - image.shape[1] < 2:
            return image
        elif height - image.shape[0] < 2:
            pad_top = 0
            pad_bottom = 0
            pad_left = int((width - image.shape[1]) // 2)
            pad_right = width - image.shape[1] - pad_left
        elif width - image.shape[1] < 2:
            pad_top = int((height - image.shape[0]) // 2)
            pad_bottom = height - image.shape[0] - pad_top
            pad_left = 0
            pad_right = 0
        else:
            pad_top = int((height - image.shape[0]) // 2)
            pad_bottom = height - image.shape[0] - pad_top
            pad_left = int((width - image.shape[1]) // 2)
            pad_right = width - image.shape[1] - pad_left

        # Aggiunge il padding
        padded_ear_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        if self.ear_config.show_images.padded_ear_image:
            cv2.imshow("Padded ear image", padded_ear_image)
            cv2.moveWindow("Padded ear image", 0, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return padded_ear_image

    def resize_image(self, image, width, height):
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)    # cv2.INTER_NEAREST
