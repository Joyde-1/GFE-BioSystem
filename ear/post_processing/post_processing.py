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

    def _apply_clahe(self, image):
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=self.ear_config.post_processing.clahe.cliplimit, tileGridSize=self.ear_config.post_processing.clahe.tilegridsize)
        return clahe.apply(image)
    
    def _resize_image(self, image):
        return cv2.resize(image, (self.ear_config.post_processing.image_size, self.ear_config.post_processing.image_size), interpolation=cv2.INTER_AREA)    # cv2.INTER_NEAREST

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