import cv2
import numpy as np
from rembg import remove
from PIL import Image


class EarPreProcessing:
    def __init__(self, ear_config):
        self.ear_config = ear_config

    def pre_processing_image(self, image):
        # Converti l'immagine OpenCV in PIL (necessario per rembg)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Rimuove lo sfondo
        if self.ear_config.pre_processing.remove_background:
            pre_processed_ear_image = remove(image)
        else:
            pre_processed_ear_image = image.copy()

        # Converte l'immagine PIL nuovamente in OpenCV
        pre_processed_ear_image = np.array(pre_processed_ear_image)
        pre_processed_ear_image = cv2.cvtColor(pre_processed_ear_image, cv2.COLOR_RGB2BGR)
        if self.ear_config.show_images.pre_processed_ear_image:
            cv2.imshow("No background ear image", pre_processed_ear_image)
            cv2.moveWindow("No background ear image", 0, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return pre_processed_ear_image