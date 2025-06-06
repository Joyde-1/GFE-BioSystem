import cv2
import numpy as np
from rembg import remove
from PIL import Image


class FacePreProcessing:
    def __init__(self, face_config):
        self.face_config = face_config

    def pre_processing_image(self, image):
        # Converti l'immagine OpenCV in PIL (necessario per rembg)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Rimuove lo sfondo
        if self.face_config.pre_processing.remove_background:
            pre_processed_face_image = remove(image)
        else:
            pre_processed_face_image = image.copy()

        # Converte l'immagine PIL nuovamente in OpenCV
        pre_processed_face_image = np.array(pre_processed_face_image)
        # pre_processed_face_image = np.array(pre_processed_face_image)
        pre_processed_face_image = cv2.cvtColor(pre_processed_face_image, cv2.COLOR_RGB2BGR)
        if self.face_config.show_images.pre_processed_face_image:
            cv2.imshow("No background face image", pre_processed_face_image)
            cv2.moveWindow("No background face image", 0, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return pre_processed_face_image