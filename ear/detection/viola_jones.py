import cv2
import numpy as np


class ViolaJones:
    def __init__(self, face_config):
        self.face_config = face_config

    def _prepare_detection_process(self):
        # Usa il file Haar Cascade fornito da OpenCV per il rilevamento dell'orecchio
        #self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Inizializza diverse varianti di cascata per volti frontali e di profilo
        self.face_cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            #cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        ]

    def _preprocess(self, image, target_width=640):
        """
        Pre-elabora l'immagine: conversione in scala di grigi, equalizzazione dell'istogramma
        e correzione gamma.
        """
        # Ottieni le dimensioni originali
        h, w = image.shape[:2]

        # Calcola il nuovo rapporto d'aspetto
        aspect_ratio = h / w
        new_height = int(target_width * aspect_ratio)

        # Ridimensiona l'immagine
        image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # Correzione gamma per migliorare il contrasto
        gamma = 1.2  # Puoi sperimentare con altri valori
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        adjusted = cv2.LUT(gray, table)
        return adjusted
    
    def detect_face(self, image):
        """
        Rileva il volto nell'immagine e restituisce le coordinate normalizzate della bounding box.
        
        :param image: Immagine di input (array NumPy in formato BGR).
        :return: Tuple (x_min, y_min, x_max, y_max) con coordinate normalizzate,
                 oppure None se nessun volto viene trovato.
        """
        self._prepare_detection_process()
        
        # Otteniamo le dimensioni dell'immagine
        original_height, original_width = image.shape[:2]

        # Convertiamo l'immagine in scala di grigi, essenziale per il rilevamento con Viola-Jones
        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image_gray = cv2.equalizeHist(image_gray)

        preprocessed_image = self._preprocess(image.copy())

        # 2. Prova diverse configurazioni per la rilevazione
        param_sets = [
            {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},
            {'scaleFactor': 1.1, 'minNeighbors': 3, 'minSize': (30, 30)},
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20)}
        ]

        # Rilevamento dei volti: scaleFactor e minNeighbors possono essere regolati in base all'applicazione
        # detected_faces = self.face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5)
        # detected_faces = self.face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30), maxSize=(180, 180), flags=cv2.CASCADE_SCALE_IMAGE)
        
        detected_faces = []
        
        # for params in param_sets:
        #     faces = self.face_cascade.detectMultiScale(image_gray, **params)
        #     if len(faces) > 0:
        #         detected_faces = faces

        # 1. Prova a rilevare il volto usando ciascun classificatore e ciascun set di parametri
        for face_cascade in self.face_cascades:
            for params in param_sets:
                faces = face_cascade.detectMultiScale(preprocessed_image, **params)
                if len(faces) > 0:
                    detected_faces.extend(faces)
        
        # Se non viene rilevato nessun volto, restituiamo None
        if len(detected_faces) == 0:
            print("Any face detected in the image with the actual params.")

        print("BBOXES:", len(detected_faces))
        
        # Se vengono rilevati più volti, scegliamo quello con la maggiore area
        x, y, w, h = max(detected_faces, key=lambda rect: rect[2] * rect[3])

        # Otteniamo le dimensioni dell'immagine
        new_height, new_width = preprocessed_image.shape[:2]
        
        # Calcoliamo le coordinate normalizzate (valori tra 0 e 1)
        x_min_norm = x / new_width
        y_min_norm = y / new_height
        x_max_norm = (x + w) / new_width
        y_max_norm = (y + h) / new_height

        x_min = int(x_min_norm * original_width)
        y_min = int(y_min_norm * original_height)
        x_max = int(x_max_norm * original_width)
        y_max = int(y_max_norm * original_height)

        plot_best_bounding_box_image = image.copy()

        # Disegna la buonding box predetta sull'immagine
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)
        cv2.rectangle(plot_best_bounding_box_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)
        

        # Estrae i volti rilevati
        for (x, y, w, h) in detected_faces:
            x_min = x / new_width
            y_min = y / new_height
            x_max = (x + w) / new_width
            y_max = (y + h) / new_height

            x_min = int(x_min * original_width)
            y_min = int(y_min * original_height)
            x_max = int(x_max * original_width)
            y_max = int(y_max * original_height)

            # Disegna un rettangolo attorno a ciascun volto rilevato
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if self.face_config.show_images.detected_face_bounding_box:
            cv2.imshow("Detected faces bounding box image", image)
            cv2.moveWindow("Detected faces bounding box image", 0, 0)

        if self.face_config.show_images.detected_face_bounding_box:
            cv2.imshow("Detected best face bounding box image", plot_best_bounding_box_image)
            cv2.moveWindow("Detected best face bounding box image", 0, 400)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image, [x_min_norm, y_min_norm, x_max_norm, y_max_norm]