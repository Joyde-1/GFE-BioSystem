import cv2
from tqdm import tqdm
import os
import json
import random
import glob
from PIL import Image
from sklearn.model_selection import train_test_split


class PrepareData():
    """
    Prepares detection data for deep learning model.

    Attributes
    ----------
    data_paths : str
        Base directory where detection data is stored.
    data : Dict
        This dictionary holds the processed data. It contains two keys: 'images' and 'labels'.
    """

    def __init__(self, gait_keypoints_detection_config):
        """
        Initializes the PrepareData instance

        Parameters
        ----------
        data_paths : str
            The path to the base directory containing database
        """
        
        self._gait_keypoints_detection_config = gait_keypoints_detection_config

    def _get_image_names(self):
        images_path = []
        images_file = []

        data = {
            'image_names': [],
            'subjects': []
        }

        images_path = os.path.join(self._gait_keypoints_detection_config.data_dir, self.biometric_trait)

        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Directory {images_path} not found.")

        images_file = [f for f in os.listdir(images_path) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.png')]

        # tqdm is used to show the progress barr
        for image_file in tqdm(images_file, desc=f"Loading {self.biometric_trait} image names", unit="file"):
            if image_file.endswith(".bmp") or image_file.endswith('.jpg') or image_file.endswith('.png'):
                # Carica l'immagine
                image_path = os.path.join(images_path, image_file)

                image_name = os.path.basename(os.path.splitext(image_path)[0])

                data['image_names'].append(image_name)

                # Extract the subject number from the image name
                data['subjects'].append(image_name.split('_')[0])

        return images_path, images_file, data
    
    def _load_silhouette_paths(self):
        """
        Carica i percorsi di tutte le silhouette per i soggetti da 00001 a 00049
        da tutte le cartelle Silhouette_* disponibili.
        
        Returns
        -------
        dict
            Dizionario con soggetti come chiavi e liste di percorsi come valori
        """
        silhouette_data = {
            'frame_paths': [],
            'subjects': [],
            'subject_to_paths': {}
        }
        
        # Cerca tutte le cartelle Silhouette_* nella directory dei dati
        silhouette_dirs = glob.glob(os.path.join(self._gait_keypoints_detection_config.frames_dir, "Silhouette_*"))
        
        if not silhouette_dirs:
            raise FileNotFoundError(f"Nessuna directory Silhouette_* trovata in {self._gait_keypoints_detection_config.frames_dir}")
        
        # Per ogni cartella Silhouette_*
        for silhouette_dir in silhouette_dirs:
            print(f"Elaborazione della directory: {silhouette_dir}")
            
            # Per ogni soggetto da 00001 a 00049
            for subject_num in range(1, 50):
                subject_id = f"{subject_num:05d}"  # Formatta come 00001, 00002, ecc.
                
                # Cerca tutte le immagini per questo soggetto nella cartella corrente
                subject_pattern = os.path.join(silhouette_dir, subject_id, "*.png")
                subject_files = glob.glob(subject_pattern)

                print("Silhouette files found for subject:", subject_id)
                print(subject_files)
                
                # Se abbiamo trovato file per questo soggetto
                for file_path in subject_files:
                    silhouette_data['frame_paths'].append(file_path)
                    silhouette_data['subjects'].append(subject_id)
                    
                    # Aggiungi al dizionario soggetto -> percorsi
                    if subject_id not in silhouette_data['subject_to_paths']:
                        silhouette_data['subject_to_paths'][subject_id] = []
                    
                    silhouette_data['subject_to_paths'][subject_id].append(file_path)
        
        # Verifica che abbiamo trovato dati
        if not silhouette_data['frame_paths']:
            raise FileNotFoundError(f"Nessuna silhouette trovata per i soggetti da 00001 a 00049")
        
        print(f"Trovate {len(silhouette_data['frame_paths'])} silhouette per {len(silhouette_data['subject_to_paths'])} soggetti")
        print("subjects:", len(silhouette_data['subjects']))
        
        return silhouette_data
    
    def _data_splitting(self, data):
        """
        Divide i dati in set di training, validation e test assicurando che
        ogni soggetto appaia solo in uno dei tre subset.
    
        Parameters
        ----------
        data : dict
            Dizionario contenente 'image_names' e 'subjects' come chiavi.
    
        Returns
        -------
        tuple
            Restituisce tre liste: nomi delle immagini per training, validation e test.
        """
        # Ottieni soggetti unici
        unique_subjects = list(set(data['subjects']))
        
        # Dividi i soggetti in train, validation e test
        train_subjects, test_subjects = train_test_split(
            unique_subjects, 
            test_size=self._gait_keypoints_detection_config.data.test_size,
            random_state=42, 
            shuffle=True
        )

        # Dividi i soggetti rimanenti tra validation e test
        train_subjects, val_subjects = train_test_split(
            train_subjects, 
            test_size=self._gait_keypoints_detection_config.data.val_size,
            random_state=42, 
            shuffle=True
        )
        
        # Inizializza le liste per i nomi delle immagini
        train_set_frame_paths = []
        val_set_frame_paths = []
        test_set_frame_paths = []
        
        # Assegna ogni immagine al set appropriato in base al soggetto
        for idx, subject in enumerate(data['subjects']):
            frame_path = data['frame_paths'][idx]
            
            if subject in train_subjects:
                train_set_frame_paths.append(frame_path)
            elif subject in val_subjects:
                val_set_frame_paths.append(frame_path)
            else:  # subject in test_subjects
                test_set_frame_paths.append(frame_path)
        
        print(f"Split completato: {len(train_set_frame_paths)} immagini di training, "
              f"{len(val_set_frame_paths)} immagini di validation, "
              f"{len(test_set_frame_paths)} immagini di test")
        print(f"Soggetti: {len(train_subjects)} in training, "
              f"{len(val_subjects)} in validation, "
              f"{len(test_subjects)} in test")
        
        return train_set_frame_paths, val_set_frame_paths, test_set_frame_paths
    
    def _data_augmentatation(self, image):
        if random.random() < 0.5:
            alpha = 1.03
            beta = 3
        else:
            alpha = 0.97
            beta = -3
        image = cv2.convertScaleAbs(image.copy(), alpha=alpha, beta=beta)
        return image
    
    def _convert_openpose_to_coco(self, keypoints_data, frame, frame_path):
        """
        Converte i keypoints di OpenPose nel formato COCO compatibile con MMPose.
        
        Parameters
        ----------
        keypoints_data : dict
            Dizionario contenente i keypoints estratti con OpenPose
        frame : numpy.ndarray
            Immagine del frame
        frame_path : str
            Percorso del frame
            
        Returns
        -------
        dict
            Dizionario in formato COCO
        """
        # Estrai il nome del file dal percorso
        frame_filename = os.path.basename(frame_path)
        frame_name = os.path.splitext(frame_filename)[0]
        
        # Ottieni le dimensioni dell'immagine
        height, width = frame.shape[:2]
        
        # Crea l'informazione dell'immagine
        image_info = {
            "id": frame_name,
            "file_name": frame_filename,
            "width": width,
            "height": height
        }
        
        # Verifica se ci sono persone rilevate
        if "people" not in keypoints_data or len(keypoints_data["people"]) == 0:
            print(f"Nessuna persona rilevata nel frame: {frame_path}")
            return None
        
        # Prendi i keypoints della prima persona (assumiamo che ci sia solo una persona per frame)
        person = keypoints_data["people"][0]
        
        # Estrai i keypoints 2D della posa
        pose_keypoints = person.get("pose_keypoints_2d", [])
        
        # OpenPose restituisce i keypoints come [x1, y1, c1, x2, y2, c2, ...] dove c è la confidenza
        # MMPose si aspetta [x1, y1, c1, x2, y2, c2, ...]
        # Dobbiamo solo assicurarci che ci siano tutti i keypoints necessari
        
        # Verifica che ci siano abbastanza keypoints (18 keypoints * 3 valori = 54)
        if len(pose_keypoints) < 54:
            print(f"Numero insufficiente di keypoints nel frame: {frame_path}")
            return None
        
        # Calcola la bounding box che racchiude tutti i keypoints
        xs = []
        ys = []
        for i in range(0, len(pose_keypoints), 3):
            x, y, conf = pose_keypoints[i:i+3]
            if conf > 0:  # Considera solo i keypoints con confidenza > 0
                xs.append(x)
                ys.append(y)
        
        if not xs or not ys:
            print(f"Nessun keypoint valido nel frame: {frame_path}")
            return None
        
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        # Conta il numero di keypoints validi
        num_keypoints = sum(1 for i in range(0, len(pose_keypoints), 3) if pose_keypoints[i+2] > 0)
        
        # Crea l'informazione dell'annotazione
        annotation_info = {
            "id": frame_name,
            "image_id": frame_name,
            "category_id": 1,  # Categoria "person"
            "keypoints": pose_keypoints,
            "num_keypoints": num_keypoints,
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "area": bbox_width * bbox_height,
            "iscrowd": 0
        }
        
        # Crea il dizionario COCO
        coco_dict = {
            "images": [image_info],
            "annotations": [annotation_info],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": [
                        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
                        "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
                        "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
                        "left_eye", "right_ear", "left_ear"
                    ],
                    "skeleton": [
                        [1, 2], [2, 3], [3, 4], [4, 5], [2, 6], [6, 7], [7, 8], [2, 9], [9, 10],
                        [10, 11], [2, 12], [12, 13], [13, 14], [1, 15], [15, 17], [1, 16], [16, 18]
                    ]
                }
            ]
        }
        
        return coco_dict
    
    def _convert_labelme_to_coco(self, gait_keypoints_path, image_id, annotation_id, category_id=1):
        # Carica il file LabelMe
        with open(gait_keypoints_path, 'r') as f:
            data = json.load(f)
        
        # Estrai le informazioni dell'immagine
        file_name = os.path.basename(data['imagePath'])
        width = data['imageWidth']
        height = data['imageHeight']

        # Definisci il target di ridimensionamento (es. image_size x image_size)
        scale_x = self._gait_keypoints_detection_config.data.image_size / width
        scale_y = self._gait_keypoints_detection_config.data.image_size / height
        
        image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": self._gait_keypoints_detection_config.data.image_size,
            "height": self._gait_keypoints_detection_config.data.image_size
        }
        
        # Definisci l'ordine dei keypoint che vuoi utilizzare
        # In questo esempio consideriamo quattro keypoint: "top", "bottom", "outer", "inner"
        keypoints_order = ['top', 'bottom', 'outer', 'inner']
        keypoints = []
        
        # Per ciascun keypoint, cerca la corrispondente annotazione in LabelMe
        for kp in keypoints_order:
            point = None
            for shape in data['shapes']:
                if shape['label'] == kp:
                    # Presupponiamo che ogni shape contenga un solo punto
                    point = shape['points'][0]
                    break
            if point is None:
                raise ValueError(f"Annotazione mancante per il keypoint: {kp} and {image_id}")
            x, y = point
            # Applica il ridimensionamento: scala le coordinate
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            # Aggiungi la tripla [x, y, v] (v = 2 per indicare che è visibile)
            keypoints.extend([x_scaled, y_scaled, 2])
        
        num_keypoints = len(keypoints_order)
        
        # Calcola la bounding box che racchiude tutti i keypoint
        xs = []
        ys = []
        for shape in data["shapes"]:
            # Considera solo le annotazioni per i keypoint che ci interessano
            if shape["label"].lower() in [k.lower() for k in keypoints_order]:
                pt = shape["points"][0]
                xs.append(pt[0] * scale_x)
                ys.append(pt[1] * scale_y)
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            # "category_id": category_id,
            "keypoints": keypoints,
            "num_keypoints": num_keypoints,
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "area": bbox_width * bbox_height,
            "iscrowd": 0
        }

        coco_dict = {
            # "info": {
            #     "description": "Dataset per il rilevamento dei landmark dell'orecchio",
            #     "version": "1.0",
            #     "year": 2023,
            #     "contributor": "Tuo Nome",
            #     "date_created": "2023-09-30"
            # },
            # "licenses": [
            #     {
            #         "id": 1,
            #         "name": "Attribution-NonCommercial",
            #         "url": "http://creativecommons.org/licenses/by-nc/4.0/"
            #     }
            # ],
            "images": [image_info],
            "annotations": [annotation_info],
            "categories": [
                {
                    # "id": 1,
                    "name": self.biometric_trait,
                    "supercategory": self.biometric_trait,
                    "keypoints": ["top", "bottom", "left", "right"],
                    "skeleton": [[1, 2], [2, 3], [3, 4]]
                }
            ]
        }
        
        return coco_dict
    
    def _visualize_gait_keypoints(self, gait_keypoints_path, image_path):
        # Carica il file COCO JSON
        with open(gait_keypoints_path, 'r') as f:
            coco_data = json.load(f)
        
        # Costruisci un dizionario per accedere rapidamente alle informazioni dell'immagine (chiave: image id)
        images_dict = {img["id"]: img for img in coco_data["images"]}
        
        # Costruisci un dizionario per raggruppare le annotazioni per image_id
        annotations_dict = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in annotations_dict:
                annotations_dict[image_id] = []
            annotations_dict[image_id].append(ann)
        
        # Recupera i nomi dei keypoint dalla categoria (si assume che il dataset abbia una sola categoria)
        keypoint_names = []
        if "categories" in coco_data and len(coco_data["categories"]) > 0:
            keypoint_names = coco_data["categories"][0].get("keypoints", [])
        
        # Itera su tutte le immagini del dataset
        for image_id, img_info in images_dict.items():
            image = cv2.imread(image_path)
            if image is None:
                print(f"Immagine non trovata: {image_path}")
                continue
            
            # Se ci sono annotazioni per questa immagine, disegna i keypoint
            if image_id in annotations_dict:
                for ann in annotations_dict[image_id]:
                    keypoints = ann["keypoints"]  # Lista di [x, y, v, x, y, v, ...]
                    num_keypoints = len(keypoints) // 3
                    for i in range(num_keypoints):
                        x = int(keypoints[3 * i])
                        y = int(keypoints[3 * i + 1])
                        v = keypoints[3 * i + 2]  # visibilità (2: visibile)
                        if v > 0:
                            # Disegna un cerchio verde sul keypoint
                            cv2.circle(image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
                            # Se sono disponibili i nomi, li disegna accanto al punto
                            if i < len(keypoint_names):
                                cv2.putText(image, keypoint_names[i], (x + 5, y - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            # Mostra l'immagine con i keypoint sovrapposti
            cv2.imshow("Image with gait_keypoints", image)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()

    def _load_and_save_images_and_gait_keypoints(self, images_path, images_file, train_set_image_names, val_set_image_names):
        # Percorsi di salvataggio
        for subset in ['train', 'val', 'test']:
            subset_path = os.path.join(self._gait_keypoints_detection_config.save_data_splitted_path, "splitted_gait_keypoints_database", self.biometric_trait, subset)

            os.makedirs(subset_path, exist_ok=True)
    
            os.makedirs(os.path.join(subset_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(subset_path, "labels"), exist_ok=True)

        for image_file in tqdm(images_file, desc=f"Loading {self.biometric_trait} image files", unit="file"):
            if image_file.endswith(".bmp") or image_file.endswith('.jpg') or image_file.endswith('.png'):
                # Carica l'immagine
                image_path = os.path.join(images_path, image_file)

                image_name = os.path.basename(os.path.splitext(image_path)[0])

                gait_keypoints_path = os.path.join(self._gait_keypoints_detection_config.gait_keypoints_dir, self.biometric_trait, image_name + '.json')

                # image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                if image is None:
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                if image_name in train_set_image_names or image_name in val_set_image_names:
                    image = self._data_augmentatation(image.copy())

                # Converti da BGR (OpenCV) a RGB (Pillow)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = Image.fromarray(image).resize((self._gait_keypoints_detection_config.data.image_size, self._gait_keypoints_detection_config.data.image_size))

                gait_keypoints = self._convert_labelme_to_coco(gait_keypoints_path, image_name, image_name)

                # Determina il subset (train, val, test)
                if image_name in train_set_image_names:
                    subset = 'train'
                elif image_name in val_set_image_names:
                    subset = 'val'
                else:
                    subset = 'test'

                # Salva l'immagine e i gait_keypoints in formato COCO nella directory corrispondente
                subset_image_path = os.path.join(self._gait_keypoints_detection_config.save_data_splitted_path, "splitted_gait_keypoints_database", self.biometric_trait, subset, "images", f"{image_name}.bmp")
                subset_gait_keypoints_path = os.path.join(self._gait_keypoints_detection_config.save_data_splitted_path, "splitted_gait_keypoints_database", self.biometric_trait, subset, "labels", f"{image_name}.json")
                
                # Salva l'immagine nella directory corrispondente
                image.save(subset_image_path)

                # Salva i gait_keypoints in formato COCO nel file corrispondente
                with open(subset_gait_keypoints_path, 'w') as f:
                    json.dump(gait_keypoints, f, indent=4)

                # self._visualize_gait_keypoints(subset_gait_keypoints_path, subset_image_path)
    
    # Funzione per caricare le immagini
    def prepare_data(self):
        # Carica i percorsi delle silhouette
        silhouette_data = self._load_silhouette_paths()
        
        # Esegui lo split dei dati
        train_set_frame_paths, val_set_frame_paths, test_set_frame_paths = self._data_splitting(silhouette_data)

        # Percorsi di salvataggio
        for subset in ['train', 'val', 'test']:
            subset_path = os.path.join(self._gait_keypoints_detection_config.save_data_splitted_path, "splitted_gait_keypoints_database", subset)

            os.makedirs(subset_path, exist_ok=True)
    
            os.makedirs(os.path.join(subset_path, "frames"), exist_ok=True)
            os.makedirs(os.path.join(subset_path, "keypoints"), exist_ok=True)
        
        # Carica i keypoints per ciascun subset
        train_data = self._load_keypoints_for_frames(train_set_frame_paths, "train")
        val_data = self._load_keypoints_for_frames(val_set_frame_paths, "val")
        test_data = self._load_keypoints_for_frames(test_set_frame_paths, "test")
        
        return train_data, val_data, test_data
    
    def _load_keypoints_for_frames(self, subset_frame_paths, subset):
        """
        Carica i file di keypoints per i frame specificati dalla cartella OUMVLP-Pose.
        
        Parameters
        ----------
        subset_frame_paths : list
            Lista dei percorsi dei frame per cui caricare i keypoints
            
        Returns
        -------
        dict
            Dizionario contenente i frame caricati e i relativi keypoints
        """
        result = {
            'frames': [],
            'keypoints': []
        }
        
        # Percorso base della cartella dei keypoints
        keypoints_base_dir = os.path.join(self._gait_keypoints_detection_config.keypoints_dir, "00")
        
        if not os.path.exists(keypoints_base_dir):
            raise FileNotFoundError(f"Directory dei keypoints {keypoints_base_dir} non trovata.")
        
        # Itera su tutti i percorsi dei frame
        for frame_path in tqdm(subset_frame_paths, desc="Caricamento frame e keypoints", unit="frame"):
            # Estrai il soggetto e il tipo di sequenza dal percorso del frame
            # Esempio: /path/to/Silhouette_00-01/00001/000_00_180_090/0001.png
            parts = frame_path.split(os.sep)
            
            # Verifica che il percorso abbia la struttura attesa
            if len(parts) < 4:
                print(f"Percorso del frame non valido: {frame_path}")
                continue

            # print(f"Percorso del frame: {frame_path}")
            # print(f"Numero di parti: {len(parts)}")
            # print(f"Parti: {parts}")
            
            # Estrai il soggetto (es. 00001)
            subject_id = None
            for part in parts:
                if part.isdigit() and len(part) == 5 and 1 <= int(part) <= 49:
                    subject_id = part
                    # print(f"ID soggetto trovato: {subject_id}")
                    break

            if not subject_id:
                print(f"ID soggetto non trovato nel percorso: {frame_path}")
                continue
                
            # Estrai il nome della sequenza (es. 000_00_180_090)
            sequence_name = None
            for i, part in enumerate(parts):
                if part == subject_id and i+1 < len(parts):
                    sequence_name = parts[i-1].split('_')[1].split('-')[0] + '_' + parts[i-1].split('_')[1].split('-')[1]
                    #sequence_name = parts[i+1]
                    break
            # print(f"Nome sequenza trovato: {sequence_name}")
            
            if not sequence_name:
                print(f"Nome sequenza non trovato nel percorso: {frame_path}")
                continue

            # Estrai il nome del file (es. 0001.png)
            frame_filename = os.path.basename(frame_path)
            frame_name = os.path.splitext(frame_filename)[0]
            
            # Costruisci il percorso del file dei keypoints
            # La struttura attesa è: OUMVLP-Pose/00/00001/000_00_180_090/0001.json
            keypoints_path = os.path.join(keypoints_base_dir, subject_id, sequence_name, f"{frame_name}_keypoints.json")
            # print(f"Percorso keypoints: {keypoints_path}")
            
            # Verifica se esiste il file dei keypoints
            if not os.path.exists(keypoints_path):
                # Se non esiste, salta questo frame
                # print(f"File keypoints non trovato: {keypoints_path}")
                continue
            
            # Carica l'immagine del frame
            # frame = cv2.imread(frame_path)
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            
            if frame is None:
                print(f"Impossibile caricare il frame: {frame_path}")
                continue
            
            # Carica il file JSON dei keypoints
            try:
                with open(keypoints_path, 'r') as f:
                    keypoints = json.load(f)
                
                # Converti i keypoints da OpenPose a COCO
                keypoints = self._convert_openpose_to_coco(keypoints, frame, frame_path)
                
                # Se la conversione ha avuto successo, aggiungi il frame e i keypoints ai risultati
                if keypoints:
                    result['frames'].append(frame)
                    result['keypoints'].append(keypoints)
                else:
                    # Se la conversione fallisce, salta questo frame
                    print(f"Conversione fallita per il frame: {frame_path}")
                    continue
                
            except Exception as e:
                print(f"Errore nel caricamento del file keypoints {keypoints_path}: {e}")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = Image.fromarray(frame).resize((self._gait_keypoints_detection_config.data.image_size, self._gait_keypoints_detection_config.data.image_size))

            # Salva l'immagine e i gait_keypoints in formato COCO nella directory corrispondente
            subset_image_path = os.path.join(self._gait_keypoints_detection_config.save_data_splitted_path, "splitted_gait_keypoints_database", subset, "frames", f"{subject_id}_{sequence_name}_{frame_name}.png")
            subset_gait_keypoints_path = os.path.join(self._gait_keypoints_detection_config.save_data_splitted_path, "splitted_gait_keypoints_database", subset, "keypoints", f"{subject_id}_{sequence_name}_{frame_name}.json")
            
            # Salva l'immagine nella directory corrispondente
            frame.save(subset_image_path)

            # Salva i gait_keypoints in formato COCO nel file corrispondente
            with open(subset_gait_keypoints_path, 'w') as f:
                json.dump(keypoints, f, indent=4)

            # self._visualize_gait_keypoints(subset_gait_keypoints_path, subset_image_path)
        
        print(f"Caricati {len(result['frames'])} frame con {len(result['keypoints'])} keypoints corrispondenti")
        
        return result