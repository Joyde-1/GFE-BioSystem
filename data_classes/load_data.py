import cv2
from tqdm import tqdm
import os
import glob


class LoadData:

    def __init__(self):
        """
        Initializes the PrepareData instance
        """
    
    # Funzione per caricare le immagini
    def load_images(self, config, biometric_trait):
        
        images = []

        image_names = []

        image_paths = []
            
        path = os.path.join(config.data_dir, biometric_trait)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory {path} not found.")
        
        image_files = [f for f in os.listdir(path) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.png')]

        # Ordinamento in base al primo e secondo numero nel nome:
        image_files = sorted(
            image_files,
            key=lambda x: (
                int(x.split('_')[0]),                          # primo numero
                int(x.split('_')[1].split('.')[0])               # secondo numero (rimuovo l'estensione)
            )
        )

        for image_file in tqdm(image_files, desc=f"Loading {biometric_trait} images"):
            image_path = os.path.join(path, image_file)

            # Carica l'immagine in scala di grigi
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if image is None:
                raise FileNotFoundError(f"Image {image_path} not found.")

            image_name = os.path.basename(os.path.splitext(image_path)[0])

            images.append(image)

            image_names.append(image_name)

            image_paths.append(image_path)
            
        return images, image_names, image_paths
  
    def load_frames(self, config, biometric_trait):
        """
        Carica le silhouette solo dalle cartelle
        000-00, 000-01, 015-00, 180-00 e 180-01
        e restituisce un dict:
        frame_sequences = {
            '1': [  # soggetto 1
            [<paths dei frame in 000-00/00001, ordinati>],
            [<paths dei frame in 000-01/00001, ordinati>],
            [<paths dei frame in 180-00/00001, ordinati>],
            [<paths dei frame in 180-01/00001, ordinati>],
            [<paths dei frame in 015-00/00001, ordinati>],
            ],
            '2': [ ... ],
            …
            '49': [ ... ]
        }
        """

        # Percorso base della cartella dei keypoints
        keypoints_base_dir = os.path.join(config.keypoints_sequences_dir, "00")
        
        if not os.path.exists(keypoints_base_dir):
            raise FileNotFoundError(f"Directory dei keypoints {keypoints_base_dir} non trovata.")
        
        frame_sequences = []
        frame_sequences_names = []
        frame_sequences_paths = []
        all_subject_ids = []
        all_sequence_names = []
        all_frame_names = []

        check = 0

        step = 5

        # per ogni soggetto da 1 a 49
        for subject in tqdm(range(1, config.data.num_classes + 1 + step), desc=f"Loading {biometric_trait} frames", unit="frame"):
            # frame_sequences[subject_id] = []
            # frame_sequences_paths[subject_id] = []

            if subject == 1 or subject == 7 or subject == 16 or subject == 20 or subject == 28:
                check += 1
                continue

            real_subject = subject - check
            real_subject_id = str(real_subject)
            
            # per ogni sequenza di interesse
            for num_sequence, sequence_name in enumerate(config.data.sequence_names):
                # path alla cartella della sequenza per il soggetto
                # es: root_dir/Silhouette_000-00/00001
                sequence_folder = os.path.join(
                    config.data_dir,
                    f"Silhouette_{sequence_name}",
                    f"{subject:05d}"    # soggetto formattato con 5 cifre, es '00001'
                )
                if not os.path.isdir(sequence_folder):
                    # se la cartella non esiste, salta
                    continue
                
                # prendi tutti i file .png e ordina per numero di frame
                png_files = [
                    fn for fn in os.listdir(sequence_folder)
                    if fn.lower().endswith('.png')
                ]
                # sort numerico basato sul nome (senza estensione)
                png_files.sort(key=lambda fn: int(os.path.splitext(fn)[0]))
                
                # crea la lista completa di path assoluti
                full_frame_paths = [os.path.join(sequence_folder, fn) for fn in png_files]

                frame_sequence = []
                frame_paths = []
                subject_ids = [] 
                sequence_names = []
                frame_names = []

                for frame_path in full_frame_paths:
                    # Estrai il nome del file (es. 0001.png)
                    frame_filename = os.path.basename(frame_path)
                    frame_name = os.path.splitext(frame_filename)[0]

                    keypoints_path = os.path.join(keypoints_base_dir, f"{subject:05d}", sequence_name.split('-')[0] + '_' + sequence_name.split('-')[1], f"{frame_name}_keypoints.json")
                    # print(f"Percorso keypoints: {keypoints_path}")
                    
                    # Verifica se esiste il file dei keypoints
                    if not os.path.exists(keypoints_path):
                        # Se non esiste, salta questo frame
                        # print(f"File keypoints non trovato: {keypoints_path}")
                        continue
                    else:
                        # Carica l'immagine del frame
                        # frame = cv2.imread(frame_path)
                        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                        frame_sequence.append(frame)
                        frame_paths.append(frame_path)
                        subject_ids.append(f"{real_subject:05d}")
                        sequence_names.append(sequence_name.split('-')[0] + '_' + sequence_name.split('-')[1])
                        frame_names.append(f"{frame_name}_keypoints.txt")
                if len(frame_sequence) == 0 and len(frame_paths) == 0:
                    # Se non ci sono frame validi, salta questa sequenza
                    continue
                frame_sequences.append(frame_sequence)
                # frame_sequences_names.append(f"{subject_id}_{sequence_name.split('-')[0] + '_' + sequence_name.split('-')[1]}")
                frame_sequences_names.append(f"{real_subject_id}_{num_sequence + 1}")
                frame_sequences_paths.append(frame_paths)
                all_subject_ids.append(subject_ids)
                all_sequence_names.append(sequence_names)
                all_frame_names.append(frame_names)
        
        # per es. stampo quanti frame trova per il soggetto '1' in ogni sequenza
        # for i, seq in enumerate(frame_sequences):
        #     print(f"Sequenza {i}: {len(seq)} frame")

        return frame_sequences, frame_sequences_names, frame_sequences_paths, all_subject_ids, all_sequence_names, all_frame_names