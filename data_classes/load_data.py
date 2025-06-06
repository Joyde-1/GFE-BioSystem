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
        # i codici delle sequenze che ci interessano
        sequence_names = ['000-00', '000-01', '180-00', '180-01', '015-00']

        # Percorso base della cartella dei keypoints
        keypoints_base_dir = os.path.join(config.keypoints_sequences_dir, "00")
        
        if not os.path.exists(keypoints_base_dir):
            raise FileNotFoundError(f"Directory dei keypoints {keypoints_base_dir} non trovata.")
        
        frame_sequences = []
        frame_sequences_names = []
        frame_sequences_paths = []

        # per ogni soggetto da 1 a 49
        for subject in tqdm(range(1, config.data.num_classes + 1), desc=f"Loading {biometric_trait} frames", unit="frame"):
            subject_id = str(subject)
            # frame_sequences[subject_id] = []
            # frame_sequences_paths[subject_id] = []
            
            # per ogni sequenza di interesse
            for num_sequence, sequence_name in enumerate(sequence_names):
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

                frame_sequences.append(frame_sequence)
                # frame_sequences_names.append(f"{subject_id}_{sequence_name.split('-')[0] + '_' + sequence_name.split('-')[1]}")
                frame_sequences_names.append(f"{subject_id}_{num_sequence + 1}")
                frame_sequences_paths.append(frame_paths)
        
        # per es. stampo quanti frame trova per il soggetto '1' in ogni sequenza
        for i, seq in enumerate(frame_sequences):
            print(f"Sequenza {i}: {len(seq)} frame")

        return frame_sequences, frame_sequences_names, frame_sequences_paths


    
    # def load_frames(self):
    #     """
    #     Carica i percorsi di tutte le silhouette per i soggetti da 00001 a 00049
    #     da tutte le cartelle Silhouette_* disponibili.
        
    #     Returns
    #     -------
    #     dict
    #         Dizionario con soggetti come chiavi e liste di percorsi come valori
    #     """
    #     silhouette_data = {
    #         'frame_paths': [],
    #         'subjects': [],
    #         'subject_to_paths': {}
    #     }
        
    #     # Cerca tutte le cartelle Silhouette_* nella directory dei dati
    #     silhouette_dirs = glob.glob(os.path.join(gait_keypoints_detection_config.frames_dir, "Silhouette_*"))
        
    #     if not silhouette_dirs:
    #         raise FileNotFoundError(f"Nessuna directory Silhouette_* trovata in {gait_keypoints_detection_config.frames_dir}")
        
    #     # Per ogni cartella Silhouette_*
    #     for silhouette_dir in silhouette_dirs:
    #         # print(f"Elaborazione della directory: {silhouette_dir}")
            
    #         # Per ogni soggetto da 00001 a 00049
    #         for subject_num in range(1, 50):   #1, 50
    #             subject_id = f"{subject_num:05d}"  # Formatta come 00001, 00002, ecc.
                
    #             # Cerca tutte le immagini per questo soggetto nella cartella corrente
    #             subject_pattern = os.path.join(silhouette_dir, subject_id, "*.png")
    #             subject_files = glob.glob(subject_pattern)

    #             # print("Silhouette files found for subject:", subject_id)
    #             # print(subject_files)
                
    #             # Se abbiamo trovato file per questo soggetto
    #             for file_path in subject_files:
    #                 silhouette_data['frame_paths'].append(file_path)
    #                 silhouette_data['subjects'].append(subject_id)
                    
    #                 # Aggiungi al dizionario soggetto -> percorsi
    #                 if subject_id not in silhouette_data['subject_to_paths']:
    #                     silhouette_data['subject_to_paths'][subject_id] = []
                    
    #                 silhouette_data['subject_to_paths'][subject_id].append(file_path)
        
    #     # Verifica che abbiamo trovato dati
    #     if not silhouette_data['frame_paths']:
    #         raise FileNotFoundError(f"Nessuna silhouette trovata per i soggetti da 00001 a 00049")
        
    #     print(f"Trovate {len(silhouette_data['frame_paths'])} silhouette per {len(silhouette_data['subject_to_paths'])} soggetti")
    #     print("subjects:", len(silhouette_data['subjects']))
        
    #     return silhouette_data