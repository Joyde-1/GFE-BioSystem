import os
import sys
import multiprocessing
import torch
import numpy as np
import random
import cv2
from yaml_config_override import add_arguments # Custom YAML config handling
from addict import Dict # Dictionary-like class that allows attribute access
import yaml # YAML parsing
from pathlib import Path  # Object-oriented filesystem paths
from ultralytics import YOLO
from tqdm import tqdm
from PyQt6.QtWidgets import QApplication, QFileDialog

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from yolo_detection.yolo_plots import plot_prediction


def load_config(my_config_path):
    """
    Loads the configuration from a YAML file and returns it as a dictionary.

    Parameters
    ----------
    my_config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    config : dict
        Configuration object loaded from the YAML file as a dictionary.

    Raises
    ------
    SystemExit
        If the configuration file does not exist at the specified path.

    Notes
    -----
    The configuration file needs to be in YAML format. This function will terminate the program
    if the file cannot be found, indicating the necessity of a valid configuration file for further operations.
    """

    # Check if the configuration file exists at the specified path
    if not os.path.exists(my_config_path):
        # Print an error message if the file does not exist and exit the program
        print("Error: configuration file does not exists: ", my_config_path)
        sys.exit(1)

    # Load the configuration from the YAML file
    config = yaml.safe_load(Path(my_config_path).read_text())
    
    # Convert the configuration to a dictionary using the add_arguments function
    config = Dict(add_arguments(config))

    return config

def browse_path(message):
    """
    Opens a file dialog to browse for a directory path.
    """

    # Assicura che QApplication sia avviata se non è già in esecuzione
    app = QApplication.instance()  # Controlla se esiste già un'applicazione PyQt
    if app is None:
        app = QApplication(sys.argv)  # Crea un'istanza se non esiste
        
    # Get the path of the Desktop folder
    desktop_path = str(Path.home() / "Desktop")
    
    # Open the QFileDialog starting from the Desktop folder
    path_directory = QFileDialog.getExistingDirectory(None, message, desktop_path)

    return path_directory

def select_device(yolo_detection_config):
    """
    Seleziona il dispositivo di calcolo appropriato (MPS, CUDA o CPU) in base alla preferenza e alla disponibilità.
    
    Parameters
    ----------
    config : dict, opzionale
        Il dispositivo preferito. Può essere "mps", "cuda" o "cpu". Se non specificato (o se il dispositivo
        preferito non è disponibile), la funzione seleziona automaticamente il dispositivo migliore.
        
    Returns
    -------
    device : str
        Il dispositivo di calcolo selezionato da utilizzare (ad es. "cuda", "mps" o "cpu").
        
    Note
    -----
    Vengono impostati dei seed fissi per PyTorch, NumPy e random per garantire la riproducibilità degli esperimenti.
    Inoltre, se si utilizza una GPU, viene configurato PyTorch in modalità deterministica.
    """
    
    # Controlla se il dispositivo preferito è disponibile
    if yolo_detection_config.training.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    elif yolo_detection_config.training.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Imposta i seed per garantire la riproducibilità
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(42)
        except Exception:
            pass  # In alcuni ambienti la funzione potrebbe non essere implementata
    np.random.seed(0)
    random.seed(0)
    
    # Configurazioni per rendere deterministico l'addestramento su GPU (se presente)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return device

def select_model(yolo_detection_config):
    """
    Initializes and returns a model based on the configuration settings and the specified number of input features.

    Parameters
    ----------
    config : dict
        Configuration dictionary with parameters for the model.

    Returns
    -------
    model : 
        The initialized model as specified in the configuration.

    Raises
    ------
    SystemExit
        If the specified model name is not supported.

    Notes
    -----
    Currently supports 'yolo11n' model type as specified in the configuration. If an unsupported model
    type is provided, the function will terminate the program.
    """

    model = None

    # Check the model type specified in the configuration and initialize accordingly    
    if yolo_detection_config.training.model_name == 'yolo11n' or yolo_detection_config.training.model_name == 'yolo11s' or yolo_detection_config.training.model_name == 'yolo11m':
        # Initialize model with the provided configuration
        model = YOLO(yolo_detection_config.model_weights_path + f'/{yolo_detection_config.training.model_name}.pt', verbose=True)

    # If no valid model type is specified, print an error message and exit
    if model is None:
        print("Model name is not valid. Check yolo_detection_config.yaml")
        sys.exit(1)
    
    return model

def get_num_workers():
    try:
        thread_count = multiprocessing.cpu_count()  # Conta il numero di thread disponibili
    except Exception as e:
        print(f"Error to determine the number of worker threads: {e}")
    
    return thread_count

def evaluate_model(yolo_detection_config, biometric_trait, model, device):
    """
    Esegue la validazione del modello sul dataset di test definito in data.yaml.
    
    Parametri:
      - model: il modello addestrato
      - data_yaml: file YAML del dataset
    
    Restituisce:
      - results: i risultati della validazione
    """

    results = model.val(
        data=yolo_detection_config.train_yolo_model_path,
        imgsz=yolo_detection_config.data.image_size,                   # Dimensione immagine
        batch=yolo_detection_config.training.batch_size,               # Dimensione batch
        save_json=True,
        conf=0.4,                                                         # Sets the minimum confidence threshold for detections. Detections with confidence below this threshold are discarded.
        iou=0.7,                                                            # Sets the Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Helps in reducing duplicate detections.
        device=device,                                                     # Usa GPU (se disponibile)
        plots=True,
        split='test',
        project=f'{yolo_detection_config.save_path}/{biometric_trait}/yolo/test',
        name=yolo_detection_config.training.model_name,
        exist_ok=True
    )

    return results

def plot_inferences(yolo_detection_config, biometric_trait, model, device):
    """
    Evaluates a model using the given DataLoader and loss criterion.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    dataloader : DataLoader
        DataLoader containing the validation or test dataset.
    criterion : loss function
        The loss function used to compute the model's loss.
    device : torch.device
        The device tensors will be sent to before model evaluation.

    Returns
    -------
    tuple
        Returns a dictionary with validation metrics and the raw predictions and references.
    """

    images_path = os.path.join(yolo_detection_config.save_data_splitted_path, "splitted database", biometric_trait, "test", "images")
    bounding_boxes_path = os.path.join(yolo_detection_config.save_data_splitted_path, "splitted database", biometric_trait, "test", "labels")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Directory {images_path} not found.")
    
    if not os.path.exists(bounding_boxes_path):
        raise FileNotFoundError(f"Directory {bounding_boxes_path} not found.")

    image_files = [f for f in os.listdir(images_path) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.png')]
    bounding_box_files = [f for f in os.listdir(bounding_boxes_path) if f.endswith('.txt')]

    # bounding_box_files.remove("classes.txt")

    # Ordinamento in base al primo e secondo numero nel nome:
    image_files = sorted(
        image_files,
        key=lambda x: (
            int(x.split('_')[0]),                              # primo numero
            int(x.split('_')[1].split('.')[0])               # secondo numero (rimuovo l'estensione)
        )
    )
    bounding_box_files = sorted(
        bounding_box_files,
        key=lambda x: (
            int(x.split('_')[0]),                              # primo numero
            int(x.split('_')[1].split('.')[0])               # secondo numero (rimuovo l'estensione)
        )
    )

    # tqdm is used to show the progress barr
    for image_file, bounding_box_file in tqdm(zip(image_files, bounding_box_files), desc="Plotting inference images", unit=" plot"):
        if image_file.endswith(".bmp") and bounding_box_file.endswith(".txt"):
            # Carica l'immagine
            image_path = os.path.join(images_path, image_file)

            acquisition_name = os.path.basename(os.path.splitext(image_path)[0])

            reference_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

            if reference_image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")

            bounding_box_path = os.path.join(bounding_boxes_path, bounding_box_file)

            height, width, _ = reference_image.shape

            with open(bounding_box_path, "r") as bounding_box_file:
                bounding_box = bounding_box_file.readline().strip().split()

                _, x_center, y_center, box_width, box_height = map(float, bounding_box)

                # Calcola le coordinate delle bounding box in pixel
                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                # Disegna il rettangolo sull'immagine
                cv2.rectangle(reference_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Usa il modello per effettuare inferenze
            # prediction = model(image_file)
            prediction = model.predict(
                source=image_path,
                conf=0.4,
                iou=0.7,
                imgsz=yolo_detection_config.data.image_size,
                device=device,
                retina_masks=True,
                # Visualization params:
                show=False,
                save=False,
                save_crop=False,    # True sul main
                show_labels=True,   # False sul main       
                show_conf=True,     # False sul main
                show_boxes=True,    # False sul main
                line_width=2,       # Non specificare sul main (None o rimuovere)
            )
            
            # Ottieni l'immagine annotata (bounding box, etichette, ecc.)
            prediction_image = prediction[0].plot()

            prediction_image = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2RGB)

            plot_prediction(yolo_detection_config, biometric_trait, reference_image, prediction_image, acquisition_name)