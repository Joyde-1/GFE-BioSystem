import torch
import os
import sys
import random
import cv2
import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
from yaml_config_override import add_arguments # Custom YAML config handling
from addict import Dict # Dictionary-like class that allows attribute access
import yaml # YAML parsing
from pathlib import Path  # Object-oriented filesystem paths
from tqdm import tqdm
from PyQt6.QtWidgets import QApplication, QFileDialog

try:
    from ear_landmarks_detection.model_class.cnn import MobileNetV2EarLandmarks, ResNet50EarLandmarks, ResNet18EarLandmarks
    from ear_landmarks_detection.ear_landmarks_detection_plots import plot_prediction, plot_reference_vs_prediction
except ModuleNotFoundError:
    from model_class.cnn import MobileNetV2EarLandmarks, ResNet50EarLandmarks, ResNet18EarLandmarks
    from ear_landmarks_detection_plots import plot_prediction, plot_reference_vs_prediction


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

def create_dataloader(ear_landmarks_detection_config, dataset, shuffle):
    """
    Creates a DataLoader for batch processing of data.

    Parameters
    ----------
    dataset : Dataset
        The dataset to load into the DataLoader.
    config : dict
        Configuration object containing DataLoader parameters such as batch size and number of workers.
    shuffle : bool
        Whether to shuffle the data at every epoch.

    Returns
    -------
    DataLoader
        A configured DataLoader ready for iteration.
    """

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ear_landmarks_detection_config.training.batch_size,
        num_workers = ear_landmarks_detection_config.training.num_workers,
        shuffle=shuffle
    )

    return dataloader

def select_device(ear_landmarks_detection_config):
    """
    Selects the appropriate computation device (MPS, CUDA, or CPU) based on availability and configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary specifying the preferred device.

    Returns
    -------
    device : torch.device
        The selected computation device.

    Notes
    -----
    The device is selected in the order of preference: MPS, CUDA, CPU. If the preferred device
    is not available, the function defaults to the next available option.
    """

    # Check if MPS (Apple Silicon) is available and select it if possible
    if ear_landmarks_detection_config.training.device == "mps" and torch.backends.mps.is_available():
        device = torch.device('mps')  # Use MPS
    # Otherwise, check if CUDA (NVIDIA GPU) is available and select it if possible
    elif ear_landmarks_detection_config.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device('cuda') # Use CUDA
    # If neither MPS nor CUDA are available, default to CPU
    else:
        device = torch.device('cpu') # Use CPU

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

def select_model(ear_landmarks_detection_config):
    """
    Initializes and returns a model based on the configuration settings and the specified number of input features.

    Parameters
    ----------
    config : dict
        Configuration dictionary with parameters for the model.
    n_features : int
        Number of input features the model should handle.

    Returns
    -------
    model : torch.nn.Module
        The initialized model as specified in the configuration.

    Raises
    ------
    SystemExit
        If the specified model name is not supported.

    Notes
    -----
    Currently supports 'cnn' model type as specified in the configuration. If an unsupported model
    type is provided, the function will terminate the program.
    """

    model = None

    # Check the model type specified in the configuration and initialize accordingly    
    if ear_landmarks_detection_config.training.model_name == 'mobilenetv2':
        model = MobileNetV2EarLandmarks()
    if ear_landmarks_detection_config.training.model_name == 'resnet18':
        model = ResNet18EarLandmarks()
    if ear_landmarks_detection_config.training.model_name == 'resnet50':
        model = ResNet50EarLandmarks()

    # If no valid model type is specified, print an error message and exit
    if model is None:
        print("Model name is not valid. Check ear_landmarks_detection_config.yaml")
        sys.exit(1)
    
    return model

def compute_metrics(predictions, references):
    """
    Computes regression metrics: MAE and RMSE.

    Parameters
    ----------
    predictions : list or array-like
        Predicted ear landmarks list by the model.
    references : list or array-like
        Actual ear landmarks list from the dataset.

    Returns
    -------
    dict
        Dictionary containing computed metrics: MAE and RMSE.
    """
    predictions = np.array(predictions)
    references = np.array(references)

    mae = mean_absolute_error(references, predictions)
    mse = mean_squared_error(references, predictions)
    rmse = root_mean_squared_error(references, predictions)
    r2 = r2_score(references, predictions)
    mape = mean_absolute_percentage_error(references, predictions)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

def evaluate(model, dataloader, criterion, device, return_test_images=False):
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

    model.eval()

    running_loss = 0.0

    predictions = []
    references = []
    test_images = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            landmarks = batch['landmarks'].to(device)
            
            outputs = model(images)

            loss = criterion(outputs, landmarks)

            running_loss += loss.item()

            predictions.extend(outputs.cpu().numpy())
            
            references.extend(landmarks.cpu().numpy())

            if return_test_images:
                test_images.extend(images.cpu().numpy())
            
    val_metrics = compute_metrics(predictions, references)
    val_metrics['loss'] = running_loss / len(dataloader)

    if return_test_images:
        return val_metrics, test_images, references, predictions
    else:
        return val_metrics

def plot_inferences(ear_landmarks_detection_config, test_images, reference_landmarks_list, prediction_landmarks_list, mean, std):
    """
    Plotta immagini, bounding box di riferimento e bounding box predette.

    Parameters
    ----------
    test_images : list of torch.Tensor
        Lista di immagini di test in formato tensor (C, H, W).
    reference_bounding_boxes : list of torch.Tensor
        Lista delle bounding box di riferimento, normalizzate in [0,1].
    prediction_bounding_boxes : list of torch.Tensor
        Lista delle bounding box predette, normalizzate in [0,1].
    mean : list
        Media usata per normalizzare l'immagine.
    std : list
        Deviazione standard usata per normalizzare l'immagine.
    """

    for counter, (test_image, reference_landmarks, prediction_landmarks) in tqdm(enumerate(zip(test_images, reference_landmarks_list, prediction_landmarks_list)), desc="Plotting inference images", unit=" plot"):
        # Assicurati che siano numpy array
        if torch.is_tensor(test_image):
            test_image = test_image.cpu().numpy()
        if torch.is_tensor(reference_landmarks):
            reference_landmarks = reference_landmarks.cpu().numpy()
        if torch.is_tensor(prediction_landmarks):
            prediction_landmarks = prediction_landmarks.cpu().numpy()

        # Passaggio da [C, H, W] a [H, W, C]
        test_image = np.transpose(test_image, (1, 2, 0))  # Canali da prima a ultima dimensione

        # Convertire da [0,1] a [0,255]
        test_image = (test_image * 255).astype(np.uint8)

        # Convertire da RGB a BGR per OpenCV
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
        
        reference_landmarks = np.squeeze(reference_landmarks)  # Da (1, H, W) a (H, W)
        prediction_landmarks = np.squeeze(prediction_landmarks)  # Da (1, H, W) a (H, W)

        reference_image = test_image.copy()
        prediction_image = test_image.copy()

        height, width, _ = test_image.shape

        x_top_ref, y_top_ref, x_bottom_ref, y_bottom_ref, x_outer_ref, y_outer_ref, x_inner_ref, y_inner_ref = map(float, reference_landmarks)

        # Calcola le coordinate dei reference landmarks in pixel
        x_top_ref *= width
        y_top_ref *= height
        x_bottom_ref *= width
        y_bottom_ref *= height
        x_outer_ref *= width
        y_outer_ref *= height
        x_inner_ref *= width
        y_inner_ref *= height

        # x1_ref = int(x_center_ref - box_width_ref / 2)
        # y1_ref = int(y_center_ref - box_height_ref / 2)
        # x2_ref = int(x_center_ref + box_width_ref / 2)
        # y2_ref = int(y_center_ref + box_height_ref / 2)

        # Disegna i landmarks reference sull'immagine
        cv2.circle(test_image, (int(x_top_ref), int(y_top_ref)), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(test_image, (int(x_bottom_ref), int(y_bottom_ref)), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(test_image, (int(x_outer_ref), int(y_outer_ref)), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(test_image, (int(x_inner_ref), int(y_inner_ref)), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(reference_image, (int(x_top_ref), int(y_top_ref)), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(reference_image, (int(x_bottom_ref), int(y_bottom_ref)), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(reference_image, (int(x_outer_ref), int(y_outer_ref)), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(reference_image, (int(x_inner_ref), int(y_inner_ref)), radius=3, color=(0, 255, 0), thickness=-1)

        x_top_pred, y_top_pred, x_bottom_pred, y_bottom_pred, x_outer_pred, y_outer_pred, x_inner_pred, y_inner_pred = map(float, prediction_landmarks)

        # Calcola le coordinate dei predicted landmarks in pixel
        x_top_pred *= width
        y_top_pred *= height
        x_bottom_pred *= width
        y_bottom_pred *= height
        x_outer_pred *= width
        y_outer_pred *= height
        x_inner_pred *= width
        y_inner_pred *= height

        # x1_pred = int(x_center_pred - box_width_pred / 2)
        # y1_pred = int(y_center_pred - box_height_pred / 2)
        # x2_pred = int(x_center_pred + box_width_pred / 2)
        # y2_pred = int(y_center_pred + box_height_pred / 2)

        # Disegna i landmarks predetti sull'immagine
        cv2.circle(test_image, (int(x_top_pred), int(y_top_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(test_image, (int(x_bottom_pred), int(y_bottom_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(test_image, (int(x_outer_pred), int(y_outer_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(test_image, (int(x_inner_pred), int(y_inner_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(prediction_image, (int(x_top_pred), int(y_top_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(prediction_image, (int(x_bottom_pred), int(y_bottom_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(prediction_image, (int(x_outer_pred), int(y_outer_pred)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(prediction_image, (int(x_inner_pred), int(y_inner_pred)), radius=3, color=(0, 0, 255), thickness=-1)

        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        prediction_image = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2RGB)

        plot_prediction(ear_landmarks_detection_config, reference_image, prediction_image, counter)

        plot_reference_vs_prediction(ear_landmarks_detection_config, test_image, counter)