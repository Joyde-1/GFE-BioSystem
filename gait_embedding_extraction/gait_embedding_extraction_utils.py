import numpy as np
import os
import sys
import random
import torch
import torch.nn.functional as F
import yaml # YAML parsing
from yaml_config_override import add_arguments # Custom YAML config handling
from typing import List, Tuple
from addict import Dict # Dictionary-like class that allows attribute access
from pathlib import Path  # Object-oriented filesystem paths
from PyQt6.QtWidgets import QApplication, QFileDialog

try:
    from gait_embedding_extraction.model_class.gait_model import GaitModel
except ModuleNotFoundError:
    from model_class.gait_model import GaitModel


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

def create_dataloader(gait_embedding_extraction_config, dataset, shuffle):
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
        batch_size=gait_embedding_extraction_config.training.batch_size,
        num_workers = gait_embedding_extraction_config.training.num_workers,
        shuffle=shuffle
    )

    return dataloader

def select_device(gait_embedding_extraction_config):
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
    if gait_embedding_extraction_config.training.device == "mps" and torch.backends.mps.is_available():
        device = torch.device('mps')  # Use MPS
    # Otherwise, check if CUDA (NVIDIA GPU) is available and select it if possible
    elif gait_embedding_extraction_config.training.device == "cuda" and torch.cuda.is_available():
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
    np.random.seed(42)
    random.seed(42)
    
    # Configurazioni per rendere deterministico l'addestramento su GPU (se presente)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return device

def select_model(gait_embedding_extraction_config, device):
    """
    Initializes and returns a model based on the configuration settings.

    Parameters
    ----------
    config : dict
        Configuration dictionary with parameters for the model.

    Returns
    -------
    model : torch.nn.Module
        The initialized model as specified in the configuration.

    Raises
    ------
    SystemExit
        If the specified model name is not supported.
    """

    model = None

    # Check the model type specified in the configuration and initialize accordingly    
    if gait_embedding_extraction_config.training.model_name == 'gait_model':
        model = GaitModel(gait_embedding_extraction_config)
        model.to(device)

    # If no valid model type is specified, print an error message and exit
    if model is None:
        print("Model name is not valid. Check gait_embedding_extraction_config.yaml")
        sys.exit(1)
    
    return model

def _pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, dim=1)
    return x @ x.t()

def _compute_far_frr(embeddings: torch.Tensor, labels: torch.Tensor, far_target: float = 0.01) -> Tuple[float, float, float, float]:
    """Compute EER and FAR / FRR at a given FAR target.

    Parameters
    ----------
    embeddings : (N, C) tensor (need not be normalised)
    labels     : (N,)   tensor of ints
    far_target : float  target FAR (e.g. 0.01 → 1 %)

    Returns
    -------
    eer      : float
    far_at   : float  FAR at the threshold that first drops below `far_target`
    frr_at   : float  FRR at the *same* threshold
    thr      : float  similarity threshold used
    """
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    sim = _pairwise_cosine(embeddings)

    # Upper‑triangular masks (exclude self‑pairs)
    N = sim.size(0)
    triu = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    same = labels.view(-1, 1).eq(labels.view(1, -1))
    genuine = sim[same & triu]
    imposter = sim[(~same) & triu]

    scores = torch.cat([genuine, imposter])
    y_true = torch.cat([torch.ones_like(genuine), torch.zeros_like(imposter)])

    scores_sorted, idx = torch.sort(scores, descending=True)
    y_sorted = y_true[idx]

    cum_true = torch.cumsum(y_sorted, 0)
    cum_false = torch.cumsum(1 - y_sorted, 0)
    P = genuine.numel()
    N_neg = imposter.numel()

    frr = (P - cum_true).float() / P  # False Rejection
    far = cum_false.float() / N_neg   # False Acceptance

    # Equal Error Rate
    diff = torch.abs(far - frr)
    eer_idx = torch.argmin(diff)
    eer = ((far[eer_idx] + frr[eer_idx]) / 2).item()

    # FAR @ target
    try:
        thr_idx = (far <= far_target).nonzero(as_tuple=True)[0][0]
    except IndexError:
        thr_idx = -1  # take lowest threshold
    far_at = far[thr_idx].item()
    frr_at = frr[thr_idx].item()
    thr = scores_sorted[thr_idx].item()

    return eer, far_at, frr_at, thr

def compute_metrics(embeddings, labels):
    """
    Computes various metrics based on embeddings and labels.
    """

    eer, far_at, frr_at, thr = _compute_far_frr(embeddings, labels, far_target=0.01)

    return {
        'EER': eer,
        'FAR@1%': far_at,
        'FRR@1%': frr_at,
        'threshold': thr
    }

def evaluate(gait_embedding_extraction_config, model, dataloader, device, epoch, experiment):
    """
    Evaluates a model using the given DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    dataloader : DataLoader
        DataLoader containing the validation or test dataset.
    device : torch.device
        The device tensors will be sent to before model evaluation.

    Returns
    -------
    tuple
        Returns a dictionary with validation metrics and the raw predictions and references.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    embeddings_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    
    with torch.no_grad():
        for batch in dataloader:
            keypoints_sequences = batch['keypoints_sequence'].to(device)
            labels = batch['label'].to(device)  # (B,)
            
            # Forward pass
            embeddings, logits, center_loss = model(keypoints_sequences, labels)

            ce_loss = model.ce_loss(logits, labels)  # CrossEntropy Loss

            loss = ce_loss + center_loss  # Total loss

            # Update running loss
            running_loss += loss.item()

            embeddings_list.append(F.normalize(embeddings, dim=1).cpu())
            labels_list.append(labels.cpu())

            pred = logits.argmax(dim=1)  # Predicted labels
            total += labels.size(0)  # Total samples
            correct += (pred == labels).sum().item()  # Count correct predictions
            
    # After collecting all embeddings and labels:
    embeddings_list = torch.cat(embeddings_list, dim=0)
    labels_list = torch.cat(labels_list, dim=0)

    val_metrics = compute_metrics(embeddings_list, labels_list)
    val_metrics['loss'] = running_loss / len(dataloader)
    val_metrics['accuracy'] = 100.0 * correct / total

    # Loggare le metriche di validazione su Comet.ml
    experiment.log_metric("val_loss", val_metrics['loss'], epoch=epoch)
    experiment.log_metric("val_accuracy", val_metrics['accuracy'], epoch=epoch)
    experiment.log_metric("val_EER", val_metrics['EER'], epoch=epoch)
    experiment.log_metric("val_FAR@1%", val_metrics['FAR@1%'], epoch=epoch)
    experiment.log_metric("val_FRR@1%", val_metrics['FRR@1%'], epoch=epoch)
    experiment.log_metric("val_threshold", val_metrics['threshold'], epoch=epoch)

    return val_metrics