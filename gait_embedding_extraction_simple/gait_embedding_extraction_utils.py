import torch
import os
import sys
import random
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
from yaml_config_override import add_arguments # Custom YAML config handling
from addict import Dict # Dictionary-like class that allows attribute access
import yaml # YAML parsing
from pathlib import Path  # Object-oriented filesystem paths
from PyQt6.QtWidgets import QApplication, QFileDialog

try:
    from gait_embedding_extraction_simple.model_class.gait_stgcn_trans import GaitSTGCNModel
    from gait_embedding_extraction_simple.gait_embedding_extraction_plots import plot_tsne
except ModuleNotFoundError:
    from model_class.gait_stgcn_trans import GaitSTGCNModel
    from gait_embedding_extraction_plots import plot_tsne


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
    if gait_embedding_extraction_config.training.model_name == 'stgcn_trans':
        model = GaitSTGCNModel(
            num_classes=gait_embedding_extraction_config.data.num_classes,
            pretrained_stgcn_path=f"{gait_embedding_extraction_config.training.checkpoints_dir}/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth",
            device=device,
            emb_dim=gait_embedding_extraction_config.model.embedding_dim,
            transformer_layers=gait_embedding_extraction_config.model.num_transformer_layers,
            transformer_heads=gait_embedding_extraction_config.model.num_transformer_heads,
            transformer_ffn_dim=gait_embedding_extraction_config.model.transformer_ffn_dim,
            dropout=gait_embedding_extraction_config.model.dropout
        )

        # Freeze parziale dei primi blocchi ST-GCN
        for param in model.stgcn1.parameters():
            param.requires_grad = False
        for param in model.stgcn2.parameters():
            param.requires_grad = False

    # If no valid model type is specified, print an error message and exit
    if model is None:
        print("Model name is not valid. Check gait_embedding_extraction_config.yaml")
        sys.exit(1)
    
    return model

# def compute_metrics(predictions, references):
#     """
#     Computes regression metrics: MAE and RMSE.

#     Parameters
#     ----------
#     predictions : list or array-like
#         Predicted ear landmarks list by the model.
#     references : list or array-like
#         Actual ear landmarks list from the dataset.

#     Returns
#     -------
#     dict
#         Dictionary containing computed metrics: MAE and RMSE.
#     """
#     predictions = np.array(predictions)
#     references = np.array(references)

#     # mae = mean_absolute_error(references, predictions)
#     # mse = mean_squared_error(references, predictions)
#     # rmse = root_mean_squared_error(references, predictions)
#     # r2 = r2_score(references, predictions)
#     # mape = mean_absolute_percentage_error(references, predictions)

#     accuracy = accuracy_score(references, predictions)
#     f1 = f1_score(references, predictions, average='macro')
#     precision = precision_score(references, predictions, average='macro')
#     recall = recall_score(references, predictions, average='macro')

#     return {
#         'accuracy': accuracy,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }
    
#     # return {
#     #     'mae': mae,
#     #     'mse': mse,
#     #     'rmse': rmse,
#     #     'r2': r2,
#     #     'mape': mape
#     # }

def compute_distance_matrix(embeddings: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Calcola la matrice di distanza tra tutti gli embeddings.
    Input:
      - embeddings: array di shape (N, D)
      - metric: "euclidean" (default) o "cosine"
    Output:
      - dist_matrix: array (N, N) delle distanze
    """
    if metric not in ("euclidean", "cosine"):
        raise ValueError("Metric must be 'euclidean' or 'cosine'")

    if metric == "euclidean":
        # torch.cdist gestisce efficacemente l'operazione se convertiamo a tensor
        emb_tensor = torch.from_numpy(embeddings)
        dists = torch.cdist(emb_tensor, emb_tensor, p=2.0)
        return dists.numpy()

    # cosine distance = 1 - cosine similarity
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    sim_matrix = np.dot(emb_norm, emb_norm.T)
    cos_dist = 1.0 - sim_matrix
    return cos_dist


def compute_intra_inter_distances(embeddings: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Calcola le distanze medie intra-classe e inter-classe.
    Input:
      - embeddings: array (N, D)
      - labels:     array (N,)
    Output: (avg_intra_dist, avg_inter_dist)
    """
    N = embeddings.shape[0]
    # Pre-calcoliamo la matrice di distanza euclidea
    dist_matrix = compute_distance_matrix(embeddings, metric="euclidean")

    intra_dists = []
    inter_dists = []

    for i in range(N):
        for j in range(i + 1, N):
            if labels[i] == labels[j]:
                intra_dists.append(dist_matrix[i, j])
            else:
                inter_dists.append(dist_matrix[i, j])

    if len(intra_dists) > 0:
        avg_intra = float(np.mean(intra_dists))
    else:
        avg_intra = 0.0

    if len(inter_dists) > 0:
        avg_inter = float(np.mean(inter_dists))
    else:
        avg_inter = 0.0

    return avg_intra, avg_inter

def compute_tsne(gait_embedding_extraction_config, all_embeddings, all_labels, epoch):
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    embs_2d = tsne.fit_transform(all_embeddings)

    plot_tsne(gait_embedding_extraction_config, embs_2d, all_labels, epoch)

def evaluate(gait_embedding_extraction_config, model, dataloader, arcface_criterion, triplet_criterion, device, epoch):
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
    running_accuracy = 0.0
    num_samples = 0

    # List to store embeddings
    all_ambeddings = []
    
    # List to store labels
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            keypoints_sequences = batch['keypoints_sequence'].to(device)
            labels = batch['label'].to(device)
            
            # Forward: otteniamo embedding e logit
            embeddings, logits = model(keypoints_sequences, labels)  # embeddings: (B, emb_dim), logits: (B, num_classes)

            # ArcFace loss
            loss_arcface = arcface_criterion(embeddings, labels, model.classifier.weight)

            # Triplet loss
            loss_triplet = triplet_criterion(embeddings, labels)

            loss = gait_embedding_extraction_config.loss_function.triplet.lambda_triplet * loss_triplet + (1 - gait_embedding_extraction_config.loss_function.triplet.lambda_triplet) * loss_arcface

            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            accuracy = correct / labels.size(0)

            # Update running loss
            B = labels.size(0)
            running_loss += loss.item() * B
            running_accuracy += accuracy * B
            num_samples += B

            all_ambeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            
    val_metrics = {}
    val_metrics['loss'] = running_loss / num_samples
    val_metrics['accuracy'] = running_accuracy / num_samples

    all_ambeddings = torch.cat(all_ambeddings, dim=0).numpy()  # (N, emb_dim)
    all_labels = torch.cat(all_labels, dim=0).numpy()  # (N,)

    # Calcola le distanze intra e inter-classe
    avg_intra_dist, avg_inter_dist = compute_intra_inter_distances(all_ambeddings, all_labels)

    val_metrics['avg_intra_dist'] = avg_intra_dist
    val_metrics['avg_inter_dist'] = avg_inter_dist

    compute_tsne(gait_embedding_extraction_config, all_ambeddings, all_labels, epoch)

    return val_metrics