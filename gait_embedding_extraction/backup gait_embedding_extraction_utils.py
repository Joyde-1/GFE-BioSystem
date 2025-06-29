import numpy as np
import os
import sys
import random
import torch
import torch.nn.functional as F
import yaml # YAML parsing
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from yaml_config_override import add_arguments # Custom YAML config handling
from typing import List, Tuple
from addict import Dict # Dictionary-like class that allows attribute access
from pathlib import Path  # Object-oriented filesystem paths
from PyQt6.QtWidgets import QApplication, QFileDialog

try:
    from gait_embedding_extraction.model_class.gait_model import GaitModel
    from gait_embedding_extraction.gait_embedding_extraction_plots import plot_tsne
except ModuleNotFoundError:
    from model_class.gait_model import GaitModel
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
    if gait_embedding_extraction_config.training.model_name == 'gait_model':
        model = GaitModel(gait_embedding_extraction_config)
        model.to(device)

    # If no valid model type is specified, print an error message and exit
    if model is None:
        print("Model name is not valid. Check gait_embedding_extraction_config.yaml")
        sys.exit(1)
    
    return model

def create_triplets(labels, num_triplets=None):
    """
    Crea triplet (anchor, positive, negative) per Triplet Loss
    """
    if num_triplets is None:
        num_triplets = len(labels) // 2

    triplets = []
    labels_np = labels.detach().cpu().numpy()

    for _ in range(num_triplets):
        # Seleziona anchor
        anchor_idx = random.randint(0, len(labels) - 1)
        anchor_label = labels_np[anchor_idx]

        # Trova positive (stessa classe)
        positive_candidates = np.where(labels_np == anchor_label)[0]
        positive_candidates = positive_candidates[positive_candidates != anchor_idx]

        if len(positive_candidates) == 0:
            continue

        positive_idx = random.choice(positive_candidates)

        # Trova negative (classe diversa)
        negative_candidates = np.where(labels_np != anchor_label)[0]
        if len(negative_candidates) == 0:
            continue

        negative_idx = random.choice(negative_candidates)

        triplets.append((anchor_idx, positive_idx, negative_idx))

    return triplets

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

def compute_intra_inter_distances(embeddings, labels, distance_matrix):
    """
    Calcola le distanze medie intra-classe e inter-classe.
    Input:
      - embeddings: array (N, D)
      - labels:     array (N,)
    Output: (avg_intra_dist, avg_inter_dist)
    """
    N = embeddings.shape[0]

    intra_dists = []
    inter_dists = []

    for i in range(N):
        for j in range(i + 1, N):
            if labels[i] == labels[j]:
                intra_dists.append(distance_matrix[i, j])
            else:
                inter_dists.append(distance_matrix[i, j])

    if len(intra_dists) > 0:
        avg_intra = float(np.mean(intra_dists))
    else:
        avg_intra = 0.0

    if len(inter_dists) > 0:
        avg_inter = float(np.mean(inter_dists))
    else:
        avg_inter = 0.0

    return avg_intra, avg_inter

def compute_eer(y_true, y_scores):
    """
    Compute Equal Error Rate (EER) given true binary labels and scores.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # find threshold where FPR ≈ FNR
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2

    threshold = thresholds[idx]
    far = fpr[idx]
    frr = fnr[idx]

    return eer, threshold, far, frr

def compute_cmc(labels, distance_matrix, topk=(1,5)):
    """
    Compute Cumulative Matching Characteristic (CMC) at specified ranks.
    Returns a dict mapping rank k to accuracy.
    """

    N = labels.shape[0]

    ranks = []
    for i in range(N):
        # sort other samples by ascending distance
        idx_sorted = np.argsort(distance_matrix[i])
        # exclude self
        idx_sorted = idx_sorted[idx_sorted != i]
        # find first correct match position
        correct = np.where(labels[idx_sorted] == labels[i])[0]
        rank = correct[0] + 1 if correct.size > 0 else np.inf
        ranks.append(rank)
    cmc = {}
    for k in topk:
        cmc[k] = np.mean([1.0 if r <= k else 0.0 for r in ranks])
    return cmc

def compute_tsne(gait_embedding_extraction_config, all_embeddings, all_labels, epoch):
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    embs_2d = tsne.fit_transform(all_embeddings)

    plot_tsne(gait_embedding_extraction_config, embs_2d, all_labels, epoch)

# def compute_metrics(all_embeddings, all_labels):
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

#     # # If embeddings/labels were collected per batch (as lists of arrays), concatenate them
#     # if isinstance(all_embeddings, list):
#     #     all_embeddings = np.concatenate(all_embeddings, axis=0)
#     #     all_labels     = np.concatenate(all_labels,     axis=0)

#     # Pre-calcoliamo la matrice di distanza euclidea
#     distance_matrix = compute_distance_matrix(all_embeddings, metric="cosine")

#     # Calcola le distanze intra e inter-classe
#     avg_intra_dist, avg_inter_dist = compute_intra_inter_distances(all_embeddings, all_labels, distance_matrix)
#     inter_intra_ratio = avg_inter_dist / (avg_intra_dist + 1e-9)

#     # Verification metrics (ROC AUC, EER)
#     # build pairwise labels/scores
#     y_true, y_scores = [], []

#     N = all_labels.shape[0]
#     for i in range(N):
#         for j in range(i+1, N):
#             y_true.append(1 if all_labels[i] == all_labels[j] else 0)
#             y_scores.append(-distance_matrix[i, j])  # higher score = more likely same
#     y_true = np.array(y_true)
#     y_scores = np.array(y_scores)
#     roc_auc = roc_auc_score(y_true, y_scores)
#     eer, threshold, far, frr = compute_eer(y_true, y_scores)

#     # Identification metrics (CMC)
#     cmc = compute_cmc(all_labels, distance_matrix, topk=(1, 5))
#     rank1 = cmc[1]
#     rank5 = cmc[5]

#     # Silhouette score (cosine)
#     try:
#         silhouette_scr = silhouette_score(all_embeddings, all_labels, metric='cosine')
#     except Exception:
#         silhouette_scr = float('nan')

#     davies_bouldin_scr = davies_bouldin_score(all_embeddings, all_labels)
#     calinski_harabasz_scr = calinski_harabasz_score(all_embeddings, all_labels)
    
#     return {
#         'avg_intra_dist': avg_intra_dist,
#         'avg_inter_dist': avg_inter_dist,
#         'inter_intra_ratio': inter_intra_ratio,
#         'eer': eer,
#         'threshold': threshold,
#         'far': far,
#         'frr': frr,
#         'rank1': rank1,
#         'rank5': rank5,
#         'roc_auc': roc_auc,
#         'silhouette_score': silhouette_scr,
#         'davies_bouldin_score': davies_bouldin_scr,
#         'calinski_harabasz_score': calinski_harabasz_scr
#     }

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
    correct = 0
    total = 0

    embeddings_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    # # List to store embeddings
    # all_embeddings = []
    
    # # List to store labels
    # all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            keypoints_sequences = batch['keypoints_sequence'].to(device)
            labels = batch['label'].to(device)  # (B,)
            
            # Forward pass
            embeddings, logits, center_loss = model(keypoints_sequences, labels)

            ce_loss = model.ce_loss(logits, labels)  # CrossEntropy Loss

            loss = ce_loss + center_loss  # Total loss

            # # Compute loss (solo CrossEntropy per validazione)
            # loss, _, _ = criterion(embeddings, logits, labels)

            # Update running loss
            running_loss += loss.item()

            embeddings_list.append(F.normalize(embeddings, dim=1).cpu())
            labels_list.append(labels.cpu())

            pred = logits.argmax(dim=1)  # Predicted labels
            total += labels.size(0)  # Total samples
            correct += (pred == labels).sum().item()  # Count correct predictions

            # _, predicted = torch.max(logits.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

            # all_embeddings.append(embeddings.cpu())
            # all_labels.append(labels.cpu())
            
    # After collecting all embeddings and labels:
    # all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    # all_labels = torch.cat(all_labels, dim=0).numpy()
    embeddings_list = torch.cat(embeddings_list, dim=0)
    labels_list = torch.cat(labels_list, dim=0)

    # val_metrics = compute_metrics(all_embeddings, all_labels)
    val_metrics = compute_metrics(embeddings_list, labels_list)
    val_metrics['loss'] = running_loss / len(dataloader)
    val_metrics['accuracy'] = 100.0 * correct / total

    # TSNE visualization
    # try:
    #     compute_tsne(gait_embedding_extraction_config, embeddings_list.numpy(), labels_list.numpy(), epoch)
    # except Exception as e:
    #     print(f"TSNE plotting failed: {e}")

    # Loggare le metriche di validazione su Comet.ml
    experiment.log_metric("val_loss", val_metrics['loss'], epoch=epoch)
    experiment.log_metric("val_accuracy", val_metrics['accuracy'], epoch=epoch)
    experiment.log_metric("val_EER", val_metrics['EER'], epoch=epoch)
    experiment.log_metric("val_FAR@1%", val_metrics['FAR@1%'], epoch=epoch)
    experiment.log_metric("val_FRR@1%", val_metrics['FRR@1%'], epoch=epoch)
    experiment.log_metric("val_threshold", val_metrics['threshold'], epoch=epoch)
    # experiment.log_metric("val_avg_intra_dist", val_metrics['avg_intra_dist'], epoch=epoch)
    # experiment.log_metric("val_avg_inter_dist", val_metrics['avg_inter_dist'], epoch=epoch)
    # experiment.log_metric("val_inter_intra_ratio", val_metrics['inter_intra_ratio'], epoch=epoch)
    # experiment.log_metric("val_EER", val_metrics['EER'], epoch=epoch)
    # experiment.log_metric("val_FAR", val_metrics['FAR'], epoch=epoch)
    # experiment.log_metric("val_FRR", val_metrics['FRR'], epoch=epoch)
    # experiment.log_metric("val_threshold", val_metrics['threshold'], epoch=epoch)
    # experiment.log_metric("val_rank1", val_metrics['rank1'], epoch=epoch)
    # experiment.log_metric("val_rank5", val_metrics['rank5'], epoch=epoch)
    # experiment.log_metric("val_ROC_AUC", val_metrics['ROC_AUC'], epoch=epoch)
    # experiment.log_metric("val_silhouette_score", val_metrics['silhouette_score'], epoch=epoch)
    # experiment.log_metric("val_davies_bouldin_score", val_metrics['davies_bouldin_score'], epoch=epoch)
    # experiment.log_metric("val_calinski_harabasz_score", val_metrics['calinski_harabasz_score'], epoch=epoch)

    return val_metrics