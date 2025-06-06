import os
import torch
import numpy as np


def save_checkpoint(state: dict, filename: str):
    """
    Salva lo stato del training nel file specificato.
    `state` dovrebbe contenere almeno:
      - 'epoch': numero di epoca
      - 'model_state_dict': model.state_dict()
      - 'optimizer_state_dict': optimizer.state_dict()
      - eventuali altri campi (val_loss, val_acc, ecc.)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    filename: str = None,
                    device: torch.device = torch.device("cpu")):
    """
    Carica checkpoint salvato in `filename` e ripristina i pesi del modello.
    Se `optimizer` non è None e il checkpoint contiene 'optimizer_state_dict',
    ripristina anche lo stato dell'ottimizzatore.
    Ritorna il numero di epoca in cui il checkpoint è stato salvato (se presente),
    oppure None.
    """
    if filename is None or not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint non trovato: {filename}")

    checkpoint = torch.load(filename, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    start_epoch = checkpoint.get("epoch", None)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return start_epoch


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


def make_dir_if_not_exists(path: str):
    """
    Crea la cartella `path` se non esiste.
    """
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    """
    Imposta il seed globale per torch e numpy per riproducibilità.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False