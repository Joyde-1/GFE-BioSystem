# evaluate.py

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import GaitEmbeddingExtractionDataset
from model import GaitSTGCNModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for Gait ST-GCN + Attention + Transformer")
    parser.add_argument("--data", type=str, required=True,
                        help="Percorso al file .npz contenente i dati (con 'keypoints_sequences' e 'labels').")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Percorso al file .pth del modello salvato (checkpoint).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per DataLoader.")
    parser.add_argument("--emb-dim", type=int, default=256, help="Dimensione embedding usata nel modello.")
    parser.add_argument("--transformer-layers", type=int, default=2, help="Numero di layer del Transformer Encoder usato.")
    parser.add_argument("--transformer-heads", type=int, default=4, help="Numero di teste di attenzione del Transformer usato.")
    parser.add_argument("--transformer-ffn-dim", type=int, default=512,
                        help="Dimensione FFN interno al Transformer usato.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout usato nel modello.")
    parser.add_argument("--pretrained-stgcn", type=str, default=None,
                        help="Percorso al checkpoint .pth del backbone ST-GCN pre-addestrato (opzionale).")
    parser.add_argument("--num-workers", type=int, default=4, help="Numero di workers per DataLoader.")
    return parser.parse_args()


def load_data(npz_path):
    """
    Carica un file .npz con due array:
      - 'keypoints_sequences': numpy array di shape (N, T, 51)
      - 'labels': numpy array di shape (N,) con etichette zero-based
    Restituisce un dict { 'keypoints_sequences': ..., 'labels': ... }.
    """
    data = np.load(npz_path)
    keypoints = data["keypoints_sequences"]  # (N, T, 51)
    labels = data["labels"]                  # (N,)
    return {"keypoints_sequences": keypoints, "labels": labels}


def extract_embeddings(model, dataloader, device):
    """
    Data una rete model e un DataLoader, estrae gli embedding per tutti i campioni.
    Restituisce due numpy array:
      - embeddings_all: (N, emb_dim)
      - labels_all:     (N,)
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["keypoints_sequence"].to(device)  # (B, 3, T, 17)
            labels = batch["label"].to(device)          # (B,)

            # Forward pass: otteniamo solo gli embedding (labels=None)
            embeddings = model(x)  # (B, emb_dim)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()  # (N, emb_dim)
    all_labels = torch.cat(all_labels, dim=0).numpy()          # (N,)
    return all_embeddings, all_labels


def split_gallery_probe(embeddings, labels):
    """
    Divide i dati in gallery e probe in modo deterministico:
      - Per ogni classe (etichetta), prende il primo indice trovato come gallery
      - Tutti gli altri campioni di quella classe diventano probe
    Restituisce:
      - gallery_embeddings: list di embedding (num_classes, emb_dim)
      - gallery_labels:     list di etichette (num_classes,)
      - probe_embeddings:   list di embedding (N_probe, emb_dim)
      - probe_labels:       list di etichette (N_probe,)
    """
    num_samples = labels.shape[0]
    emb_dim = embeddings.shape[1]
    # Mappa etichetta -> lista di indici
    label_to_indices = {}
    for idx, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    gallery_idxs = []
    probe_idxs = []

    for lbl, idx_list in label_to_indices.items():
        idx_list_sorted = sorted(idx_list)
        gallery_idxs.append(idx_list_sorted[0])        # primo campione di ogni classe
        probe_idxs.extend(idx_list_sorted[1:])         # tutti gli altri di quella classe

    gallery_embeddings = embeddings[gallery_idxs]      # (num_classes, emb_dim)
    gallery_labels = labels[gallery_idxs]              # (num_classes,)

    probe_embeddings = embeddings[probe_idxs]          # (N_probe, emb_dim)
    probe_labels = labels[probe_idxs]                  # (N_probe,)

    return gallery_embeddings, gallery_labels, probe_embeddings, probe_labels


def compute_rank1_accuracy(gallery_emb, gallery_lbl, probe_emb, probe_lbl):
    """
    Calcola l'accuracy Rank-1 per identificazione:
      - probe_emb: (N_probe, emb_dim)
      - gallery_emb: (num_classes, emb_dim)
    Per ogni probe, trova la gallery più vicina (L2) e compara l'etichetta.
    Restituisce rank1_accuracy (float).
    """
    # Calcoliamo la matrice di distanza L2: (N_probe, num_classes)
    probe_tensor = torch.from_numpy(probe_emb)      # (N_probe, emb_dim)
    gallery_tensor = torch.from_numpy(gallery_emb)  # (num_classes, emb_dim)
    dists = torch.cdist(probe_tensor, gallery_tensor, p=2.0).numpy()  # (N_probe, num_classes)

    pred_idxs = np.argmin(dists, axis=1)            # (N_probe,)
    pred_labels = gallery_lbl[pred_idxs]            # (N_probe,)

    correct = (pred_labels == probe_lbl).sum()
    total = probe_lbl.shape[0]
    return correct / total


def compute_metrics(gallery_emb, gallery_lbl, probe_emb, probe_lbl):
    """
    Calcola:
      - Rank-1 accuracy
      - FAR, FRR, EER
    FAR e FRR basati sulla distribuzione delle distanze:
      - genuine_distances: distanza tra ogni probe e la sua gallery corrispondente
      - imposter_distances: per ogni probe, la minima distanza alle gallery di classi diverse
    """
    probe_tensor = torch.from_numpy(probe_emb)      # (N_probe, emb_dim)
    gallery_tensor = torch.from_numpy(gallery_emb)  # (num_classes, emb_dim)

    # Calcola tutte le distanze: (N_probe, num_classes)
    dist_matrix = torch.cdist(probe_tensor, gallery_tensor, p=2.0).numpy()

    # Genuine distances: d(probe_i, gallery of same label)
    genuine_dists = []
    imposter_dists = []

    for i, lbl in enumerate(probe_lbl):
        # indice nel gallery di questa stessa etichetta
        # poiché gallery has exactly one sample per label:
        gallery_idx = np.where(gallery_lbl == lbl)[0][0]
        genuine_dists.append(dist_matrix[i, gallery_idx])

        # Tra le altre gallery, prendi la distanza minima
        mask = gallery_lbl != lbl
        imposter_dists.append(dist_matrix[i, mask].min())

    genuine_dists = np.array(genuine_dists)      # (N_probe,)
    imposter_dists = np.array(imposter_dists)    # (N_probe,)

    # Calcoliamo FAR e FRR su una griglia di soglie
    all_dists = np.concatenate([genuine_dists, imposter_dists])
    min_d, max_d = all_dists.min(), all_dists.max()
    thresholds = np.linspace(min_d, max_d, num=1000)

    far_values = []  # False Acceptance Rate
    frr_values = []  # False Rejection Rate

    num_genuine = genuine_dists.shape[0]
    num_imposter = imposter_dists.shape[0]

    for th in thresholds:
        # FRR: fraction di genuine > th
        fr = np.sum(genuine_dists > th) / num_genuine
        # FAR: fraction di imposter <= th
        fa = np.sum(imposter_dists <= th) / num_imposter

        frr_values.append(fr)
        far_values.append(fa)

    far_values = np.array(far_values)
    frr_values = np.array(frr_values)

    # Trova EER: punto dove |FAR - FRR| è minimo
    eer_index = np.argmin(np.abs(far_values - frr_values))
    eer = (far_values[eer_index] + frr_values[eer_index]) / 2.0
    eer_threshold = thresholds[eer_index]

    metrics = {
        "genuine_dists": genuine_dists,
        "imposter_dists": imposter_dists,
        "thresholds": thresholds,
        "FAR": far_values,
        "FRR": frr_values,
        "EER": eer,
        "EER_threshold": eer_threshold
    }
    return metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Caricamento dei dati di test (o val)
    print("Caricamento dei dati...")
    data_dict = load_data(args.data)
    dataset = GaitEmbeddingExtractionDataset(data_dict)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 2) Creazione e caricamento del modello
    num_classes = len(np.unique(data_dict["labels"]))
    model = GaitSTGCNModel(
        num_classes=num_classes,
        pretrained_stgcn_path=args.pretrained_stgcn,
        emb_dim=args.emb_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        dropout=args.dropout
    ).to(device)

    print(f"Caricamento checkpoint da {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 3) Estrazione degli embedding per tutti i campioni
    print("Estrazione degli embedding...")
    embeddings_all, labels_all = extract_embeddings(model, dataloader, device)
    print(f"Totale campioni: {labels_all.shape[0]}, embedding dim: {embeddings_all.shape[1]}")

    # 4) Suddivisione in gallery e probe
    gallery_emb, gallery_lbl, probe_emb, probe_lbl = split_gallery_probe(embeddings_all, labels_all)
    print(f"Gallery: {gallery_emb.shape[0]} classi, Probe: {probe_emb.shape[0]} campioni")

    # 5) Calcolo dell'accuracy Rank-1
    rank1 = compute_rank1_accuracy(gallery_emb, gallery_lbl, probe_emb, probe_lbl)
    print(f"Rank-1 Accuracy: {rank1 * 100:.2f}%")

    # 6) Calcolo delle metriche FAR/FRR/EER
    metrics = compute_metrics(gallery_emb, gallery_lbl, probe_emb, probe_lbl)
    eer = metrics["EER"]
    eer_thr = metrics["EER_threshold"]
    print(f"EER: {eer * 100:.2f}% @ threshold {eer_thr:.4f}")

    # (Opzionale) Salvare metriche in file .npz
    base_name = os.path.splitext(os.path.basename(args.data))[0]
    save_path = f"eval_{base_name}.npz"
    np.savez(
        save_path,
        gallery_embeddings=gallery_emb,
        gallery_labels=gallery_lbl,
        probe_embeddings=probe_emb,
        probe_labels=probe_lbl,
        genuine_dists=metrics["genuine_dists"],
        imposter_dists=metrics["imposter_dists"],
        thresholds=metrics["thresholds"],
        FAR=metrics["FAR"],
        FRR=metrics["FRR"],
        EER=np.array([metrics["EER"]]),
        EER_threshold=np.array([metrics["EER_threshold"]])
    )
    print(f"Metriche salvate in: {save_path}")


if __name__ == "__main__":
    main()