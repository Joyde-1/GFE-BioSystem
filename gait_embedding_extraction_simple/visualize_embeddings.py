import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_embeddings(embeddings_path: str, method: str = 'tsne', perplexity: int = 30, n_components: int = 2):
    """
    Carica embeddings e labels da un file .npz e visualizza una proiezione in 2D.
    
    Args:
        embeddings_path (str): percorso al file .npz contenente
            - 'embeddings': array (N, D)
            - 'labels': array (N,)
        method (str): 'tsne' o 'pca'
        perplexity (int): parametro per TSNE (se utilizzato)
        n_components (int): dimensione finale (2 per visualizzazione 2D)
    """
    data = np.load(embeddings_path)
    embeddings = data['embeddings']  # shape (N, D)
    labels = data['labels']          # shape (N,)

    # Riduzione dimensionale
    if method == 'pca':
        projector = PCA(n_components=n_components)
        reduced = projector.fit_transform(embeddings)
    else:
        # TSNE
        projector = TSNE(n_components=n_components, perplexity=perplexity, init='pca', random_state=42)
        reduced = projector.fit_transform(embeddings)

    # Plot 2D
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=10)
    plt.title(f"Embedding Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    # Facciamo legenda per le classi
    num_classes = len(np.unique(labels))
    handles = []
    for cls in range(num_classes):
        handles.append(plt.Line2D([], [], marker='o', color=scatter.cmap(scatter.norm(cls)),
                                  linestyle='None', markersize=5, label=str(cls)))
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize embeddings with t-SNE or PCA.")
    parser.add_argument("--embeddings-file", type=str, required=True,
                        help="Percorso al file .npz contenente 'embeddings' e 'labels'.")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "pca"],
                        help="Metodo di riduzione dimensionale: 'tsne' o 'pca'.")
    parser.add_argument("--perplexity", type=int, default=30, help="Perplexity per TSNE.")
    args = parser.parse_args()

    visualize_embeddings(
        embeddings_path=args.embeddings_file,
        method=args.method,
        perplexity=args.perplexity
    )