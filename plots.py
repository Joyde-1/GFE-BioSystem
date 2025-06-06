import matplotlib.pyplot as plt
import platform
import subprocess

from utils import save_image


def _open_image(image_path):
    """
    Opens an image file using the default system viewer based on the operating system.

    This function attempts to open an image file with the native image viewer of the user's operating system.
    It handles different commands for Windows, macOS, and Linux.

    Parameters
    ----------
    image_path : str
        The full path to the image file that needs to be opened.

    Raises
    ------
    subprocess.CalledProcessError
        If the subprocess responsible for opening the image fails, this error will be raised.

    Notes
    -----
    Ensure that the image path is correct and accessible. The function may not provide detailed
    debug information if the path is incorrect.
    """
    
    try:
        os_name = platform.system()
        if os_name == 'Windows':
            subprocess.run(['start', image_path], check=True, shell=True)
        elif os_name == 'Darwin': 
            subprocess.run(['open', image_path], check=True)
        elif os_name == 'Linux':
            subprocess.run(['xdg-open', image_path], check=True)
        else:
            print(f"Unsupported OS: {os_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to open image: {e}")

def plot_principal_components(multimodal_config, cumulative_variance_explained, optimal_components, file_suffix):
    """
    Plots the cumulative explained variance by the principal components to help determine
    the number of components needed for sufficient data representation in PCA.

    Parameters
    ----------
    cumulative_variance_explained : list of float
        Cumulative variance explained by each additional principal component.
    optimal_components : int
        The suggested number of principal components that provides optimal data representation.

    Notes
    -----
    This function creates a line plot displaying the cumulative explained variance against the number of components.
    An additional vertical line is plotted at the position indicating the optimal number of components.
    The plot is saved to the 'images/PCA' directory.
    """

    # Display cumulative explained variance with the intercepted value as label
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance_explained) + 1), cumulative_variance_explained, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Cumulative Explained Variance')
    plt.axvline(x=optimal_components, color='r', linestyle='-')
    plt.xticks([optimal_components])
    plt.grid(True)

    # pca_plots_dir = os.path.join(multimodal_config.save_path, "multimodal", "plots", "pca")

    # # Crea la directory se non esiste
    # if not os.path.exists(pca_plots_dir):
    #     try:
    #         os.makedirs(pca_plots_dir)
    #     except OSError as e:
    #         print(f"Error: {e.strerror}")
    #         return False

    # pca_plots_path = os.path.join(pca_plots_dir, "pca_cumulative_variance.png")

    # if multimodal_config.save_image.plot_pca_cumulative_variance:
    #     plt.savefig(pca_plots_path)

    if multimodal_config.save_images.plot_pca_cumulative_variance:
        figure_path = save_image(None, multimodal_config, "", file_suffix=file_suffix)
        plt.savefig(figure_path)
        plt.close()

    if multimodal_config.show_image.plot_pca_cumulative_variance:
        _open_image(figure_path)