import matplotlib.pyplot as plt
import matplotlib
import subprocess
import platform
import os
import cv2
import numpy as np
matplotlib.use('Agg')


def open_image(image_path):
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

def display_faces(image_size, original, transformed):
    # Assicurati che le immagini trasformate siano nel tipo e nella scala corretti
    transformed = np.clip(transformed, 0, 255)  # Assicura che i valori siano tra 0 e 255
    transformed = transformed.astype(np.uint8)  # Converte i valori in uint8

    # Mostra le immagini originale e trasformata
    for orig, trans in zip(original, transformed):
        orig = orig.reshape(image_size).astype(np.uint8)
        trans = trans.reshape(image_size)
        combined = np.hstack((orig, trans))
        cv2.imshow("Original vs. Transformed", combined)
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Premi 'q' per uscire prima della fine
            break
    cv2.destroyAllWindows()

def plot_images(original_image, blurred_image, enhanced_image, normalized_image, is_saved=True):
    # Mostra i risultati
    plt.figure(figsize=(15, 5))

    # Immagine originale
    plt.subplot(1, 4, 1)
    plt.title("Immagine Originale")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Scala di grigi e Gaussian Blur
    plt.subplot(1, 4, 2)
    plt.title("Scala di Grigi + Blur")
    plt.imshow(blurred_image, cmap='gray')
    plt.axis('off')

    # CLAHE
    plt.subplot(1, 4, 3)
    plt.title("CLAHE")
    plt.imshow(enhanced_image, cmap='gray')
    plt.axis('off')

    # Normalizzata e ridimensionata
    plt.subplot(1, 4, 4)
    plt.title("Ridimensionata e Normalizzata")
    plt.imshow(normalized_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()

    image_dir = "face/images"

    # Create the folder if it does not exist
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_path = os.path.join(image_dir, "original_vs_processed.png")

    if is_saved:
        plt.savefig(image_path)

    plt.close()
    
    open_image(image_path)