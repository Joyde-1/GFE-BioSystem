import matplotlib.pyplot as plt
import matplotlib
import os
import subprocess
import platform

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

def plot_prediction(face_detection_yolo_config, reference_image, predicted_image, acquisition_name):
    """
    Plots the predicted and ground truth bounding boxes on the image.

    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        The input image. Shape should be (C, H, W) if torch.Tensor or (H, W, C) if np.ndarray.
    """

    # Plotta l'immagine, la maschera di riferimento e la maschera predetta
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(reference_image, cmap="gray")
    plt.axis("off")
    plt.title("Reference Bounding Box")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_image, cmap="gray")
    plt.axis("off")
    plt.title("Predicted Bounding Box")

    plt.tight_layout()
    
    prediction_plots_dir = os.path.join(face_detection_yolo_config.save_path, "face", "yolo", "test", "plots", "prediction plots")

    # Crea la directory se non esiste
    if not os.path.exists(prediction_plots_dir):
        try:
            os.makedirs(prediction_plots_dir)
        except OSError as e:
            print(f"Error: {e.strerror}")
            return False

    prediction_plots_path = os.path.join(prediction_plots_dir, f"prediction_plot_{acquisition_name}.png")

    if face_detection_yolo_config.save_image.plot_prediction:
        plt.savefig(prediction_plots_path)

    plt.close()

    if face_detection_yolo_config.show_image.plot_prediction:
        open_image(prediction_plots_path)