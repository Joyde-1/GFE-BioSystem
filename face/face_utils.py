from yaml_config_override import add_arguments # Custom YAML config handling
from addict import Dict # Dictionary-like class that allows attribute access
import yaml # YAML parsing
from pathlib import Path
import sys
import os
import cv2
from PyQt6.QtWidgets import QApplication, QFileDialog


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

def path_extractor(config, image_name, file_suffix=""):
    if file_suffix == "pca_cumulative_variance":
        save_path = os.path.join(config.save_path, "processed", "multimodal", "pca")
        # Crea la directory se non esiste
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except OSError as e:
                print(f"Error: {e.strerror}")
                return False
        file_name = f"{file_suffix}.png"
        full_path = os.path.join(save_path, file_name)
        return full_path

    save_path = os.path.join(config.save_path, "processed", "face", config.algorithm_type.replace("_", " "), file_suffix.replace("_", " "))

    # Crea la directory se non esiste
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except OSError as e:
            print(f"Error: {e.strerror}")
            return False
        
    # Genera il nome del file con suffisso
    if file_suffix == "plot_original_vs_processed":
        file_name = f"{image_name}_{file_suffix}.jpg"    
    else:
        file_name = f"{image_name}_{file_suffix}.bmp"

    full_path = os.path.join(save_path, file_name)
    
    return full_path

def save_image(config, image, image_name, file_suffix=""):
    """
    Save an image to a specified directory with a given filename.
    
    Parameters:
        image (numpy.ndarray): The image to save.
        save_path (str): Il percorso della directory di salvataggio principale.
        image_name (str): Il nome del file originale.
        eye_side (str): Il lato dell'occhio ("dx" o "sx").
    
    Returns:
        bool: True if the image was saved successfully, False otherwise.
    """

    full_path = path_extractor(config, image_name, file_suffix)
    
    # Salva l'immagine
    try:
        cv2.imwrite(full_path, image)
        print(f"Image saved successfully at {full_path}")
        return True
    except Exception as e:
        print(f"Failed to save image: {e}")
        return False