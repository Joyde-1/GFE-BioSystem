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

def load_checkpoint(file_path):
    import os
    import json

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint
    else:
        return None

def save_checkpoint(file_path, current_index):
    import json

    checkpoint = {
        'current_index': current_index
    }
    with open(file_path, 'w') as f:
        json.dump(checkpoint, f)

def path_extractor(config, biometric_trait, image_name, file_suffix="", index=None):
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

    save_path = os.path.join(config.save_path, "processed", biometric_trait, file_suffix.replace("_", " "))
   
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
        if index is None:
            file_name = f"{image_name}_{file_suffix}.png"
        else:
            file_name = f"{image_name}_{index}_{file_suffix}.png"

    full_path = os.path.join(save_path, file_name)
    
    return full_path

def save_image(config, biometric_trait, image, image_name, file_suffix="", index=None):
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

    full_path = path_extractor(config, biometric_trait, image_name, file_suffix, index)
    
    # Salva l'immagine
    try:
        cv2.imwrite(full_path, image)
        print(f"Image saved successfully at {full_path}")
        return True
    except Exception as e:
        print(f"Failed to save image: {e}")
        return False
    
def save_gif(frames, config, gif_name, eye_side=None, file_suffix=""):
    if eye_side != None:
        save_path = os.path.join(config.save_path, "processed", "ear", config.algorithm_type.replace("_", " "), file_suffix.replace("_", " "))
    else:
        save_path = os.path.join(config.save_path, "processed", "face", config.algorithm_type.replace("_", " "), file_suffix.replace("_", " "))

    # Crea la directory se non esiste
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except OSError as e:
            print(f"Error: {e.strerror}")
            return False

    if eye_side != None:
        # Crea la sottocartella per il lato (dx o sx)
        save_path = os.path.join(save_path, eye_side)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Genera il nome del file con suffisso
    file_name = f"{gif_name}_{file_suffix}.gif"
    full_path = os.path.join(save_path, file_name)

    # Salva la gif
    if len(frames) > 0:
        imageio.mimsave(full_path, frames, fps=25)
        print(f"Saved evolution as GIF: {full_path}")
        return True
    else:
        print("No frames collected to save GIF.")
        return False    