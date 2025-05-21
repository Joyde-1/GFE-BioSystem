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

def plot_data_splitting(gait_embedding_extraction_config, train, test, val=[]):
    """
    Displays a pie chart showing the distribution of samples across training, validation, and testing datasets.

    This visualization helps to understand the proportion of data split among different subsets used for model training
    and evaluation.

    Parameters
    ----------
    train : list or array-like
        The training dataset.
    test : list or array-like
        The testing dataset.
    val : list or array-like, optional
        The validation dataset. If not provided, the function will plot only training and testing sets.

    Returns
    -------
    None
        The function does not return any value but saves a pie chart plot in a specified directory.

    Examples
    --------
    >>> train_samples = [1, 2, 3, 4, 5]
    >>> test_samples = [6, 7, 8]
    >>> plot_data_splitting(train_samples, test_samples)

    Notes
    -----
    The function checks if the 'images/data_splitting' directory exists and creates it if not. The pie chart
    is saved under this directory with the name 'data_splitting_plot.png'.
    """
    
    n_train = len(train)
    n_val = len(val)
    n_test = len(test)

    if n_val == 0:
        sizes = [n_train, n_test]
        sets = ["Train", "Test"]
        colors = ["limegreen", "red"]
        explode = (0, 0)
    else:
        sizes = [n_train, n_val, n_test]
        sets = ["Train", "Validation", "Test"]
        colors = ["limegreen", "blue", "red"]
        explode = (0, 0, 0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(sizes,
           colors=colors,
           explode=explode,
           labels=sets,
           wedgeprops={'edgecolor':'white'},
           autopct='%1.1f%%',
           shadow=False)
    plt.axis('equal')
    plt.title('Dataset division')

    plt.tight_layout()

    data_split_dir = os.path.join(gait_embedding_extraction_config.save_path, "face", "cnn", "plots", "data split plot")

    # Crea la directory se non esiste
    if not os.path.exists(data_split_dir):
        try:
            os.makedirs(data_split_dir)
        except OSError as e:
            print(f"Error: {e.strerror}")
            return False

    data_split_path = os.path.join(data_split_dir, "data_split_plot.png")

    if gait_embedding_extraction_config.save_image.plot_data_splitting:
        plt.savefig(data_split_path)
    
    plt.close()

    if gait_embedding_extraction_config.show_image.plot_data_splitting:
        open_image(data_split_path)