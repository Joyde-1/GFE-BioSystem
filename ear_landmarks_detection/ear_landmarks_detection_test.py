# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os # For operating system interactions
import sys
import torch # PyTorch for machine learning
import torch.nn as nn # Neural network modules in PyTorch

# Local application/library specific imports
# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ear_landmarks_detection.ear_landmarks_detection_utils import load_config, browse_path, select_device, create_dataloader, select_model, evaluate, plot_inferences
from ear_landmarks_detection.preprocessing_class.ear_landmarks_detection_prepare_data import PrepareData # Data preparation class
from ear_landmarks_detection.data_class.ear_landmarks_dataset import EarLandmarksDataset  # Custom dataset class


def load_data(ear_landmarks_detection_config):
    """
    Loads and processes the training, validation, and test datasets based on the provided configuration.

    Parameters
    ----------
    config : object
        Contains the configuration information.

    Returns
    -------
    train_dl : torch.utils.data.DataLoader
        Contains train DataLoader.
    val_dl : torch.utils.data.DataLoader
        Contains validation DataLoader.
    val_dl : torch.utils.data.DataLoader
        Contains test DataLoader.        
    """

    # Initialize data preparation object with the data directory from the config file
    prepare_data = PrepareData(ear_landmarks_detection_config)

    # Load and process training and test datasets
    train_set, _, test_set = prepare_data.load_data()

    print(f"{ear_landmarks_detection_config.biometric_trait} images and landmarks processed correctly. \n\n")

    # Calculate images mean and std
    if ear_landmarks_detection_config.data.use_mean_and_std == 'current_database':
        mean, std = prepare_data.calculate_mean_and_std(train_set['images'])
        print(f"Mean (R, G, B) (current database): {mean}")
        print(f"Standard Deviation (R, G, B) (current database): {std} \n\n")
    elif ear_landmarks_detection_config.data.use_mean_and_std == 'image_net':
        mean, std = ear_landmarks_detection_config.data.image_net.mean, ear_landmarks_detection_config.data.image_net.std
        print(f"Media (R, G, B) (ImageNet): {ear_landmarks_detection_config.data.image_net.mean}")
        print(f"Standard Deviation (R, G, B) (ImageNet): {ear_landmarks_detection_config.data.image_net.std} \n\n")
    elif ear_landmarks_detection_config.data.use_mean_and_std == 'None':
        mean = None
        std = None
    else:
        raise ValueError("Unknown mode to use mean and std! \n")

    # Create test dataset
    test_dataset = EarLandmarksDataset(test_set, ear_landmarks_detection_config.data.image_size, mean, std)

    return test_dataset, mean, std

def create_test_dataloader(ear_landmarks_detection_config, test_dataset):
    test_dl = create_dataloader(ear_landmarks_detection_config, test_dataset, False)

    # print some statistics
    print("Data dimensions: \n")
    print(f"Test size: {len(test_dl)} \n\n\n")

    return test_dl

def load_model(ear_landmarks_detection_config, model):
    model_path = os.path.join(f"{ear_landmarks_detection_config.training.checkpoints_dir}{ear_landmarks_detection_config.biometric_trait}_landmarks_detection", f"{ear_landmarks_detection_config.training.model_name}.pt")

    # Load model weights
    model.load_state_dict(torch.load(model_path))
    print(f"Model {ear_landmarks_detection_config.training.model_name} loaded. \n\n\n")

    return model


if __name__ == '__main__':

    # ••••••••••••••••••••••
    # •• Load config file ••
    # ••••••••••••••••••••••

    # Load configuration file
    ear_landmarks_detection_config = load_config('ear_landmarks_detection/config/ear_landmarks_detection_config.yaml')

    # Check biometric trait
    if ear_landmarks_detection_config.biometric_trait != 'ear_dx' and ear_landmarks_detection_config.biometric_trait != 'ear_sx':
        raise ValueError("Unknown biometric trait! \n")
    
    if ear_landmarks_detection_config.browse_path:
        ear_landmarks_detection_config.data_dir = browse_path('Select the database folder')
        ear_landmarks_detection_config.landmarks_dir = browse_path('Select the ear landmarks folder')
        ear_landmarks_detection_config.training.checkpoints_dir = browse_path('Select the folder where model checkpoint will be saved')
        ear_landmarks_detection_config.save_data_splitted_path = browse_path('Select the folder where split dataset images and landmarks will be saved')
        ear_landmarks_detection_config.save_path = browse_path('Select the folder where images and plots will be saved')


    
    # •••••••••••••••
    # •• Load data ••
    # •••••••••••••••

    # Load data
    test_dataset, mean, std = load_data(ear_landmarks_detection_config)

    # Create test dataloader
    test_dl = create_test_dataloader(ear_landmarks_detection_config, test_dataset)



    # ••••••••••••••••
    # •• Load model ••
    # ••••••••••••••••

    # Set device
    device = select_device(ear_landmarks_detection_config)
    print(f"Using device: {device} \n\n\n")

    # Select model
    model = select_model(ear_landmarks_detection_config)

    model.to(device)

    # Load model weights
    model = load_model(ear_landmarks_detection_config, model)

    # Criterion
    criterion = nn.SmoothL1Loss()



    # ••••••••••••••••••••
    # •• Evaluate model ••
    # ••••••••••••••••••••

    # Evaluate the best model on the test set
    test_metrics, test_images, test_references, test_predictions = evaluate(model, test_dl, criterion, device, return_test_images=True)

    print("\n\n")

    print("Test metrics: \n")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n\n")



    # ••••••••••••••••••
    # •• Plot results ••
    # ••••••••••••••••••

    # Plot predictions on the test set
    plot_inferences(ear_landmarks_detection_config, test_images, test_references, test_predictions, mean, std)