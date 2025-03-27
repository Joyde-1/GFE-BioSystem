# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch # PyTorch for machine learning
import torch.nn as nn # Neural network modules in PyTorch

from multimodal.utils import load_config, browse_path, select_device, create_dataloader, select_model, evaluate, plot_inferences
from preprocessing_class.prepare_data import PrepareData # Data preparation class
from data_class.face_detection_dataset import FaceDetectionDataset  # Custom dataset class


def load_data(face_detection_cnn_config):
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
    prepare_data = PrepareData(face_detection_cnn_config)

    # Load and process training and test datasets
    _, _, test_set, mean, std = prepare_data.load_data()

    print("Face images and bounding boxes processed correctly. \n\n")

    # Create test dataset
    test_dataset = FaceDetectionDataset(test_set, face_detection_cnn_config.data.image_size, mean, std)

    return test_dataset, mean, std

def create_test_dataloader(face_detection_cnn_config, test_dataset):
    test_dl = create_dataloader(face_detection_cnn_config, test_dataset, False)

    # print some statistics
    print("Data dimensions: \n")
    print(f"Test size: {len(test_dl)} \n\n\n")

    return test_dl

def load_model(face_detection_cnn_config, model):
    # Load model weights
    model.load_state_dict(torch.load(f"{face_detection_cnn_config.training.checkpoints_dir}/{face_detection_cnn_config.training.model_name}.pt"))
    print(f"Model {face_detection_cnn_config.training.model_name} loaded. \n\n\n")

    return model


if __name__ == '__main__':

    # ••••••••••••••••••••••
    # •• Load config file ••
    # ••••••••••••••••••••••

    # Load configuration
    face_detection_cnn_config = load_config('face/face_detection_cnn/config/face_detection_cnn_config.yaml')

    if face_detection_cnn_config.browse_path:
        face_detection_cnn_config.data_dir = browse_path('Select the database folder')
        face_detection_cnn_config.bounding_boxes_dir = browse_path('Select the annotations folder')
        face_detection_cnn_config.training.checkpoints_dir = browse_path('Select the folder that contains model checkpoint')
        face_detection_cnn_config.save_data_splitted_path = browse_path('Select the folder where split dataset images will be saved')
        face_detection_cnn_config.save_path = browse_path('Select the folder where images and plots will be saved')



    # •••••••••••••••
    # •• Load data ••
    # •••••••••••••••

    # Load data
    test_dataset, mean, std = load_data(face_detection_cnn_config)

    # Create test dataloader
    test_dl = create_test_dataloader(face_detection_cnn_config, test_dataset)



    # ••••••••••••••••
    # •• Load model ••
    # ••••••••••••••••

    # Set device
    device = select_device(face_detection_cnn_config)
    print(f"Using device: {device} \n\n\n")

    # Select model
    model = select_model(face_detection_cnn_config)

    model.to(device)

    # Load model weights
    model = load_model(face_detection_cnn_config, model)

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
    plot_inferences(face_detection_cnn_config, test_images, test_references, test_predictions, mean, std)