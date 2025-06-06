# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
from ultralytics import YOLO

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from yolo_pose_detection.yolo_pose_utils import load_config, browse_path, select_device, evaluate_model, plot_inferences # Utility functions
# from plots import plot_predictions_vs_references, plot_classes_roc_curves, plot_metrics
from yolo_pose_detection.preprocessing_class.yolo_pose_prepare_data import PrepareData # Data preparation class


def prepare_data(yolo_pose_detection_config, biometric_trait):
    """
    Loads and processes the training, validation, and test datasets based on the provided configuration.

    Parameters
    ----------
    config : object
        Contains the configuration information.   
    """

    # Initialize data preparation object with the data directory from the config file
    prepare_data = PrepareData(yolo_pose_detection_config, biometric_trait)

    # Load and process training, validation and test datasets
    # prepare_data.prepare_data()

    print(f"{biometric_trait} images and bounding boxes processed correctly. \n\n")

def load_model(yolo_pose_detection_config):
    # Load model weights
    model = YOLO(f"{yolo_pose_detection_config.training.checkpoints_dir}/{yolo_pose_detection_config.training.model_name}.pt")
    print(f"Model {yolo_pose_detection_config.training.model_name} loaded. \n\n\n")

    return model


if __name__ == '__main__':

    # Select the biometric trait (only gait available):
    biometric_trait = 'gait'

    # ••••••••••••••••••••••
    # •• Load config file ••
    # ••••••••••••••••••••••

    # Load configuration file
    if biometric_trait == 'gait':
        yolo_pose_detection_config = load_config('yolo_pose_detection/config/gait_yolo_pose_detection_config.yaml')
    else:
        raise ValueError("Unknown biometric trait! \n")

    if yolo_pose_detection_config.browse_path:
        yolo_pose_detection_config.data_dir = browse_path('Select the database folder')
        yolo_pose_detection_config.bounding_boxes_dir = browse_path('Select the annotations folder')
        yolo_pose_detection_config.training.checkpoints_dir = browse_path('Select the folder that contains model checkpoint')
        yolo_pose_detection_config.save_data_splitted_path = browse_path('Select the folder where split dataset images will be saved')
        yolo_pose_detection_config.save_path = browse_path('Select the folder where images and plots will be saved')



    # ••••••••••••••••••
    # •• Prepare data ••
    # ••••••••••••••••••

    # Load data
    prepare_data(yolo_pose_detection_config, biometric_trait)



    # ••••••••••••••••
    # •• Load model ••
    # ••••••••••••••••

    # Set device
    device = select_device(yolo_pose_detection_config)
    print(f"Using device: {device} \n\n\n")

    # # Get number of workers in this pc
    # num_workers = get_num_workers()

    # print(f"Number of worker threads available on this pc: {num_workers} \n\n\n")

    # Load model weights
    model = load_model(yolo_pose_detection_config)



    # ••••••••••••••••••••
    # •• Evaluate model ••
    # ••••••••••••••••••••

    # Evaluate the best model on the test set
    test_metrics = evaluate_model(yolo_pose_detection_config, biometric_trait, model, device)

    print("\n\n")

    print("Detection Metrics:")
    for key, value in test_metrics.results_dict.items():
        print(f"- {key.replace('metrics/', '').replace('(B)', '')}: {value * 100:.2f} %")

    print("\n\n")



    # ••••••••••••••••••
    # •• Plot results ••
    # ••••••••••••••••••

    # Plot predictions on the test set
    plot_inferences(yolo_pose_detection_config, biometric_trait, model, device)