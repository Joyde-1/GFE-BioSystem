# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from ultralytics import YOLO

from utils import load_config, browse_path, select_device, evaluate_model, plot_inferences # Utility functions
# from plots import plot_predictions_vs_references, plot_classes_roc_curves, plot_metrics
from preprocessing_class.prepare_data import PrepareData # Data preparation class


def prepare_data(detection_yolo_config, biometric_trait):
    """
    Loads and processes the training, validation, and test datasets based on the provided configuration.

    Parameters
    ----------
    config : object
        Contains the configuration information.   
    """

    # Initialize data preparation object with the data directory from the config file
    prepare_data = PrepareData(detection_yolo_config, biometric_trait)

    # Load and process training, validation and test datasets
    prepare_data.prepare_data()

    print("Face images and bounding boxes processed correctly. \n\n")

def load_model(detection_yolo_config):
    # Load model weights
    model = YOLO(f"{detection_yolo_config.training.checkpoints_dir}/{detection_yolo_config.training.model_name}.pt")
    print(f"Model {detection_yolo_config.training.model_name} loaded. \n\n\n")

    return model


if __name__ == '__main__':

    # Select the biometric trait between:
    # 'face'
    # 'ear_dx'
    # 'ear_sx'
    biometric_trait = 'ear_dx'

    # ••••••••••••••••••••••
    # •• Load config file ••
    # ••••••••••••••••••••••

    # Load configuration file
    if biometric_trait == 'face':
        detection_yolo_config = load_config('yolo_detection/config/face_detection_yolo_config.yaml')
    elif biometric_trait == 'ear_dx':
        detection_yolo_config = load_config('yolo_detection/config/ear_dx_detection_yolo_config.yaml')
    elif biometric_trait == 'ear_sx':
        detection_yolo_config = load_config('yolo_detection/config/ear_sx_detection_yolo_config.yaml')
    else:
        raise ValueError("Unknown biometric trait! \n")

    if detection_yolo_config.browse_path:
        detection_yolo_config.data_dir = browse_path('Select the database folder')
        detection_yolo_config.bounding_boxes_dir = browse_path('Select the annotations folder')
        detection_yolo_config.training.checkpoints_dir = browse_path('Select the folder that contains model checkpoint')
        detection_yolo_config.save_data_splitted_path = browse_path('Select the folder where split dataset images will be saved')
        detection_yolo_config.save_path = browse_path('Select the folder where images and plots will be saved')



    # ••••••••••••••••••
    # •• Prepare data ••
    # ••••••••••••••••••

    # Load data
    prepare_data(detection_yolo_config, biometric_trait)



    # ••••••••••••••••
    # •• Load model ••
    # ••••••••••••••••

    # Set device
    device = select_device(detection_yolo_config)
    print(f"Using device: {device} \n\n\n")

    # # Get number of workers in this pc
    # num_workers = get_num_workers()

    # print(f"Number of worker threads available on this pc: {num_workers} \n\n\n")

    # Load model weights
    model = load_model(detection_yolo_config)



    # ••••••••••••••••••••
    # •• Evaluate model ••
    # ••••••••••••••••••••

    # Evaluate the best model on the test set
    test_metrics = evaluate_model(detection_yolo_config, biometric_trait, model, device)

    print("\n\n")

    print("Detection Metrics:")
    for key, value in test_metrics.results_dict.items():
        print(f"- {key.replace('metrics/', '').replace('(B)', '')}: {value * 100:.2f} %")

    print("\n\n")



    # ••••••••••••••••••
    # •• Plot results ••
    # ••••••••••••••••••

    # Plot predictions on the test set
    plot_inferences(detection_yolo_config, biometric_trait, model, device)