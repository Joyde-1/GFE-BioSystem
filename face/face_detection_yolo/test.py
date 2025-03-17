# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from ultralytics import YOLO

from multimodal.utils import load_config, browse_path, select_device, evaluate_model, plot_inferences # Utility functions
# from plots import plot_predictions_vs_references, plot_classes_roc_curves, plot_metrics
from preprocessing_class.prepare_data import PrepareData # Data preparation class


def prepare_data(face_detection_yolo_config):
    """
    Loads and processes the training, validation, and test datasets based on the provided configuration.

    Parameters
    ----------
    config : object
        Contains the configuration information.   
    """

    # Initialize data preparation object with the data directory from the config file
    prepare_data = PrepareData(face_detection_yolo_config)

    # Load and process training, validation and test datasets
    prepare_data.prepare_data()

    print("Face images and bounding boxes processed correctly. \n\n")

def load_model(face_detection_yolo_config):
    # Load model weights
    model = YOLO(f"{face_detection_yolo_config.training.checkpoints_dir}/{face_detection_yolo_config.training.model_name}.pt")
    print(f"Model {face_detection_yolo_config.training.model_name} loaded. \n\n\n")

    return model


if __name__ == '__main__':

    # ••••••••••••••••••••••
    # •• Load config file ••
    # ••••••••••••••••••••••

    # Load configuration
    face_detection_yolo_config = load_config('face/face_detection_yolo/config/face_detection_yolo_config.yaml')

    if face_detection_yolo_config.browse_path:
        face_detection_yolo_config.data_dir = browse_path('Select the database folder')
        face_detection_yolo_config.bounding_boxes_dir = browse_path('Select the annotations folder')
        face_detection_yolo_config.training.checkpoints_dir = browse_path('Select the folder that contains model checkpoint')
        face_detection_yolo_config.save_data_splitted_path = browse_path('Select the folder where split dataset images will be saved')
        face_detection_yolo_config.save_path = browse_path('Select the folder where images and plots will be saved')



    # ••••••••••••••••••
    # •• Prepare data ••
    # ••••••••••••••••••

    # Load data
    prepare_data(face_detection_yolo_config)



    # ••••••••••••••••
    # •• Load model ••
    # ••••••••••••••••

    # Set device
    device = select_device(face_detection_yolo_config)
    print(f"Using device: {device} \n\n\n")

    # # Get number of workers in this pc
    # num_workers = get_num_workers()

    # print(f"Number of worker threads available on this pc: {num_workers} \n\n\n")

    # Load model weights
    model = load_model(face_detection_yolo_config)



    # ••••••••••••••••••••
    # •• Evaluate model ••
    # ••••••••••••••••••••

    # Evaluate the best model on the test set
    test_metrics = evaluate_model(face_detection_yolo_config, model, device)

    print("\n\n")

    print("Detection Metrics:")
    for key, value in test_metrics.results_dict.items():
        print(f"- {key.replace('metrics/', '').replace('(B)', '')}: {value * 100:.2f} %")

    print("\n\n")



    # ••••••••••••••••••
    # •• Plot results ••
    # ••••••••••••••••••

    # Plot predictions on the test set
    plot_inferences(face_detection_yolo_config, model, device)