# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from comet_ml import Experiment

import os # For operating system interactions
import shutil

# Local application/library specific imports
from multimodal.utils import load_config, browse_path, select_device, select_model, get_num_workers # Utility functions
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

def train_model(face_detection_yolo_config, model, device, num_workers):
    """
    Executes the training loop for a given model configuration.

    Parameters
    ----------
    face_detection_yolo_config : object
        Configuration object containing training parameters.
    model : torch.nn.Module
        The neural network model to be trained.

    Returns
    -------
    best_model : tuple
        A tuple containing the best model.    
    """

    print("\n\n")

    # TODO: (FACOLTATIVO) visualizzare la documentazione al seguente link e vedere quali altri iperparametri utili implementare:
    # https://docs.ultralytics.com/modes/train/#train-settings

    train_results = model.train(
        data=face_detection_yolo_config.train_yolo_model_path,              # File YAML che descrive il dataset
        epochs=face_detection_yolo_config.training.epochs,                  # Numero di epoche
        patience=face_detection_yolo_config.training.patience,
        batch=face_detection_yolo_config.training.batch_size,               # Dimensione batch
        imgsz=face_detection_yolo_config.data.image_size,                   # Dimensione immagine
        save=True,
        device=device,                                                      # Usa GPU (se disponibile)
        workers=num_workers,
        project=f'{face_detection_yolo_config.save_path}/face/yolo/training',
        name=face_detection_yolo_config.training.model_name,
        exist_ok=True,
        # optimizer=,
        seed=42,
        deterministic=True,
        single_cls=True,
        # cos_lr=,
        # lr0=,
        # lrf=,
        # momentum=,
        # weight_decay=,
        # warmup_epochs=,
        # warmup_momentum=,
        # warmup_bias_lr=,
        plots=True,
        # TODO: (FACOLTATIVO) verificare la possibilità di aggiungere altri parametri sull'augmentation per rendere più robusto il modello
    )

    print("\n")

    return model, train_results

def save_model(face_detection_yolo_config):
    """
    Saves the state dictionary of the best model to the specified checkpoint directory.

    Parameters
    ----------
    config : object
        Configuration object containing the training parameters including the checkpoint directory path.
    best_model : torch.nn.Module
        The model with the best performance to be saved.
    """

    # If the path exists, modify the name to make it unique
    model_path = os.path.join(face_detection_yolo_config.training.checkpoints_dir, f"{face_detection_yolo_config.training.model_name}.pt")

    # Save the best model's state dictionary to the checkpoint directory
    shutil.copyfile(f'{face_detection_yolo_config.save_path}/face/yolo/training/{face_detection_yolo_config.training.model_name}/weights/best.pt', model_path)
    
    # Print a message indicating the model has been saved
    print(f"Model saved as {model_path}. \n\n\n")

    
if __name__ == '__main__':

    # ••••••••••••••••••••••
    # •• Load config file ••
    # ••••••••••••••••••••••

    face_detection_yolo_config = load_config('face/face_detection_yolo/config/face_detection_yolo_config.yaml')

    if face_detection_yolo_config.browse_path:
        face_detection_yolo_config.data_dir = browse_path('Select the database folder')
        face_detection_yolo_config.bounding_boxes_dir = browse_path('Select the annotations folder')
        face_detection_yolo_config.training.checkpoints_dir = browse_path('Select the folder where model checkpoint will be saved')
        face_detection_yolo_config.save_data_splitted_path = browse_path('Select the folder where split dataset images will be saved')
        face_detection_yolo_config.save_path = browse_path('Select the folder where images and plots will be saved')



    # •••••••••••••••••••••••••••••
    # •• Create comet experiment ••
    # •••••••••••••••••••••••••••••

    experiment = Experiment(
        api_key=face_detection_yolo_config.comet.api_key,
        project_name=face_detection_yolo_config.comet.project_name,
        workspace=face_detection_yolo_config.comet.workspace,
        auto_metric_logging=False
    )

    experiment.set_name(face_detection_yolo_config.training.model_name)
    
    experiment.add_tag("yolo")

    experiment.log_parameters(face_detection_yolo_config.data)
    experiment.log_parameters(face_detection_yolo_config.training)



    # ••••••••••••••••••
    # •• Prepare data ••
    # ••••••••••••••••••
    
    prepare_data(face_detection_yolo_config)


    
    # ••••••••••••••••
    # •• Load model ••
    # ••••••••••••••••

    device = select_device(face_detection_yolo_config)
    print(f"Using device: {device} \n\n\n")

    num_workers = get_num_workers()

    print(f"Number of worker threads available on this pc: {num_workers} \n\n\n")

    model = select_model(face_detection_yolo_config)
    
    experiment.set_model_graph(str(model))



    # •••••••••••••••••
    # •• Train model ••
    # •••••••••••••••••

    trained_model, train_results = train_model(face_detection_yolo_config, model, device, num_workers)
    
    print("Detection Metrics:")
    for key, value in train_results.results_dict.items():
        print(f"- {key.replace('metrics/', '').replace('(B)', '')}: {value * 100:.2f} %")

    print("\n\n")



    # ••••••••••••••••
    # •• Save model ••
    # ••••••••••••••••

    save_model(face_detection_yolo_config)
    
    experiment.end()