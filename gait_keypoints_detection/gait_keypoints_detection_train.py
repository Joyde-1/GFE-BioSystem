# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# from comet_ml import Experiment

import os # For operating system interactions
import sys
import shutil

# import mmcv
# from mmengine import Config
# from mmpose.datasets import build_dataset
# from mmpose.models import build_posenet
# from mmpose.apis import train_model

# Local application/library specific imports
# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gait_keypoints_detection.gait_keypoints_detection_utils import load_config, browse_path, select_device, select_model, get_num_workers # Utility functions
from gait_keypoints_detection.preprocessing_class.gait_keypoints_detection_prepare_data import PrepareData # Data preparation class


def prepare_data(gait_keypoints_detection_config):
    """
    Loads and processes the training, validation, and test datasets based on the provided configuration.

    Parameters
    ----------
    config : object
        Contains the configuration information.   
    """

    # Initialize data preparation object with the data directory from the config file
    prepare_data = PrepareData(gait_keypoints_detection_config)

    # Load and process training, validation and test datasets
    prepare_data.prepare_data()

    print(f"Gait images and keypoints processed correctly. \n\n")

def train_model(gait_keypoints_detection_config, biometric_trait, model, device, num_workers):
    """
    Executes the training loop for a given model configuration.

    Parameters
    ----------
    gait_keypoints_detection_config : object
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
        data=gait_keypoints_detection_config.train_gait_keypoints_model_path,              # File YAML che descrive il dataset
        epochs=gait_keypoints_detection_config.training.epochs,                  # Numero di epoche
        patience=gait_keypoints_detection_config.training.patience,
        batch=gait_keypoints_detection_config.training.batch_size,               # Dimensione batch
        imgsz=gait_keypoints_detection_config.data.image_size,                   # Dimensione immagine
        save=True,
        device=device,                                                      # Usa GPU (se disponibile)
        workers=num_workers,
        project=f'{gait_keypoints_detection_config.save_path}/{biometric_trait}/gait_keypoints/training',
        name=gait_keypoints_detection_config.training.model_name,
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

def save_model(gait_keypoints_detection_config, biometric_trait):
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
    model_path = os.path.join(gait_keypoints_detection_config.training.checkpoints_dir, f"{gait_keypoints_detection_config.training.model_name}.pt")

    # Save the best model's state dictionary to the checkpoint directory
    shutil.copyfile(f'{gait_keypoints_detection_config.save_path}/{biometric_trait}/gait_keypoints/training/{gait_keypoints_detection_config.training.model_name}/weights/best.pt', model_path)
    
    # Print a message indicating the model has been saved
    print(f"Model saved as {model_path}. \n\n\n")

    
if __name__ == '__main__':

    # ••••••••••••••••••••••
    # •• Load config file ••
    # ••••••••••••••••••••••

    gait_keypoints_detection_config = load_config('gait_keypoints_detection/config/gait_keypoints_detection_config.yaml')
    # train_gait_keypoints_detection_config = Config.fromfile('gait_keypoints_detection/config/train_gait_keypoints_detection_model.py')

    if gait_keypoints_detection_config.browse_path:
        gait_keypoints_detection_config.data_dir = browse_path('Select the database folder')
        gait_keypoints_detection_config.bounding_boxes_dir = browse_path('Select the annotations folder')
        gait_keypoints_detection_config.training.checkpoints_dir = browse_path('Select the folder where model checkpoint will be saved')
        gait_keypoints_detection_config.save_data_splitted_path = browse_path('Select the folder where split dataset images will be saved')
        gait_keypoints_detection_config.save_path = browse_path('Select the folder where images and plots will be saved')



    # •••••••••••••••••••••••••••••
    # •• Create comet experiment ••
    # •••••••••••••••••••••••••••••

    # experiment = Experiment(
    #     api_key=gait_keypoints_detection_config.comet.api_key,
    #     project_name=gait_keypoints_detection_config.comet.project_name,
    #     workspace=gait_keypoints_detection_config.comet.workspace,
    #     auto_metric_logging=False
    # )

    # experiment.set_name(gait_keypoints_detection_config.training.model_name)
    
    # experiment.add_tag("gait_keypoints_detection")

    # experiment.log_parameters(gait_keypoints_detection_config.data)
    # experiment.log_parameters(gait_keypoints_detection_config.training)



    # ••••••••••••••••••
    # •• Prepare data ••
    # ••••••••••••••••••
    
    prepare_data(gait_keypoints_detection_config)

    # Costruisci il dataset di training
    datasets = [build_dataset(train_gait_keypoints_detection_config.data.train)]

    # Costruisci il modello in base alla configurazione
    model = build_posenet(train_gait_keypoints_detection_config.model)
    
    # Crea la directory di work_dir, se non esiste
    mmcv.mkdir_or_exist(os.path.abspath(train_gait_keypoints_detection_config.work_dir))
    
    # Avvia il training
    train_model(model, datasets, train_gait_keypoints_detection_config, distributed=False, validate=True)


    
    # ••••••••••••••••
    # •• Load model ••
    # ••••••••••••••••

    # device = select_device(gait_keypoints_detection_config)
    # print(f"Using device: {device} \n\n\n")

    # num_workers = get_num_workers()

    # print(f"Number of worker threads available on this pc: {num_workers} \n\n\n")

    # model = select_model(gait_keypoints_detection_config)
    
    # experiment.set_model_graph(str(model))



    # •••••••••••••••••
    # •• Train model ••
    # •••••••••••••••••

    # trained_model, train_results = train_model(gait_keypoints_detection_config, biometric_trait, model, device, num_workers)
    
    # print("Detection Metrics:")
    # for key, value in train_results.results_dict.items():
    #     print(f"- {key.replace('metrics/', '').replace('(B)', '')}: {value * 100:.2f} %")

    # print("\n\n")



    # ••••••••••••••••
    # •• Save model ••
    # ••••••••••••••••

    # save_model(gait_keypoints_detection_config, biometric_trait)
    
    # experiment.end()