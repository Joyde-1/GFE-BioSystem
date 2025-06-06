# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import logging
import os # For operating system interactions
import sys
import time
import torch # PyTorch for machine learning
import torch.nn as nn # Neural network modules in PyTorch
from tqdm import tqdm  # Progress bar
from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.distances import CosineSimilarity

# Local application/library specific imports
# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gait_embedding_extraction_simple.gait_embedding_extraction_utils import load_config, browse_path, select_device, create_dataloader, select_model, compute_metrics, evaluate
from gait_embedding_extraction_simple.gait_embedding_extraction_plots import plot_data_splitting
from gait_embedding_extraction_simple.preprocessing_class.gait_embedding_extraction_prepare_data import PrepareData # Data preparation class
from gait_embedding_extraction_simple.preprocessing_class.gait_embedding_extraction_data_scaler import DataScaler # Data preparation class
from gait_embedding_extraction_simple.data_class.gait_embedding_extraction_dataset import GaitEmbeddingExtractionDataset  # Custom dataset class
from gait_embedding_extraction_simple.checkpoint_classes.early_stopping import EarlyStopping
from gait_embedding_extraction_simple.checkpoint_classes.model_checkpoint import ModelCheckpoint


def load_data(gait_embedding_extraction_config):
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
    prepare_data = PrepareData(gait_embedding_extraction_config)

    # Load and process training, validation and test datasets
    train_set, val_set, test_set = prepare_data.prepare_data()
    print(f"Gait keypoints sequences processed correctly. \n\n")

    print("\n\n")

    print("Train data shape: ", train_set['keypoints_sequences'].shape)
    print("Validation data shape: ", val_set['keypoints_sequences'].shape)
    print("Test data shape: ", test_set['keypoints_sequences'].shape)

    # # Reshape the data
    # print("Reshape Train data: \n")
    # print("Train data shape before reshape: ", train_set['keypoints_sequences'].shape)
    # train_set = prepare_data.reshape_data(train_set) 
    # print("Train data shape after reshape: ", train_set['keypoints_sequences'].shape, "\n\n")
    
    # print("Reshape Validation data: \n")
    # print("Validation data shape before reshape: ", val_set['keypoints_sequences'].shape)
    # val_set = prepare_data.reshape_data(val_set) 
    # print("Validation data shape after reshape: ", val_set['keypoints_sequences'].shape, "\n\n")

    # print("Reshape Test data: \n")
    # print("Test data shape before reshape: ", test_set['keypoints_sequences'].shape)
    # test_set = prepare_data.reshape_data(test_set)
    # print("Test data shape after reshape: ", test_set['keypoints_sequences'].shape, "\n\n")

    # Determine the number of features from the training dataset
    # n_features = train_set['keypoints_sequences'].shape[2]

    plot_data_splitting(gait_embedding_extraction_config, train_set['keypoints_sequences'], test_set['keypoints_sequences'], val_set['keypoints_sequences'])

    # data_scaler = DataScaler(train_set['keypoints_sequences'], test_set['keypoints_sequences'], val_set['keypoints_sequences'])
    data_scaler = DataScaler(gait_embedding_extraction_config)

    if gait_embedding_extraction_config.data.scaler == 'standard' or gait_embedding_extraction_config.data.scaler == 'min-max':
        data_scaler.fit_scaler(train_set['keypoints_sequences'])
        train_set['keypoints_sequences'] = data_scaler.scaling(train_set['keypoints_sequences'])
        val_set['keypoints_sequences'] = data_scaler.scaling(val_set['keypoints_sequences'])
        test_set['keypoints_sequences'] = data_scaler.scaling(test_set['keypoints_sequences'])
        
        print("\n\n")

        print("Train data shape after scaling: ", train_set['keypoints_sequences'].shape)
        print("Validation data shape after scaling: ", val_set['keypoints_sequences'].shape)
        print("Test data shape after scaling: ", test_set['keypoints_sequences'].shape)
    elif gait_embedding_extraction_config.data.scaler == 'None':
        pass
    else:
        raise ValueError("Unknown scaler type! \n")

    # Create train dataset
    train_dataset = GaitEmbeddingExtractionDataset(train_set)
    # Create validation dataset
    val_dataset = GaitEmbeddingExtractionDataset(val_set)
    # Create test dataset
    test_dataset = GaitEmbeddingExtractionDataset(test_set)

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(gait_embedding_extraction_config, train_dataset, val_dataset, test_dataset):
    train_dl = create_dataloader(gait_embedding_extraction_config, train_dataset, True)
    val_dl = create_dataloader(gait_embedding_extraction_config, val_dataset, False)
    test_dl = create_dataloader(gait_embedding_extraction_config, test_dataset, False)

    # print some statistics
    print("Data dimensions: \n")
    print(f"Train size: {len(train_dl)}")
    print(f"Validation size: {len(val_dl)}")
    print(f"Test size: {len(test_dl)} \n\n\n")

    return train_dl, val_dl, test_dl

def prepare_training_process(gait_embedding_extraction_config, model, train_dl, device):
    """
    Prepares the training process by defining the loss function, optimizer, and learning rate scheduler.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained.
    config : object
        Configuration object containing training parameters.
    train_dl : torch.utils.data.DataLoader
        DataLoader for the training dataset.

    Returns
    -------
    criterion : nn.CrossEntropyLoss
        The selected criterion.
    optimizer : torch.optim.Adam     
        The selected optimizer.
    scheduler : torch.optim.lr_scheduler.LambdaLR        
        The selected scheduler.
    """

    # Define loss function
    # criterion = nn.CrossEntropyLoss()

    criterion = ArcFaceLoss(
        num_classes=gait_embedding_extraction_config.data.num_classes,
        embedding_size=gait_embedding_extraction_config.model.embedding_dim, # model.fc.out_features,  # dimensione embedding
        margin=gait_embedding_extraction_config.loss_function.arcface.margin,
        scale=gait_embedding_extraction_config.loss_function.arcface.scale
    ).to(device)

    # Define optimizer with learning rate from config
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), #model.parameters(), 
                                  lr=gait_embedding_extraction_config.optimizer.adamw.lr, 
                                  betas=gait_embedding_extraction_config.optimizer.adamw.betas, 
                                  eps=float(gait_embedding_extraction_config.optimizer.adamw.eps), 
                                  weight_decay=float(gait_embedding_extraction_config.optimizer.adamw.weight_decay))
    
    # learning rate scheduler
    total_steps = len(train_dl) * gait_embedding_extraction_config.training.epochs

    if gait_embedding_extraction_config.training.scheduler_name == "reduce_lr_on_plateau":
        mode = 'min' if gait_embedding_extraction_config.training.lower_is_better else 'max'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               mode=mode, 
                                                               factor=gait_embedding_extraction_config.scheduler.reduce_lr_on_plateau.factor, 
                                                               patience=gait_embedding_extraction_config.scheduler.reduce_lr_on_plateau.patience,
                                                               threshold=gait_embedding_extraction_config.scheduler.reduce_lr_on_plateau.threshold,
                                                               threshold_mode=gait_embedding_extraction_config.scheduler.reduce_lr_on_plateau.threshold_mode,
                                                               cooldown=gait_embedding_extraction_config.scheduler.reduce_lr_on_plateau.cooldown,
                                                               min_lr=gait_embedding_extraction_config.scheduler.reduce_lr_on_plateau.min_lr,
                                                               eps=float(gait_embedding_extraction_config.scheduler.reduce_lr_on_plateau.eps))
    elif gait_embedding_extraction_config.training.scheduler_name == "cosine_annealing_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                         T_0=gait_embedding_extraction_config.scheduler.cosine_annealing_warm_restarts.t_0, 
                                                                         T_mult=gait_embedding_extraction_config.scheduler.cosine_annealing_warm_restarts.t_mult, 
                                                                         eta_min=gait_embedding_extraction_config.scheduler.cosine_annealing_warm_restarts.eta_min, 
                                                                         last_epoch=gait_embedding_extraction_config.scheduler.cosine_annealing_warm_restarts.last_epoch)
    elif gait_embedding_extraction_config.training.scheduler_name == "warm_up":
        warmup_steps = int(total_steps * gait_embedding_extraction_config.scheduler.warm_up.warmup_ratio)

        # warmup + linear decay
        scheduler_lambda = lambda step: (step / warmup_steps) if step < warmup_steps else max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    
        # Define scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)
    else:
        raise ValueError(f"Unknown scheduler name: {gait_embedding_extraction_config.training.scheduler_name}")
    
    return criterion, optimizer, scheduler

def train_one_epoch(gait_embedding_extraction_config, model, dataloader, criterion, optimizer, scheduler, device, epoch, experiment):
    """
    Trains the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    dataloader : DataLoader
        DataLoader for the training data.
    criterion : nn.Module
        The loss function.
    optimizer : Optimizer
        The optimizer.
    scheduler : _LRScheduler
        Learning rate scheduler.
    device : torch.device
        Device to run the training on (CPU, CUDA, or MPS).

    Returns
    -------
    dict
        Training metrics including loss and accuracy.
    """

    # Set model to training mode
    model.train()

    # Track running loss
    running_loss = 0.0

    # List to store predictions
    predictions = []
    
    # List to store true labels
    references = []
    
    # Iterate over batches
    for i, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
        
        # Move keypoints sequences to device  
        keypoints_sequences = batch['keypoints_sequence'].to(device)

        # Move labels to device
        labels = batch['label'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        embeddings = model(keypoints_sequences)
        
        # Compute loss
        loss = criterion(embeddings, labels)
        
        # Backward pass
        loss.backward()

        # Optimize parameters
        optimizer.step()

        # Adjust learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            scheduler.step()
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(epoch + i / len(dataloader))
        
        # Update running loss
        running_loss += loss.item()
        
        # Store predictions and true labels
        pred = torch.argmax(embeddings, dim=1)
        predictions.extend(pred.cpu().numpy())
        references.extend(labels.cpu().numpy())

    # Compute training metrics    
    train_metrics = compute_metrics(predictions, references)

    # Average loss
    train_metrics['loss'] = running_loss / len(dataloader)

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(train_metrics[gait_embedding_extraction_config.training.evaluation_metric])

    # Loggare le metriche su Comet.ml
    experiment.log_metric("train_loss", train_metrics['loss'], epoch=epoch)
    experiment.log_metric("train_accuracy", train_metrics['accuracy'], epoch=epoch)
    experiment.log_metric("train_f1", train_metrics['f1'], epoch=epoch)
    experiment.log_metric("train_precision", train_metrics['precision'], epoch=epoch)
    experiment.log_metric("train_recall", train_metrics['recall'], epoch=epoch)
    # experiment.log_metric("train_loss", train_metrics['loss'], epoch=epoch)
    # experiment.log_metric("train_mae", train_metrics['mae'], epoch=epoch)
    # experiment.log_metric("train_mse", train_metrics['mse'], epoch=epoch)
    # experiment.log_metric("train_rmse", train_metrics['rmse'], epoch=epoch)
    # experiment.log_metric("train_r2", train_metrics['r2'], epoch=epoch)
    # experiment.log_metric("train_mape", train_metrics['mape'], epoch=epoch)

    return train_metrics

# Calculate the time taken 
def training_time(start_time, current_time):
    tot_time = current_time - start_time
    hours = int(tot_time / 3600)
    mins = int((tot_time / 60) - (hours * 60))
    secs = int(tot_time - (mins * 60))

    return hours, mins, secs

def training_loop(gait_embedding_extraction_config, model, device, train_dl, val_dl, criterion, optimizer, scheduler, experiment):
    """
    Executes the training loop for a given model configuration.

    Parameters
    ----------
    config : object
        Configuration object containing training parameters.
    model : torch.nn.Module
        The neural network model to be trained.
    train_dl : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_dl : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    criterion : torch.nn.Module
        Loss function used for training.
    optimizer : torch.optim.Optimizer
        Optimizer used for updating model parameters.
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.

    Returns
    -------
    best_val_metric : tuple
        A tuple containing the best validation metric.
    best_model : tuple
        A tuple containing the best model.    
    """
    
    model.to(device)

    early_stopping = EarlyStopping(monitor=gait_embedding_extraction_config.training.evaluation_metric, 
                                   min_delta=gait_embedding_extraction_config.training.min_delta,
                                   patience=gait_embedding_extraction_config.training.patience, 
                                   verbose=True,
                                   lower_is_better=gait_embedding_extraction_config.training.lower_is_better,
                                   baseline=None if gait_embedding_extraction_config.training.initial_value_threshold == "None" else gait_embedding_extraction_config.training.initial_value_threshold,
                                   start_from_epoch=gait_embedding_extraction_config.training.start_from_epoch)
    
    if gait_embedding_extraction_config.training.save_best_weights:
        model_checkpoint = ModelCheckpoint(monitor=gait_embedding_extraction_config.training.evaluation_metric, 
                                           min_delta=gait_embedding_extraction_config.training.min_delta,
                                           verbose=True,
                                           lower_is_better=gait_embedding_extraction_config.training.lower_is_better,
                                           initial_value_threshold=None if gait_embedding_extraction_config.training.initial_value_threshold == "None" else gait_embedding_extraction_config.training.initial_value_threshold)

    start_time = time.time()

    # Training loop
    for epoch in range(gait_embedding_extraction_config.training.epochs):
        print(f"Epoch {epoch + 1}/{gait_embedding_extraction_config.training.epochs}")
        
        # Train for one epoch
        train_metrics = train_one_epoch(gait_embedding_extraction_config, model, train_dl, criterion, optimizer, scheduler, device, epoch, experiment)
        
        # Validate model
        val_metrics = evaluate(model, val_dl, criterion, device)

        print("")
        
        # Print metrics
        print("Train metrics:")
        print(f"ArcFaceLoss: {train_metrics['loss']:.4f}",
              f"Accuracy: {train_metrics['accuracy']:.4f}",
              f"F1: {train_metrics['f1']:.4f}",
              f"Precision: {train_metrics['precision']:.4f}",
              f"Recall: {train_metrics['recall']:.4f} \n", sep="\n")
        # print(f"Loss: {train_metrics['loss']:.4f}",
        #       f"MAE: {train_metrics['mae']:.4f}",
        #       f"MSE: {train_metrics['mse']:.4f}",
        #       f"RMSE: {train_metrics['rmse']:.4f}",
        #       f"R2: {train_metrics['r2']:.4f}",
        #       f"MAPE: {train_metrics['mape']:.4f} \n", sep="\n")
              
        print("Val metrics:")
        print(f"ArcFaceLoss: {val_metrics['loss']:.4f}",
              f"Accuracy: {val_metrics['accuracy']:.4f}",
              f"F1: {val_metrics['f1']:.4f}",
              f"Precision: {val_metrics['precision']:.4f}",
              f"Recall: {val_metrics['recall']:.4f} \n", sep="\n")
        # print(f"Loss: {val_metrics['loss']:.4f}",
        #       f"MAE: {val_metrics['mae']:.4f}",
        #       f"MSE: {val_metrics['mse']:.4f}",
        #       f"RMSE: {val_metrics['rmse']:.4f}",
        #       f"R2: {val_metrics['r2']:.4f}",
        #       f"MAPE: {val_metrics['mape']:.4f} \n", sep="\n")
        # print(f"Val loss: {val_metrics['loss']:.4f} - Val MAE: {val_metrics['mae']:.4f} - Val MSE: {val_metrics['mse']:.4f}", sep="", end="")
        # print(f"Val RMSE: {val_metrics['rmse']:.4f} - Val R2: {val_metrics['r2']:.4f} - Val MAPE: {val_metrics['mape']:.4f} \n")

        # Loggare le metriche di validazione su Comet.ml
        experiment.log_metric("val_loss", val_metrics['loss'], epoch=epoch)
        experiment.log_metric("val_accuracy", val_metrics['accuracy'], epoch=epoch)
        experiment.log_metric("val_f1", val_metrics['f1'], epoch=epoch)
        experiment.log_metric("val_precision", val_metrics['precision'], epoch=epoch)
        experiment.log_metric("val_recall", val_metrics['recall'], epoch=epoch)
        # experiment.log_metric("val_loss", val_metrics['loss'], epoch=epoch)
        # experiment.log_metric("val_mae", val_metrics['mae'], epoch=epoch)
        # experiment.log_metric("val_mse", val_metrics['mse'], epoch=epoch)
        # experiment.log_metric("val_rmse", val_metrics['rmse'], epoch=epoch)
        # experiment.log_metric("val_r2", val_metrics['r2'], epoch=epoch)
        # experiment.log_metric("val_mape", val_metrics['mape'], epoch=epoch)

        early_stopping(val_metrics, epoch + 1)

        if gait_embedding_extraction_config.training.save_best_weights:
            model_checkpoint(val_metrics, model)

        if early_stopping.early_stop:
            break

        current_time = time.time()
        hours, mins, secs = training_time(start_time, current_time)

        print(f'Training time: {hours}:{mins}:{secs}\n')

        print("\n\n")

    if gait_embedding_extraction_config.training.save_best_weights:
        model.load_state_dict(model_checkpoint.best_model_state)

    print("\n")

    return model

def save_model(gait_embedding_extraction_config, model):
    """
    Saves the state dictionary of the best model to the specified checkpoint directory.

    Parameters
    ----------
    config : object
        Configuration object containing the training parameters including the checkpoint directory path.
    best_model : torch.nn.Module
        The model with the best performance to be saved.
    """

    # Ensure the checkpoint directory exists, create if it doesn't
    os.makedirs(f"{gait_embedding_extraction_config.training.checkpoints_dir}", exist_ok=True)

    # If the path exists, modify the name to make it unique
    model_path = os.path.join(f"{gait_embedding_extraction_config.training.checkpoints_dir}", f"{gait_embedding_extraction_config.training.model_name}.pt")

    # Save the best model's state dictionary to the checkpoint directory
    torch.save(model.state_dict(), model_path)
    
    # Print a message indicating the model has been saved
    print(f"Model saved as {model_path}. \n\n\n")

    
if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # ••••••••••••••••••••••
    # •• Load config file ••
    # ••••••••••••••••••••••

    # Load configuration file
    gait_embedding_extraction_config = load_config('gait_embedding_extraction/config/gait_embedding_extraction_config.yaml')
    
    if gait_embedding_extraction_config.browse_path:
        gait_embedding_extraction_config.data_dir = browse_path('Select the database folder')
        gait_embedding_extraction_config.keypoints_sequences_dir = browse_path('Select the gait keypoints sequences folder')
        gait_embedding_extraction_config.training.checkpoints_dir = browse_path('Select the folder where model checkpoint will be saved')
        gait_embedding_extraction_config.save_data_splitted_path = browse_path('Select the folder where split dataset will be saved')
        gait_embedding_extraction_config.save_path = browse_path('Select the folder where images and plots will be saved')



    # •••••••••••••••••••••••••••••
    # •• Create comet experiment ••
    # •••••••••••••••••••••••••••••

    experiment = Experiment(
        api_key=gait_embedding_extraction_config.comet.api_key,
        project_name=f'{gait_embedding_extraction_config.comet.project_name}',
        workspace=gait_embedding_extraction_config.comet.workspace,
        auto_metric_logging=False
    )

    experiment.set_name(gait_embedding_extraction_config.training.model_name)
    
    experiment.add_tag(f"{gait_embedding_extraction_config.comet.project_name}")

    experiment.log_parameters(gait_embedding_extraction_config.data)
    experiment.log_parameters(gait_embedding_extraction_config.training)

    experiment.log_parameters(gait_embedding_extraction_config.model)
    experiment.log_parameters(gait_embedding_extraction_config.optimizer.adamw)
    experiment.log_parameters(gait_embedding_extraction_config.scheduler)



    # •••••••••••••••
    # •• Load data ••
    # •••••••••••••••

    # Load data
    train_dataset, val_dataset, test_dataset = load_data(gait_embedding_extraction_config)

    # Create dataloaders
    train_dl, val_dl, test_dl = create_dataloaders(gait_embedding_extraction_config, train_dataset, val_dataset, test_dataset)


    
    # ••••••••••••••••
    # •• Load model ••
    # ••••••••••••••••
    
    # Set device
    device = select_device(gait_embedding_extraction_config)
    print(f"Using device: {device} \n\n\n")

    # Select model
    model = select_model(gait_embedding_extraction_config, device)

    experiment.set_model_graph(str(model))



    # •••••••••••••••••
    # •• Train model ••
    # •••••••••••••••••
    
    # Prepare training process
    criterion, optimizer, scheduler = prepare_training_process(gait_embedding_extraction_config, model, train_dl, device)

    # Train model
    best_model = training_loop(gait_embedding_extraction_config, model, device, train_dl, val_dl, criterion, optimizer, scheduler, experiment)



    # ••••••••••••••••
    # •• Save model ••
    # ••••••••••••••••
    
    save_model(gait_embedding_extraction_config, best_model)

    # Log the model to Comet for easy tracking and deployment
    log_model(experiment, model, gait_embedding_extraction_config.training.model_name)
    
    experiment.end()