# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import os # For operating system interactions
import torch # PyTorch for machine learning
import torch.nn as nn # Neural network modules in PyTorch
import time
from tqdm import tqdm  # Progress bar

# Local application/library specific imports
from multimodal.utils import load_config, browse_path, select_device, create_dataloader, select_model, compute_metrics, evaluate
from multimodal.plots import plot_data_splitting
from preprocessing_class.prepare_data import PrepareData # Data preparation class
from data_class.face_detection_dataset import FaceDetectionDataset  # Custom dataset class
from checkpoint_classes.early_stopping import EarlyStopping
from checkpoint_classes.model_checkpoint import ModelCheckpoint


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

    # Load and process training, validation and test datasets
    train_set, val_set, test_set, mean, std = prepare_data.load_data()

    print("Face images and bounding boxes processed correctly. \n\n")

    plot_data_splitting(face_detection_cnn_config, train_set['bounding_boxes'], test_set['bounding_boxes'], val_set['bounding_boxes'])
    
    # Create train dataset
    train_dataset = FaceDetectionDataset(train_set, face_detection_cnn_config.data.image_size, mean, std)

    # Create validation dataset
    val_dataset = FaceDetectionDataset(val_set, face_detection_cnn_config.data.image_size, mean, std)

    # Create test dataset
    test_dataset = FaceDetectionDataset(test_set, face_detection_cnn_config.data.image_size, mean, std)

    return train_dataset, val_dataset, test_dataset, mean, std

def create_dataloaders(face_detection_cnn_config, train_dataset, val_dataset, test_dataset):
    train_dl = create_dataloader(face_detection_cnn_config, train_dataset, True)
    val_dl = create_dataloader(face_detection_cnn_config, val_dataset, False)
    test_dl = create_dataloader(face_detection_cnn_config, test_dataset, False)

    # print some statistics
    print("Data dimensions: \n")
    print(f"Train size: {len(train_dl)}")
    print(f"Validation size: {len(val_dl)}")
    print(f"Test size: {len(test_dl)} \n\n\n")

    return train_dl, val_dl, test_dl

def prepare_training_process(face_detection_cnn_config, model, train_dl):
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
    criterion = nn.SmoothL1Loss()

    # Define optimizer with learning rate from config
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=face_detection_cnn_config.optimizer.adamw.lr, 
                                  betas=face_detection_cnn_config.optimizer.adamw.betas, 
                                  eps=float(face_detection_cnn_config.optimizer.adamw.eps), 
                                  weight_decay=float(face_detection_cnn_config.optimizer.adamw.weight_decay))
    
    # learning rate scheduler
    total_steps = len(train_dl) * face_detection_cnn_config.training.epochs

    if face_detection_cnn_config.training.scheduler_name == "reduce_lr_on_plateau":
        mode = 'min' if face_detection_cnn_config.training.lower_is_better else 'max'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               mode=mode, 
                                                               factor=face_detection_cnn_config.scheduler.reduce_lr_on_plateau.factor, 
                                                               patience=face_detection_cnn_config.scheduler.reduce_lr_on_plateau.patience,
                                                               threshold=face_detection_cnn_config.scheduler.reduce_lr_on_plateau.threshold,
                                                               threshold_mode=face_detection_cnn_config.scheduler.reduce_lr_on_plateau.threshold_mode,
                                                               cooldown=face_detection_cnn_config.scheduler.reduce_lr_on_plateau.cooldown,
                                                               min_lr=face_detection_cnn_config.scheduler.reduce_lr_on_plateau.min_lr,
                                                               eps=float(face_detection_cnn_config.scheduler.reduce_lr_on_plateau.eps))
    elif face_detection_cnn_config.training.scheduler_name == "cosine_annealing_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                         T_0=face_detection_cnn_config.scheduler.cosine_annealing_warm_restarts.t_0, 
                                                                         T_mult=face_detection_cnn_config.scheduler.cosine_annealing_warm_restarts.t_mult, 
                                                                         eta_min=face_detection_cnn_config.scheduler.cosine_annealing_warm_restarts.eta_min, 
                                                                         last_epoch=face_detection_cnn_config.scheduler.cosine_annealing_warm_restarts.last_epoch)
    elif face_detection_cnn_config.training.scheduler_name == "warm_up":
        warmup_steps = int(total_steps * face_detection_cnn_config.scheduler.warm_up.warmup_ratio)

        # warmup + linear decay
        scheduler_lambda = lambda step: (step / warmup_steps) if step < warmup_steps else max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    
        # Define scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)
    else:
        raise ValueError(f"Unknown scheduler name: {face_detection_cnn_config.training.scheduler_name}")
    
    return criterion, optimizer, scheduler

def train_one_epoch(face_detection_cnn_config, model, dataloader, criterion, optimizer, scheduler, device, epoch):
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
        
        # Move sequences to device  
        images = batch['image'].to(device)

        # Move labels to device
        bounding_boxes = batch['bounding_box'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, bounding_boxes)
        
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
        predictions.extend(outputs.cpu().detach().numpy())
        # pred = torch.argmax(outputs, dim=4)
        # predictions.extend(pred.cpu().numpy())
        references.extend(bounding_boxes.cpu().detach().numpy())

    # Compute training metrics    
    train_metrics = compute_metrics(predictions, references)

    # Average loss
    train_metrics['loss'] = running_loss / len(dataloader)

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(train_metrics[face_detection_cnn_config.training.evaluation_metric])

    # Loggare le metriche su Comet.ml
    experiment.log_metric("train_loss", train_metrics['loss'], epoch=epoch)
    experiment.log_metric("train_mae", train_metrics['mae'], epoch=epoch)
    experiment.log_metric("train_mse", train_metrics['mse'], epoch=epoch)
    experiment.log_metric("train_rmse", train_metrics['rmse'], epoch=epoch)
    experiment.log_metric("train_r2", train_metrics['r2'], epoch=epoch)
    experiment.log_metric("train_mape", train_metrics['mape'], epoch=epoch)

    return train_metrics

# Calculate the time taken 
def training_time(start_time, current_time):
    tot_time = current_time - start_time
    hours = int(tot_time / 3600)
    mins = int((tot_time / 60) - (hours * 60))
    secs = int(tot_time - (mins * 60))

    return hours, mins, secs

def training_loop(face_detection_cnn_config, model, device, train_dl, val_dl, criterion, optimizer, scheduler):
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

    early_stopping = EarlyStopping(monitor=face_detection_cnn_config.training.evaluation_metric, 
                                   min_delta=face_detection_cnn_config.training.min_delta,
                                   patience=face_detection_cnn_config.training.patience, 
                                   verbose=True,
                                   lower_is_better=face_detection_cnn_config.training.lower_is_better,
                                   baseline=None if face_detection_cnn_config.training.initial_value_threshold == "None" else face_detection_cnn_config.training.initial_value_threshold,
                                   start_from_epoch=face_detection_cnn_config.training.start_from_epoch)
    
    if face_detection_cnn_config.training.save_best_weights:
        model_checkpoint = ModelCheckpoint(monitor=face_detection_cnn_config.training.evaluation_metric, 
                                           min_delta=face_detection_cnn_config.training.min_delta,
                                           verbose=True,
                                           lower_is_better=face_detection_cnn_config.training.lower_is_better,
                                           initial_value_threshold=None if face_detection_cnn_config.training.initial_value_threshold == "None" else face_detection_cnn_config.training.initial_value_threshold)

    start_time = time.time()

    # Training loop
    for epoch in range(face_detection_cnn_config.training.epochs):
        print(f"Epoch {epoch + 1}/{face_detection_cnn_config.training.epochs}")
        
        # Train for one epoch
        train_metrics = train_one_epoch(face_detection_cnn_config, model, train_dl, criterion, optimizer, scheduler, device, epoch)
        
        # Validate model
        val_metrics = evaluate(model, val_dl, criterion, device)

        print("")
        
        # Print metrics
        print("Train metrics:")
        print(f"Loss: {train_metrics['loss']:.4f}",
              f"MAE: {train_metrics['mae']:.4f}",
              f"MSE: {train_metrics['mse']:.4f}",
              f"RMSE: {train_metrics['rmse']:.4f}",
              f"R2: {train_metrics['r2']:.4f}",
              f"MAPE: {train_metrics['mape']:.4f} \n", sep="\n")
              
        print("Val metrics:")
        print(f"Loss: {val_metrics['loss']:.4f}",
              f"MAE: {val_metrics['mae']:.4f}",
              f"MSE: {val_metrics['mse']:.4f}",
              f"RMSE: {val_metrics['rmse']:.4f}",
              f"R2: {val_metrics['r2']:.4f}",
              f"MAPE: {val_metrics['mape']:.4f} \n", sep="\n")
        # print(f"Val loss: {val_metrics['loss']:.4f} - Val MAE: {val_metrics['mae']:.4f} - Val MSE: {val_metrics['mse']:.4f}", sep="", end="")
        # print(f"Val RMSE: {val_metrics['rmse']:.4f} - Val R2: {val_metrics['r2']:.4f} - Val MAPE: {val_metrics['mape']:.4f} \n")

        # Loggare le metriche di validazione su Comet.ml
        experiment.log_metric("val_loss", val_metrics['loss'], epoch=epoch)
        experiment.log_metric("val_mae", val_metrics['mae'], epoch=epoch)
        experiment.log_metric("val_mse", val_metrics['mse'], epoch=epoch)
        experiment.log_metric("val_rmse", val_metrics['rmse'], epoch=epoch)
        experiment.log_metric("val_r2", val_metrics['r2'], epoch=epoch)
        experiment.log_metric("val_mape", val_metrics['mape'], epoch=epoch)

        early_stopping(val_metrics, epoch + 1)

        if face_detection_cnn_config.training.save_best_weights:
            model_checkpoint(val_metrics, model)

        if early_stopping.early_stop:
            break

        current_time = time.time()
        hours, mins, secs = training_time(start_time, current_time)

        print(f'Training time: {hours}:{mins}:{secs}\n')

        print("\n\n")

    if face_detection_cnn_config.training.save_best_weights:
        model.load_state_dict(model_checkpoint.best_model_state)

    print("\n")

    return model

def save_model(face_detection_cnn_config, model):
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
    os.makedirs(face_detection_cnn_config.training.checkpoints_dir, exist_ok=True)

    # If the path exists, modify the name to make it unique
    model_path = os.path.join(face_detection_cnn_config.training.checkpoints_dir, f"{face_detection_cnn_config.training.model_name}.pt")

    # Save the best model's state dictionary to the checkpoint directory
    torch.save(model.state_dict(), model_path)
    
    # Print a message indicating the model has been saved
    print(f"Model saved as {model_path}. \n\n\n")

    
if __name__ == '__main__':

    # ••••••••••••••••••••••
    # •• Load config file ••
    # ••••••••••••••••••••••

    face_detection_cnn_config = load_config('face/face_detection_cnn/config/face_detection_cnn_config.yaml')



    # •••••••••••••••••••••••••••••
    # •• Create comet experiment ••
    # •••••••••••••••••••••••••••••

    experiment = Experiment(
        api_key=face_detection_cnn_config.comet.api_key,
        project_name=face_detection_cnn_config.comet.project_name,
        workspace=face_detection_cnn_config.comet.workspace,
        auto_metric_logging=False
    )

    experiment.set_name(face_detection_cnn_config.training.model_name)
    
    experiment.add_tag("pytorch")

    experiment.log_parameters(face_detection_cnn_config.data)
    experiment.log_parameters(face_detection_cnn_config.training)

    experiment.log_parameters(face_detection_cnn_config.model)
    experiment.log_parameters(face_detection_cnn_config.optimizer.adamw)
    experiment.log_parameters(face_detection_cnn_config.scheduler)



    # •••••••••••••••
    # •• Load data ••
    # •••••••••••••••

    train_dataset, val_dataset, test_dataset, mean, std = load_data(face_detection_cnn_config)

    train_dl, val_dl, test_dl = create_dataloaders(face_detection_cnn_config, train_dataset, val_dataset, test_dataset)


    
    # ••••••••••••••••
    # •• Load model ••
    # ••••••••••••••••
    
    device = select_device(face_detection_cnn_config)
    print(f"Using device: {device} \n\n\n")

    model = select_model(face_detection_cnn_config)
    
    experiment.set_model_graph(str(model))



    # •••••••••••••••••
    # •• Train model ••
    # •••••••••••••••••
            
    criterion, optimizer, scheduler = prepare_training_process(face_detection_cnn_config, model, train_dl)

    best_model = training_loop(face_detection_cnn_config, model, device, train_dl, val_dl, criterion, optimizer, scheduler)



    # ••••••••••••••••
    # •• Save model ••
    # ••••••••••••••••
    
    save_model(face_detection_cnn_config, best_model)

    # Log the model to Comet for easy tracking and deployment
    log_model(experiment, model, face_detection_cnn_config.training.model_name)
    
    experiment.end()