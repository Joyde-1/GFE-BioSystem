# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import math
import logging
import os # For operating system interactions
import sys
import time
import torch # PyTorch for machine learning
from tqdm import tqdm  # Progress bar

# Local application/library specific imports
# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gait_embedding_extraction.gait_embedding_extraction_utils import load_config, browse_path, select_device, create_dataloader, select_model, create_triplets, evaluate
from gait_embedding_extraction.gait_embedding_extraction_plots import plot_data_splitting
from gait_embedding_extraction.preprocessing_class.gait_embedding_extraction_prepare_data import PrepareData # Data preparation class
# from gait_embedding_extraction.preprocessing_class.gait_embedding_extraction_data_scaler import DataScaler # Data preparation class
from gait_embedding_extraction.preprocessing_class.gait_embedding_extraction_data_augmentation import default_augmentation # Data augmentation class
from gait_embedding_extraction.data_class.gait_embedding_extraction_dataset import GaitEmbeddingExtractionDataset  # Custom dataset class
from gait_embedding_extraction.checkpoint_classes.early_stopping import EarlyStopping
from gait_embedding_extraction.checkpoint_classes.model_checkpoint import ModelCheckpoint
# from gait_embedding_extraction.loss_function_class.loss import CombinedLoss  # Custom loss function


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
    train_set, val_set = prepare_data.prepare_data()
    # train_set, val_set, test_set = prepare_data.prepare_data()
    print(f"Gait keypoints sequences processed correctly. \n\n")

    print("\n\n")

    print("Train data shape: ", train_set['keypoints_sequences'].shape)
    print("Validation data shape: ", val_set['keypoints_sequences'].shape)
    # print("Test data shape: ", test_set['keypoints_sequences'].shape)

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

    plot_data_splitting(gait_embedding_extraction_config, train_set['keypoints_sequences'], val_set['keypoints_sequences'])
    # plot_data_splitting(gait_embedding_extraction_config, train_set['keypoints_sequences'], test_set['keypoints_sequences'], val_set['keypoints_sequences'])

    # data_scaler = DataScaler(train_set['keypoints_sequences'], test_set['keypoints_sequences'], val_set['keypoints_sequences'])
    # data_scaler = DataScaler(gait_embedding_extraction_config)

    # if gait_embedding_extraction_config.data.scaler == 'standard' or gait_embedding_extraction_config.data.scaler == 'min-max':
    #     data_scaler.fit_scaler(train_set['keypoints_sequences'])
    #     train_set['keypoints_sequences'] = data_scaler.scaling(train_set['keypoints_sequences'])
    #     val_set['keypoints_sequences'] = data_scaler.scaling(val_set['keypoints_sequences'])
    #     # test_set['keypoints_sequences'] = data_scaler.scaling(test_set['keypoints_sequences'])
        
    #     print("\n\n")

    #     print("Train data shape after scaling: ", train_set['keypoints_sequences'].shape)
    #     print("Validation data shape after scaling: ", val_set['keypoints_sequences'].shape)
    #     # print("Test data shape after scaling: ", test_set['keypoints_sequences'].shape)
    # elif gait_embedding_extraction_config.data.scaler == 'None':
    #     pass
    # else:
    #     raise ValueError("Unknown scaler type! \n")

    # Create train dataset
    train_dataset = GaitEmbeddingExtractionDataset(train_set, transform=default_augmentation(gait_embedding_extraction_config) if gait_embedding_extraction_config.data.data_augmentation else None, flatten=gait_embedding_extraction_config.data.flatten)
    # Create validation dataset
    val_dataset = GaitEmbeddingExtractionDataset(val_set, transform=None, flatten=gait_embedding_extraction_config.data.flatten)

    # # Create train dataset
    # train_dataset = GaitEmbeddingExtractionDataset(train_set)
    # # Create validation dataset
    # val_dataset = GaitEmbeddingExtractionDataset(val_set)
    # # Create test dataset
    # test_dataset = GaitEmbeddingExtractionDataset(test_set)

    return train_dataset, val_dataset
    # return train_dataset, val_dataset, test_dataset

def create_dataloaders(gait_embedding_extraction_config, train_dataset, val_dataset):
# def create_dataloaders(gait_embedding_extraction_config, train_dataset, val_dataset, test_dataset):
    train_dl = create_dataloader(gait_embedding_extraction_config, train_dataset, True)
    val_dl = create_dataloader(gait_embedding_extraction_config, val_dataset, False)
    # test_dl = create_dataloader(gait_embedding_extraction_config, test_dataset, False)

    # print some statistics
    print("Data dimensions: \n")
    print(f"Train size: {len(train_dl)}")
    print(f"Validation size: {len(val_dl)}")
    # print(f"Test size: {len(test_dl)} \n\n\n")

    return train_dl, val_dl
    # return train_dl, val_dl, test_dl

def prepare_training_process(gait_embedding_extraction_config, model, train_dl):
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

    # # Loss function
    # criterion = CombinedLoss(
    #     num_classes=gait_embedding_extraction_config.data.num_classes,
    #     embedding_dim=gait_embedding_extraction_config.model.embedding_dim,
    #     margin=gait_embedding_extraction_config.loss_function.triplet.margin,
    #     alpha=gait_embedding_extraction_config.loss_function.triplet.alpha,
    #     beta=gait_embedding_extraction_config.loss_function.center.beta
    # )
    # Define optimizer with learning rate from config
    optimizer = torch.optim.AdamW(model.parameters(), #model.parameters(), 
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
    elif gait_embedding_extraction_config.training.scheduler_name == "cosine":
        warm_steps = int(total_steps * gait_embedding_extraction_config.scheduler.cosine.warmup_epochs / gait_embedding_extraction_config.training.epochs)

        def lr_lambda(step):
            if step < warm_steps:
                return step / float(max(1, warm_steps))
            progress = (step - warm_steps) / float(max(1, total_steps - warm_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler name: {gait_embedding_extraction_config.training.scheduler_name}")
    
    return optimizer, scheduler

def train_one_epoch(gait_embedding_extraction_config, model, dataloader, optimizer, scheduler, device, epoch, experiment):
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

    # Track running loss and accuracy
    running_loss = 0.0
    running_ce_loss = 0.0
    running_center_loss = 0.0
    correct = 0
    total = 0

    # # List to store embeddings
    # all_ambeddings = []
    
    # # List to store labels
    # all_labels = []
    
    # Iterate over batches
    for step, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
        # Move keypoints sequences to device  
        keypoints_sequences = batch['keypoints_sequence'].to(device)

        # Move labels to device
        labels = batch['label'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        embeddings, logits, center_loss = model(keypoints_sequences)

        ce_loss = model.ce_loss(logits, labels)

        loss = ce_loss + center_loss

        # # Compute loss (la nuova loss function gestisce tutto internamente)
        # loss, ce_loss, triplet_loss = criterion(embeddings, logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping per stabilità
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimize parameters
        optimizer.step()

        # Adjust learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            scheduler.step()
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(epoch + step / len(dataloader))

        # Update running loss
        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        running_center_loss += center_loss.item()

        pred = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

        # _, predicted = torch.max(logits.data, 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()
        
        # Store predictions and true labels
        # pred = torch.argmax(embeddings, dim=1)
        # predictions.extend(pred.cpu().numpy())
        # references.extend(labels.cpu().numpy())

    # Compute training metrics    
    train_metrics = {}

    # Average loss
    train_metrics['loss'] = running_loss / len(dataloader)
    train_metrics['ce_loss'] = running_ce_loss / len(dataloader)
    train_metrics['center_loss'] = running_center_loss / len(dataloader)
    train_metrics['accuracy'] = 100.0 * correct / total

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(train_metrics[gait_embedding_extraction_config.training.evaluation_metric])

    return train_metrics

# Calculate the time taken 
def training_time(start_time, current_time):
    tot_time = current_time - start_time
    hours = int(tot_time / 3600)
    mins = int((tot_time / 60) - (hours * 60))
    secs = int(tot_time - (mins * 60))

    return hours, mins, secs

def training_loop(gait_embedding_extraction_config, model, device, train_dl, val_dl, optimizer, scheduler, experiment):
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

    # # Move loss function parameters to the same device
    # criterion.to(device)

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

    best_train_metrics = None
    best_val_metrics = None

    # Training loop
    for epoch in range(gait_embedding_extraction_config.training.epochs):
        print(f"Epoch {epoch + 1}/{gait_embedding_extraction_config.training.epochs}")
        
        # Train for one epoch
        train_metrics = train_one_epoch(gait_embedding_extraction_config, model, train_dl, optimizer, scheduler, device, epoch, experiment)
        
        # Validate model
        val_metrics = evaluate(gait_embedding_extraction_config, model, val_dl, device, epoch, experiment)

        print("")
        
        # Print metrics
        print("Train metrics:")
        print(f"Loss: {train_metrics['loss']:.4f}",
              f"Cross-Entropy Loss: {train_metrics['ce_loss']:.4f}",
              f"Center Loss: {train_metrics['center_loss']:.4f}",
              f"Accuracy: {train_metrics['accuracy']:.4f}", sep="\n", end="\n\n")
              
        print("Val metrics:")
        print(f"Loss: {val_metrics['loss']:.4f}",
              f"Accuracy: {val_metrics['accuracy']:.4f}",
              f"EER: {val_metrics['EER']}",
              f"FAR@1%: {val_metrics['FAR@1%']}",
              f"FRR@1%: {val_metrics['FRR@1%']}",
              f"Threshold: {val_metrics['threshold']}", sep="\n", end="\n\n")

            #   f"Average Intra Distances: {val_metrics['avg_intra_dist']}", 
            #   f"Average Inter Distances: {val_metrics['avg_inter_dist']}",
            #   f"Inter-Intra Ratio: {val_metrics['inter_intra_ratio']}",
            #   f"EER: {val_metrics['eer']}",
            #   f"Threshold: {val_metrics['threshold']}",
            #   f"FAR: {val_metrics['far']}",
            #   f"FRR: {val_metrics['frr']}",
            #   f"Rank1: {val_metrics['rank1']}",
            #   f"Rank5: {val_metrics['rank5']}",
            #   f"ROC AUC: {val_metrics['roc_auc']}",
            #   f"Silhouette Score: {val_metrics['silhouette_score']}",
            #   f"Davies-Bouldin Score: {val_metrics['davies_bouldin_score']}",
            #   f"Calinski-Harabasz Score: {val_metrics['calinski_harabasz_score']}", sep="\n")

        early_stopping(val_metrics, epoch + 1)

        if best_val_metrics is None or val_metrics["EER"] < best_val_metrics["EER"]:
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics

        if gait_embedding_extraction_config.training.save_best_weights:
            model_checkpoint(val_metrics, model)

        current_time = time.time()
        hours, mins, secs = training_time(start_time, current_time)

        print(f'Training time: {hours}:{mins}:{secs}\n')

        print("\n\n")

        if early_stopping.early_stop:
            break

    if gait_embedding_extraction_config.training.save_best_weights:
        model.load_state_dict(model_checkpoint.best_model_state)

    print("\n")

    return model, best_train_metrics, best_val_metrics

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


if __name__ == "__main__":
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
    train_dataset, val_dataset = load_data(gait_embedding_extraction_config)
    # train_dataset, val_dataset, test_dataset = load_data(gait_embedding_extraction_config)

    # Create dataloaders
    train_dl, val_dl = create_dataloaders(gait_embedding_extraction_config, train_dataset, val_dataset)
    # train_dl, val_dl, test_dl = create_dataloaders(gait_embedding_extraction_config, train_dataset, val_dataset, test_dataset)

    

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
    optimizer, scheduler = prepare_training_process(gait_embedding_extraction_config, model, train_dl)

    # Train model
    best_model, best_train_metrics, best_val_metrics = training_loop(gait_embedding_extraction_config, model, device, train_dl, val_dl, optimizer, scheduler, experiment)
    
    # Print best metrics
    print("BEST METRICS:")
    print("Best train metrics:")
    print(f"Loss: {best_train_metrics['loss']:.4f}",
          f"Cross-Entropy Loss: {best_train_metrics['ce_loss']:.4f}",
          f"Center Loss: {best_train_metrics['center_loss']:.4f}",
          f"Accuracy: {best_train_metrics['accuracy']:.4f}", sep="\n")
    print("Best val metrics:")
    print(f"Loss: {best_val_metrics['loss']:.4f}",
          f"Accuracy: {best_val_metrics['accuracy']:.4f}",
          f"EER: {best_val_metrics['EER']}",
          f"FAR@1%: {best_val_metrics['FAR@1%']}",
          f"FRR@1%: {best_val_metrics['FRR@1%']}",
          f"Threshold: {best_val_metrics['threshold']}", sep="\n")



    # ••••••••••••••••
    # •• Save model ••
    # ••••••••••••••••
    
    save_model(gait_embedding_extraction_config, best_model)

    # Log the model to Comet for easy tracking and deployment
    log_model(experiment, model, gait_embedding_extraction_config.training.model_name)
    
    experiment.end()