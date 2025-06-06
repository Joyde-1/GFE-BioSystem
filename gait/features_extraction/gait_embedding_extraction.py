# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import torch
import numpy as np

try:
    from gait_embedding_extraction.gait_embedding_extraction_utils import load_config, select_device, select_model
    from gait_embedding_extraction.preprocessing_class.gait_embedding_extraction_data_scaler import DataScaler
except ModuleNotFoundError:
    # Fallback to relative import
    sys.path.append('/Users/giovanni/Desktop/Tesi di Laurea/GFE-BioSystem')
    from gait_embedding_extraction.gait_embedding_extraction_utils import load_config, select_device, select_model
    from gait_embedding_extraction.preprocessing_class.gait_embedding_extraction_data_scaler import DataScaler


class GaitEmbeddingExtraction:
    def __init__(self, gait_config):
        self._gait_config = gait_config
        self._prepare_predict_process()

    def _load_model(self, gait_embedding_extraction_config):
        model_path = os.path.join(gait_embedding_extraction_config.training.checkpoints_dir, f"{gait_embedding_extraction_config.training.model_name}.pt")
        # Load model weights
        self.model.load_state_dict(torch.load(model_path))

    def _prepare_predict_process(self):
        # Load configuration
        gait_embedding_extraction_config = load_config('gait_embedding_extraction/config/gait_embedding_extraction_config.yaml')

        # Set device
        self.device = select_device(gait_embedding_extraction_config)

        # Select model
        self.model = select_model(gait_embedding_extraction_config, self.device)

        self.model.to(self.device)

        # Load model weights
        self._load_model(gait_embedding_extraction_config)

        if gait_embedding_extraction_config.data.scaler == 'standard' or gait_embedding_extraction_config.data.scaler == 'min-max':
            self.data_scaler = DataScaler(gait_embedding_extraction_config)
            self.data_scaler.load_scaler()
        elif gait_embedding_extraction_config.data.scaler == 'None':
            self.data_scaler = None
        else:
            raise ValueError("Unknown scaler type! \n")

    def _prepare_keypoints_sequence(self, keypoints_sequence):
        # print("Shape of keypoints_sequence:", keypoints_sequence.shape)
        
        if self.data_scaler != None:
            scaled_keypoints_sequence = self.data_scaler.scaling(keypoints_sequence)

        # Rimuoviamo la prima dimensione usando np.squeeze
        scaled_keypoints_sequence = np.squeeze(scaled_keypoints_sequence, axis=0)

        # print("Shape of scaled_keypoints_sequence:", scaled_keypoints_sequence.shape)

        T, F = scaled_keypoints_sequence.shape

        if F == 17 * 3:
            np_keypoints_sequence = np_keypoints_sequence.reshape(T, 17, 3).astype(np.float32)  # Convert to (T, 17, 3)
        else:
            raise ValueError(f"Unexpected number of features: {F}. Expected 34 or 51.")

        # Convert sequences to float32
        np_keypoints_sequence = np.array(np_keypoints_sequence, dtype=np.float32)  # Convert sequences to float32

        # Convert sequence to torch tensor
        # torch_sequence = torch.from_numpy(np_keypoints_sequence).float()
        torch_sequence = torch.from_numpy(np_keypoints_sequence)

        torch_keypoints_sequence = torch_sequence.permute(2, 0, 1).contiguous()  # now (C, T, 17)
        
        return torch_keypoints_sequence

    def extract_embedding(self, keypoints_sequence):
        torch_keypoints_sequence = self._prepare_keypoints_sequence(keypoints_sequence)

        torch_keypoints_sequence = torch_keypoints_sequence.to(self.device)

        # print("Shape of torch_keypoints_sequence:", torch_keypoints_sequence.shape)

        self.model.eval()  # Imposta il modello in modalità di valutazione

        # Effettua la predizione
        with torch.no_grad():
            embedding = self.model(torch_keypoints_sequence)

        # Rimuove la dimensione del batch e converte in numpy
        embedding = embedding.cpu().numpy()

        return embedding