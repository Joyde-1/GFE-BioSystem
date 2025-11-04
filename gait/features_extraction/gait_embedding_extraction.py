# Standard library imports
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import torch
import torch.nn.functional as F
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
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        print("Model loaded successfully!")

    def _prepare_predict_process(self):
        # Load configuration
        gait_embedding_extraction_config = load_config('gait_embedding_extraction/config/gait_embedding_extraction_config.yaml')

        # Set device
        self.device = select_device(gait_embedding_extraction_config)

        # Select model
        self.model = select_model(gait_embedding_extraction_config, self.device)

        # Load model weights
        self._load_model(gait_embedding_extraction_config)

        self.model.to(self.device)

        if gait_embedding_extraction_config.data.scaler == 'standard' or gait_embedding_extraction_config.data.scaler == 'min-max':
            self.data_scaler = DataScaler(gait_embedding_extraction_config)
            self.data_scaler.load_scaler()
        elif gait_embedding_extraction_config.data.scaler == 'None':
            self.data_scaler = None
        else:
            raise ValueError("Unknown scaler type! \n")

    def _prepare_keypoints_sequence(self, keypoints_sequence):
        # print("Shape of keypoints_sequence:", keypoints_sequence.shape)

        T, V, C = keypoints_sequence.shape  # (T, 12, 2)
        
        if self.data_scaler != None:
            scaled_keypoints_sequence = self.data_scaler.scaling(keypoints_sequence.reshape(1, T, V, C))
            # Rimuoviamo la prima dimensione usando np.squeeze
            scaled_keypoints_sequence = np.squeeze(scaled_keypoints_sequence, axis=0)  # (1, T, 12, 2) --> (T, 12, 2)
        else:
            scaled_keypoints_sequence = keypoints_sequence

        # Convert sequences to float32
        np_keypoints_sequence   = np.asarray(scaled_keypoints_sequence, dtype=np.float32)   # (L, J, 2)

        # Convert sequence to torch tensor
        torch_keypoints_sequence = torch.from_numpy(np_keypoints_sequence)

        torch_keypoints_sequence = torch_keypoints_sequence.view(1, T, V * C)
        
        return torch_keypoints_sequence

    def extract_embedding(self, keypoints_sequence):
        torch_keypoints_sequence = self._prepare_keypoints_sequence(keypoints_sequence)

        torch_keypoints_sequence = torch_keypoints_sequence.to(self.device)

        # print("Shape of torch_keypoints_sequence:", torch_keypoints_sequence.shape)

        self.model.eval()  # Imposta il modello in modalità di valutazione

        # Effettua la predizione
        with torch.no_grad():
            embedding, _, _ = self.model(torch_keypoints_sequence)

        # Rimuove la dimensione del batch e converte in numpy
        # embedding = embedding.cpu().numpy().reshape(-1)
        embedding = F.normalize(embedding, dim=1).cpu().numpy().reshape(-1)

        # print("Shape of embedding:", embedding.shape)

        return embedding