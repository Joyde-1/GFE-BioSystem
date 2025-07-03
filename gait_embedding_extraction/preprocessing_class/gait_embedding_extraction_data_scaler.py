import os
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataScaler:

    def __init__(self, gait_config):
        self._gait_config = gait_config

        if gait_config.data.scaler == 'standard':
            self._scaler = StandardScaler()
        elif gait_config.data.scaler == 'min-max':
            self._scaler = MinMaxScaler()  # Mappatura in un range di (-1, 1)
        elif gait_config.data.scaler == 'None':
            pass
        else:
            raise ValueError("Unknown scaler type! \n")
        
    def _flatten_data(self, data):
        """
        Flattens a 3D array into a 2D array for scaling.

        Parameters
        ----------
        data : array-like
            The data to be flattened.

        Returns
        -------
        array-like
            The flattened data.
        """

        # Salvo le dimensioni originali
        # num_samples, seq_length, num_features = data.shape
        num_samples, seq_length, num_keypoints, coords_and_confidence = data.shape

        # return data.reshape(num_samples * seq_length, num_features)
        return data.reshape(num_samples * seq_length, num_keypoints * coords_and_confidence)

    def _reshape_data(self, data, shape):
        """
        Reshapes a 2D array back to its original 3D shape after scaling.

        Parameters
        ----------
        data : array-like
            The data to be reshaped.
        shape : tuple
            The original shape of the dataset.

        Returns
        -------
        array-like
            The reshaped data.
        """

        return data.reshape(shape)
    
    def _save_scaler(self):
        """
		Salva un modello addestrato in un file.
		"""

		# Crea la directory se non esiste
        os.makedirs(self._gait_config.data.scaler_dir, exist_ok=True)

        model_path = os.path.join(self._gait_config.data.scaler_dir, f"gait_{self._gait_config.data.scaler}_scaler.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self._scaler, f)  # Salva con Pickle

        print(f"Gait {self._gait_config.data.scaler} scaler saved as {model_path}.")

    def load_scaler(self):
		# Load scaler using pickle
        self._scaler = pickle.load(open(f"{self._gait_config.data.scaler_dir}/gait_{self._gait_config.data.scaler}_scaler.pkl", 'rb'))

        print(f"Gait {self._gait_config.data.scaler} scaler loaded.")

    def fit_scaler(self, X_train):
        """
        Scales the features.
        """
        # Flatten train data
        X_train_flat = self._flatten_data(X_train)

        # Fit scaler
        self._scaler.fit(X_train_flat)

        self._save_scaler()

    def scaling(self, data):
        original_data_shape = data.shape
        # print("Shape di data:", original_data_shape)

        # Flatten data
        data_flattened = self._flatten_data(data)
        # print("Shape di data after flattening:", data_flattened.shape)

        # Scale data
        data_scaled = self._scaler.transform(data_flattened)
        # print("Shape di data scaled:", data_scaled.shape)

        # Reshape data back to original shape
        data_scaled = self._reshape_data(data_scaled, original_data_shape)
        # print("Shape di data scaled after reshape:", data_scaled.shape)

        return data_scaled