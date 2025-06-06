import os
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# class DataScaler:
#     """
#     Provides functionality to scale datasets using MinMax or Standard scaling.

#     Attributes
#     ----------
#     X_train : array-like
#         Training data set.
#     X_test : array-like
#         Testing data set.
#     X_val : array-like
#         Validation data set.

#     Methods
#     -------
#     min_max_scaler():
#         Applies MinMax scaling to the datasets.
#     standard_scaler():
#         Applies Standard scaling to the datasets.
#     """

#     def __init__(self, gait_config, X_train, X_test, X_val):
#         """
#         Initializes the DataScaler with training, testing, and validation datasets.

#         Parameters
#         ----------
#         X_train : array-like
#             The training dataset to be scaled.
#         X_test : array-like
#             The testing dataset to be scaled.
#         X_val : array-like
#             The validation dataset to be scaled.
#         """

#         self._config = gait_config
#         self.X_train = X_train
#         self.X_test = X_test
#         self.X_val = X_val

#     def _flatten_data(self, data):
#         """
#         Flattens a 3D array into a 2D array for scaling.

#         Parameters
#         ----------
#         data : array-like
#             The data to be flattened.

#         Returns
#         -------
#         array-like
#             The flattened data.
#         """

#         num_samples, seq_length, num_features = data.shape

#         return data.reshape(num_samples * seq_length, num_features)

#     def _reshape_data(self, data, shape):
#         """
#         Reshapes a 2D array back to its original 3D shape after scaling.

#         Parameters
#         ----------
#         data : array-like
#             The data to be reshaped.
#         shape : tuple
#             The original shape of the dataset.

#         Returns
#         -------
#         array-like
#             The reshaped data.
#         """

#         return data.reshape(shape)

#     def _save_scaler(self, scaler):
#         """
# 		Salva un modello addestrato in un file.
# 		"""

# 		# Crea la directory se non esiste
#         os.makedirs(self._gait_config.data.scaler_dir, exist_ok=True)

#         model_path = os.path.join(self._gait_config.data.scaler_dir, f"gait_{self._gait_config.data.scaler}_scaler.pkl")
#         with open(model_path, 'wb') as f:
#             pickle.dump(scaler, f)  # Salva con Pickle

#         print(f"Gait {self._gait_config.data.scaler} scaler saved as {model_path}.")

#     def _load_scaler(self):
# 		# Load LDA model using pickle
#         scaler = pickle.load(open(f"{self._gait_config.data.scaler_dir}/gait_{self._gait_config.data.scaler}_scaler.pkl", 'rb'))

#         print(f"Gait {self._gait_config.data.scaler} scaler loaded.")
            
#         return scaler

#     def min_max_scaler(self):
#         """
#         Scales the datasets using MinMax scaling to a range of (-1, 1).

#         Returns
#         -------
#         tuple
#             A tuple containing the scaled training, testing, and validation datasets.

#         Example
#         -------
#         # Example usage:
#         >>> scaler = DataScaler(X_train, X_test, X_val)
#         >>> X_train_scaled, X_test_scaled, X_val_scaled = scaler.min_max_scaler()
#         """

#         # Activation functions such as tanh, which maps input in a range of (-1, 1), might be beneficial to use (-1, 1) as a range of features for your scaler
#         scaler = MinMaxScaler(feature_range=(-1, 1))

#         original_train_shape = self.X_train.shape
#         original_test_shape = self.X_test.shape
#         original_val_shape = self.X_val.shape

#         # Flatten data
#         X_train_flat = self._flatten_data(self.X_train)
#         X_test_flat = self._flatten_data(self.X_test)
#         X_val_flat = self._flatten_data(self.X_val)

#         # Scale data
#         X_train_normalized = scaler.fit_transform(X_train_flat)
#         X_test_normalized = scaler.transform(X_test_flat)
#         X_val_normalized = scaler.transform(X_val_flat)

#         # Reshape data back to original shape
#         X_train_normalized = self._reshape_data(X_train_normalized, original_train_shape)
#         X_test_normalized = self._reshape_data(X_test_normalized, original_test_shape)
#         X_val_normalized = self._reshape_data(X_val_normalized, original_val_shape)

#         self._save_scaler(scaler)

#         return X_train_normalized, X_test_normalized, X_val_normalized
        
    # def standard_scaler(self):
    #     """
    #     Scales the datasets using Standard scaling, adjusting for mean and standard deviation.

    #     Returns
    #     -------
    #     tuple
    #         A tuple containing the scaled training, testing, and validation datasets.

    #     Example
    #     -------
    #     # Example usage:
    #     >>> scaler = DataScaler(X_train, X_test, X_val)
    #     >>> X_train_scaled, X_test_scaled, X_val_scaled = scaler.standard_scaler()
    #     """

    #     scaler = StandardScaler()

    #     original_train_shape = self.X_train.shape
    #     original_test_shape = self.X_test.shape
    #     original_val_shape = self.X_val.shape

    #     # Flatten data
    #     X_train_flat = self._flatten_data(self.X_train)
    #     X_test_flat = self._flatten_data(self.X_test)
    #     X_val_flat = self._flatten_data(self.X_val)

    #     # Scale data
    #     X_train_standardized = scaler.fit_transform(X_train_flat)
    #     X_test_standardized = scaler.transform(X_test_flat)
    #     X_val_standardized = scaler.transform(X_val_flat)

    #     # Reshape data back to original shape
    #     X_train_standardized = self._reshape_data(X_train_standardized, original_train_shape)
    #     X_test_standardized = self._reshape_data(X_test_standardized, original_test_shape)
    #     X_val_standardized = self._reshape_data(X_val_standardized, original_val_shape)

    #     self._save_scaler(scaler)

    #     return X_train_standardized, X_test_standardized, X_val_standardized
    



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

        num_samples, seq_length, num_features = data.shape

        return data.reshape(num_samples * seq_length, num_features)

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

    # def fit_scaler(self, X_train, X_val, X_test):
    #     """
    #     Scales the features using standard scaling.
    #     """

    #     original_train_shape = X_train.shape
    #     original_val_shape = X_val.shape
    #     original_test_shape = X_test.shape

    #     # Flatten data
    #     X_train_flat = self._flatten_data(X_train)
    #     X_val_flat = self._flatten_data(X_val)
    #     X_test_flat = self._flatten_data(X_test)
    #     print("Shape di X_train:", X_train_flat.shape)

    #     # Scale data
    #     X_train_scaled = self._scaler.fit_transform(X_train_flat)
    #     X_val_scaled = self._scaler.transform(X_val_flat)
    #     X_test_scaled = self._scaler.transform(X_test_flat)
    #     print("Shape di X_train standardized:", X_train_scaled.shape)

    #     # Reshape data back to original shape
    #     X_train_scaled = self._reshape_data(X_train_scaled, original_train_shape)
    #     X_val_scaled = self._reshape_data(X_val_scaled, original_val_shape)
    #     X_test_scaled = self._reshape_data(X_test_scaled, original_test_shape)
    #     print("Shape di X_train standardized after reshape:", X_train_scaled.shape)

    #     self._save_scaler()

    #     return X_train_scaled, X_val_scaled, X_test_scaled

    def fit_scaler(self, X_train):
        """
        Scales the features.
        """
        # Flatten train data
        X_train_flat = self._flatten_data(X_train)

        # Fit scaler
        self._scaler.fit(X_train_flat)

        self._save_scaler()

        # processed_templates = []

        # for template in templates:
        #     processed_templates.append(template.flatten())
        #     # if len(template.shape) == 1:  # Se è un array 1D (face template)
        #     #     processed_templates.append(template.reshape(1, -1))  # Rendi 2D con shape (1, 640)
        #     # else:  # Se è un array 2D (iris templates)
        #     #     processed_templates.append(template.flatten().reshape(1, -1))  # Appiattisci in (1, 8192)

        # # Convertire la lista in un array NumPy 2D
        # # combined_templates_array = np.vstack(processed_templates)
        # combined_templates_array = np.array(processed_templates)
        
        # print("Forma di templates dopo del np.vstack:", combined_templates_array.shape)
        # print("Numero di sample (righe):", combined_templates_array.shape[0])
        # print("Numero di feature (colonne):", combined_templates_array.shape[1])

        # self.scaler.fit(combined_templates_array)

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
        

    # def scaling(self, template):
    #     if self.scaler_type == 'standard' or self.scaler_type == 'min-max':
    #         if len(template.shape) == 1:
    #             template = template.reshape(1, -1)  # Assicura che sia 2D
    #         else:
    #             template = template.flatten().reshape(1, -1)  # Appiattisci e converti in 2D

    #         return self.scaler.transform(template)
    #     elif self.scaler_type == 'None':
    #         return template.flatten()
    #     else:
    #         raise ValueError("Unknown scaler type! \n")