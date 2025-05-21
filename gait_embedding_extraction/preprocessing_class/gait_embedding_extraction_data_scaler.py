from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataScaler:
    """
    Provides functionality to scale datasets using MinMax or Standard scaling.

    Attributes
    ----------
    X_train : array-like
        Training data set.
    X_test : array-like
        Testing data set.
    X_val : array-like
        Validation data set.

    Methods
    -------
    min_max_scaler():
        Applies MinMax scaling to the datasets.
    standard_scaler():
        Applies Standard scaling to the datasets.
    """

    def __init__(self, X_train, X_test, X_val):
        """
        Initializes the DataScaler with training, testing, and validation datasets.

        Parameters
        ----------
        X_train : array-like
            The training dataset to be scaled.
        X_test : array-like
            The testing dataset to be scaled.
        X_val : array-like
            The validation dataset to be scaled.
        """

        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val

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

    def min_max_scaler(self):
        """
        Scales the datasets using MinMax scaling to a range of (-1, 1).

        Returns
        -------
        tuple
            A tuple containing the scaled training, testing, and validation datasets.

        Example
        -------
        # Example usage:
        >>> scaler = DataScaler(X_train, X_test, X_val)
        >>> X_train_scaled, X_test_scaled, X_val_scaled = scaler.min_max_scaler()
        """

        # Activation functions such as tanh, which maps input in a range of (-1, 1), might be beneficial to use (-1, 1) as a range of features for your scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))

        original_train_shape = self.X_train.shape
        original_test_shape = self.X_test.shape
        original_val_shape = self.X_val.shape

        # Flatten data
        X_train_flat = self._flatten_data(self.X_train)
        X_test_flat = self._flatten_data(self.X_test)
        X_val_flat = self._flatten_data(self.X_val)

        # Scale data
        X_train_normalized = scaler.fit_transform(X_train_flat)
        X_test_normalized = scaler.transform(X_test_flat)
        X_val_normalized = scaler.transform(X_val_flat)

        # Reshape data back to original shape
        X_train_normalized = self._reshape_data(X_train_normalized, original_train_shape)
        X_test_normalized = self._reshape_data(X_test_normalized, original_test_shape)
        X_val_normalized = self._reshape_data(X_val_normalized, original_val_shape)

        return X_train_normalized, X_test_normalized, X_val_normalized
        
    def standard_scaler(self):
        """
        Scales the datasets using Standard scaling, adjusting for mean and standard deviation.

        Returns
        -------
        tuple
            A tuple containing the scaled training, testing, and validation datasets.

        Example
        -------
        # Example usage:
        >>> scaler = DataScaler(X_train, X_test, X_val)
        >>> X_train_scaled, X_test_scaled, X_val_scaled = scaler.standard_scaler()
        """

        scaler = StandardScaler()

        original_train_shape = self.X_train.shape
        original_test_shape = self.X_test.shape
        original_val_shape = self.X_val.shape

        # Flatten data
        X_train_flat = self._flatten_data(self.X_train)
        X_test_flat = self._flatten_data(self.X_test)
        X_val_flat = self._flatten_data(self.X_val)

        # Scale data
        X_train_standardized = scaler.fit_transform(X_train_flat)
        X_test_standardized = scaler.transform(X_test_flat)
        X_val_standardized = scaler.transform(X_val_flat)

        # Reshape data back to original shape
        X_train_standardized = self._reshape_data(X_train_standardized, original_train_shape)
        X_test_standardized = self._reshape_data(X_test_standardized, original_test_shape)
        X_val_standardized = self._reshape_data(X_val_standardized, original_val_shape)

        return X_train_standardized, X_test_standardized, X_val_standardized