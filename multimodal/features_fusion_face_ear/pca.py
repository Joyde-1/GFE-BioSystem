import numpy as np
from sklearn.decomposition import PCA

from plots import plot_principal_components


class FeaturesFusionPCA():

    def __init__(self, multimodal_config):
        self.multimodal_config = multimodal_config

    def _run_pca(self, combined_templates, n_components=None):
        """
        Performs PCA on the scaled data and returns the transformed data or explained variance ratio.

        Parameters
        ----------
        n_components : int, optional
            The number of principal components to retain. If None, returns the explained variance ratio.

        Returns
        -------
        np.ndarray
            PCA-transformed data or explained variance ratio of all components.
        """
        if n_components != None:
            pca = PCA(n_components=n_components)

            # Apply PCA to determine the number of optimal major components
            principal_components = pca.fit_transform(combined_templates)

            return principal_components
        else:
            pca = PCA()
            
            # Apply PCA to determine the number of optimal major components
            pca.fit(combined_templates)

            return pca.explained_variance_ratio_
    
    # Combine features templates
    def weighted_concatenation(self, face_features, ear_features):
        # Applica i pesi
        weighted_face = face_features * self.multimodal_config.features_fusion.weight_face
        weighted_ear = ear_features * self.multimodal_config.features_fusion.weight_ear

        combined_features = np.concatenate((weighted_face, weighted_ear), axis=1)
        # print("Shape di combined_features:", combined_features.shape)  # Deve essere sempre (1, n_total_features)
        return combined_features

    def features_fusion_pca(self, combined_templates, file_suffix):
        """
        Determines the optimal number of principal components using PCA and reduces the data dimensionality accordingly.

        Returns
        -------

        Example
        -------
        # Example usage:
        >>> dim_reduction = DimensionalityReduction(images)
        >>> reduced_data = dim_reduction.reduce_data_with_pca()
        >>> print(reduced_data.head())
        """
        explained_variance_ratio = self._run_pca(combined_templates)
        # Stampiamo il numero massimo di componenti che possiamo usare
        # print(f"PCA può scegliere max {min(combined_templates.shape)} componenti")
        # Calculate the cumulative explained variance
        cumulative_variance_explained = np.cumsum(explained_variance_ratio)

        # Calculate the derivative according to the cumulative explained variance
        second_derivative = np.diff(cumulative_variance_explained, n=2)

        # Find the elbow point as the maximum of the second derivative
        optimal_components = np.argmax(second_derivative) + 1

        plot_principal_components(self.multimodal_config, cumulative_variance_explained, optimal_components, file_suffix)
        
        print("The value of optimal features obtained through PCA - Cumulative is: ", optimal_components)

        principal_components = self._run_pca(combined_templates, optimal_components)

        return principal_components