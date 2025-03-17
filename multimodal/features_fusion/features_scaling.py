import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeaturesScalingMultimodal():

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_scaler(self, templates):
        """
        Scales the features using standard scaling.
        """
        processed_templates = []

        for template in templates:
            if len(template.shape) == 1:  # Se è un array 1D (face template)
                processed_templates.append(template.reshape(1, -1))  # Rendi 2D con shape (1, 640)
            else:  # Se è un array 2D (iris templates)
                processed_templates.append(template.flatten().reshape(1, -1))  # Appiattisci in (1, 8192)

        # Convertire la lista in un array NumPy 2D
        combined_templates_array = np.vstack(processed_templates)
        
        print("Forma di templates dopo del np.vstack:", combined_templates_array.shape)
        print("Numero di sample (righe):", combined_templates_array.shape[0])
        print("Numero di feature (colonne):", combined_templates_array.shape[1])

        self.scaler.fit(combined_templates_array)

    def scaling(self, template):
        if len(template.shape) == 1:
            template = template.reshape(1, -1)  # Assicura che sia 2D
        else:
            template = template.flatten().reshape(1, -1)  # Appiattisci e converti in 2D
        return self.scaler.transform(template)