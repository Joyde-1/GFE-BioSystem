import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeaturesScaling():

    def __init__(self, scaler_type):
        self.scaler_type = scaler_type

        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'min-max':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'None':
            pass
        else:
            raise ValueError("Unknown scaler type! \n")

    def fit_scaler(self, templates):
        """
        Scales the features using standard scaling.
        """
        processed_templates = []

        for template in templates:
            processed_templates.append(template.flatten())
            # if len(template.shape) == 1:  # Se è un array 1D (face template)
            #     processed_templates.append(template.reshape(1, -1))  # Rendi 2D con shape (1, 640)
            # else:  # Se è un array 2D (iris templates)
            #     processed_templates.append(template.flatten().reshape(1, -1))  # Appiattisci in (1, 8192)

        # Convertire la lista in un array NumPy 2D
        # combined_templates_array = np.vstack(processed_templates)
        combined_templates_array = np.array(processed_templates)
        
        print("Forma di templates dopo del np.vstack:", combined_templates_array.shape)
        print("Numero di sample (righe):", combined_templates_array.shape[0])
        print("Numero di feature (colonne):", combined_templates_array.shape[1])

        self.scaler.fit(combined_templates_array)

    def scaling(self, template):
        if self.scaler_type == 'standard' or self.scaler_type == 'min-max':
            if len(template.shape) == 1:
                template = template.reshape(1, -1)  # Assicura che sia 2D
            else:
                template = template.flatten().reshape(1, -1)  # Appiattisci e converti in 2D

            return self.scaler.transform(template)
        elif self.scaler_type == 'None':
            return template.flatten()
        else:
            raise ValueError("Unknown scaler type! \n")