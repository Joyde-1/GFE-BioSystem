import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Matching:
    def __init__(self, config):
        self._config = config

    def _euclidean_distance(self, template1, template2):
        return np.linalg.norm(template1 - template2)

    def _chi_square(self, template1, template2):
        """
        Confronta due template biometrici e restituisce una distanza.
        
        Parametri:
            template1 (np.array): primo template (vettore).
            template2 (np.array): secondo template (vettore).
        Ritorna:
            float: distanza (score) tra i due template. Valori minori indicano una maggiore somiglianza.
        """
        eps = 1e-10

        return 0.5 * np.sum(((template1 - template2) ** 2) / (template1 + template2 + eps))
    
    def _hamming_distance(self, template1, template2):
        """
        Confronta due template biometrici e restituisce una distanza.
        
        Parametri:
            template1 (np.array): primo template (vettore).
            template2 (np.array): secondo template (vettore).
        Ritorna:
            float: distanza (score) tra i due template. Valori minori indicano una maggiore somiglianza.
        """
        # Assumendo che template1 e template2 siano vettori di bit 0/1
        # Normalizziamo per la lunghezza
        return np.sum(template1 != template2) / template1.size
    
    def _cosine_similarity(self, template1, template2):
        # assicuriamoci che siano in forma (1, d)
        template1 = template1.reshape(1, -1)
        template2 = template2.reshape(1, -1)

        # restituisce una matrice 1×1
        sim_matrix = cosine_similarity(template1, template2)
        score = sim_matrix[0, 0]

        return score

    def _cosine_distance(self, template1, template2):
        """
        Calcola la distanza coseno tra due embedding.
        Restituisce un valore tra 0 e 2, dove 0 = identici, 2 = opposti.
        """
        # Normalizza i vettori
        template1_norm = template1 / (np.linalg.norm(template1) + 1e-8)
        template2_norm = template2 / (np.linalg.norm(template2) + 1e-8)
        
        # Calcola similarità coseno
        cosine_sim = np.dot(template1_norm, template2_norm)
        
        # Converte in distanza (0 = identici, 2 = opposti)
        cosine_distance = 1 - cosine_sim
        
        return cosine_distance

    def compare_templates(self, template1, template2):
        """
        Confronta due template biometrici e restituisce una distanza.
        
        Parametri:
            template1 (np.array): primo template (vettore).
            template2 (np.array): secondo template (vettore).
            method (str): metodo di confronto. Default 'chi-square'.
                        Altri metodi possono essere implementati se necessario.
                        
        Ritorna:
            float: distanza (score) tra i due template. Valori minori indicano una maggiore somiglianza.
        """
        if self._config.matching_algorithm == 'chi-square':
            return self._chi_square(template1, template2)
        elif self._config.matching_algorithm == 'hamming':
            return self._hamming_distance(template1, template2)
        elif self._config.matching_algorithm == 'euclidean':
            return self._euclidean_distance(template1, template2)
        elif self._config.matching_algorithm == 'cosine_similarity':
            return self._cosine_similarity(template1, template2)
        elif self._config.matching_algorithm == 'cosine_distance':
            return self._cosine_distance(template1, template2)
        else:
            raise ValueError("Matching algorithm not supported.")