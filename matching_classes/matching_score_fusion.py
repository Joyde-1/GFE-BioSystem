import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MatchingScoreFusion:
    def __init__(self, multimodal_config):
        self.multimodal_config = multimodal_config
    
    def _cosine_similarity(self, template1, template2):
        return cosine_similarity(template1.reshape(1, -1), template2.reshape(1, -1))[0][0]

    def _euclidean_distance(self, template1, template2):
        return np.linalg.norm(template1 - template2)

    def compare_templates(self, gait_template1, face_template1, ear_template1,
                            gait_template2, face_template2, ear_template2):
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
        gait_score = self._euclidean_distance(gait_template1, gait_template2)
        face_score = self._euclidean_distance(face_template1, face_template2)
        ear_score = self._euclidean_distance(ear_template1, ear_template2)

        weighted_gait_score = gait_score * self.multimodal_config.score_fusion.weight_gait
        weighted_face_score = face_score * self.multimodal_config.score_fusion.weight_face
        weighted_ear_score = ear_score * self.multimodal_config.score_fusion.weight_ear

        fused_score = weighted_gait_score + weighted_face_score + weighted_ear_score

        return fused_score