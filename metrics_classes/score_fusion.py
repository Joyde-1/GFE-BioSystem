import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity


class MatchingScoreFusion:
    def __init__(self, multimodal_config):
        self.multimodal_config = multimodal_config
    
    def _cosine_similarity(self, template1, template2):
        return cosine_similarity(template1.reshape(1, -1), template2.reshape(1, -1))[0][0]

    def _euclidean_distance(self, template1, template2):
        return np.linalg.norm(template1 - template2)

    def _compare_templates(self, gait_template1, face_template1, ear_template1,
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
        
    def calculate_far(self, subjects_gait, subjects_face, subjects_ear):
        """
        Calcola il False Acceptance Rate (FAR).
        
        Per ogni soggetto, ogni acquisizione viene usata come query e confrontata con tutte le acquisizioni
        dei soggetti differenti. Se il sistema "accetta" erroneamente un impostore (cioè, riconosce come match
        due acquisizioni appartenenti a soggetti diversi), viene conteggiata una false acceptance.
        
        Ritorna:
        - FAR: false acceptance rate (FA / T_imp)
        - FA: numero totale di false acceptances
        - T_imp: numero totale di tentativi di impostori
        """
        FA = 0        # Contatore per le false acceptances
        T_imp = 0     # Contatore per il totale dei tentativi di impostori

        # Opening the text file to save the matching results
        with open(self.multimodal_config.results_path + f"/multimodal_score_fusion_results_FAR", "w") as file:
            # Per ogni soggetto (che sarà la query)
            for subject_gait, subject_face, subject_ear in zip(subjects_gait, subjects_face, subjects_ear):
                gait_templates = subjects_gait[subject_gait]['template']
                face_templates = subjects_face[subject_face]['template']
                ear_templates = subjects_ear[subject_ear]['template']
                num_acquisizioni = len(face_templates)
                
                file.write(f"Matching score between {subject_gait} vs ALL: \n")
                
                # Ogni acquisizione del soggetto viene usata come query
                for i in range(num_acquisizioni):
                    query_gait_template = gait_templates[i]
                    query_face_template = face_templates[i]
                    query_ear_template = ear_templates[i]
                    
                    # Confronta la query con tutte le acquisizioni dei soggetti differenti
                    for other_subject_gait, other_subject_face, other_subject_ear in zip(subjects_gait, subjects_face, subjects_ear):
                        if other_subject_gait == subject_gait and other_subject_face == subject_face and other_subject_ear == subject_ear:
                            continue  # Salta lo stesso soggetto
                        
                        other_gait_templates = subjects_gait[other_subject_gait]['template']
                        other_face_templates = subjects_face[other_subject_face]['template']
                        other_ear_templates = subjects_ear[other_subject_ear]['template']

                        for j in range(len(other_gait_templates)):
                            T_imp += 1
                            # Se il sistema considera le due acquisizioni come appartenenti allo stesso soggetto,
                            # allora l'impostore viene accettato erroneamente: conteggiamo una false acceptance.
                            matching_score = self._compare_templates(query_gait_template, query_face_template, query_ear_template,
                                                                     other_gait_templates[j], other_face_templates[j], other_ear_templates[j])

                            if matching_score < self.multimodal_config.score_fusion.threshold:
                                FA += 1

                            # Print the results of the matching process in a text file
                            file.write(f"Acquisition {subjects_gait[subject_gait]['acquisition_name'][i]} vs {subjects_gait[other_subject_gait]['acquisition_name'][j]} \n")
                            file.write(f"Matching score: {matching_score} \n")

                        file.write("\n\n")
                file.write("\n\n\n")
                    
            FAR = (FA / T_imp) * 100 if T_imp > 0 else 0

        return FAR, FA, T_imp

    def calculate_frr(self, subjects_gait, subjects_face, subjects_ear):
        """
        Calcola la False Rejection Rate (FRR).
        
        Per ogni soggetto, vengono confrontate le acquisizioni in coppie uniche.
        Se il confronto tra due acquisizioni dello stesso soggetto restituisce False,
        viene considerato un false rejection.
        
        Ritorna:
        - FRR: false rejection rate (FR / T_legit)
        - FR: numero totale di false rejections
        - T_legit: numero totale di tentativi genuini unici
        """
        FR = 0        # Conteggio dei false rejections
        T_legit = 0   # Conteggio totale dei tentativi genuini (coppie uniche)

        # Opening the text file to save the matching results
        with open(self.multimodal_config.results_path + f"/multimodal_score_fusion_results_FRR", "w") as file:
            # Per ogni soggetto
            for subject_gait, subject_face, subject_ear in zip(subjects_gait, subjects_face, subjects_ear):
                gait_templates = subjects_gait[subject_gait]['template']
                face_templates = subjects_face[subject_face]['template']
                ear_templates = subjects_ear[subject_ear]['template']
                K = len(gait_templates)  # Numero di acquisizioni per il soggetto
                
                file.write(f"Matching score between {subject_gait} vs ALL: \n")

                # Effettuiamo confronti unici: per ogni acquisizione i, confronta solo con acquisizioni successive (j > i)
                for i in range(K):
                    for j in range(i + 1, K):
                        T_legit += 1
                        # Se il sistema NON riconosce le due acquisizioni come appartenenti allo stesso soggetto,
                        # allora abbiamo un false rejection.
                        matching_score = self._compare_templates(gait_templates[i], face_templates[i], ear_templates[i],
                                                                 gait_templates[j], face_templates[j], ear_templates[j])

                        if matching_score > self.multimodal_config.score_fusion.threshold:
                            FR += 1

                        # Print the results of the matching process in a text file
                        file.write(f"Acquisition {subjects_face[subject_gait]['acquisition_name'][i]} vs {subjects_gait[subject_gait]['acquisition_name'][j]} \n")
                        file.write(f"Matching score: {matching_score} \n")

                    file.write("\n\n")
                file.write("\n\n\n")

        FRR = (FR / T_legit) * 100 if T_legit > 0 else 0

        return FRR, FR, T_legit
    
    def calculate_accuracy(self, T_imp, T_legit, FA, FR):
        return (((T_imp + T_legit) - FA - FR) / (T_imp + T_legit)) * 100
    
    def calculate_roc_and_det(self, subjects_gait, subjects_face, subjects_ear):
        """
        Calcola e disegna la ROC curve in cui:
        - l'asse orizzontale rappresenta il FAR (%) in scala logaritmica
        - l'asse verticale rappresenta il GAR (%) in scala logaritmica

        Procedura:
        1. Raccoglie i matching score relativi a:
            - confronti genuini (stesso soggetto, etichettati come '1')
            - confronti impostori (soggetti differenti, etichettati come '0')
        2. Poiché chi-square più basso indica maggiore similarità,
            inverte i punteggi (usando -score) per adattarli alla convenzione in cui valori più alti
            indicano una maggiore probabilità di corrispondenza genuina.
        3. Calcola la curva ROC con roc_curve di scikit-learn.
        4. Converte i valori in percentuale, imposta la scala logaritmica e plottali.
        """
        scores = []
        labels = []

        # Raccolta dei punteggi per confronti genuini (stesso soggetto)
        for subject_gait, subject_face, subject_ear in zip(subjects_gait, subjects_face, subjects_ear):
            gait_templates = subjects_gait[subject_gait]['template']
            face_templates = subjects_face[subject_face]['template']
            ear_templates = subjects_ear[subject_ear]['template']
            num_acquisizioni = len(gait_templates)
            for i in range(num_acquisizioni):
                for j in range(i + 1, num_acquisizioni):
                    score = self._compare_templates(gait_templates[i], face_templates[i], ear_templates[i],
                                                    gait_templates[j], face_templates[j], ear_templates[j])
                    if score is not None:
                        scores.append(score)
                        labels.append(1)  # confronto genuino

        # Raccolta dei punteggi per confronti impostori (soggetti differenti)
        for subject_gait, subject_face, subject_ear in zip(subjects_gait, subjects_face, subjects_ear):
            query_gait_templates = subjects_gait[subject_gait]['template']
            query_face_templates = subjects_face[subject_face]['template']
            query_ear_templates = subjects_ear[subject_ear]['template']
            for i in range(len(query_face_templates)):
                for other_subject_gait, other_subject_face, other_subject_ear in zip(subjects_gait, subjects_face, subjects_ear):
                    if other_subject_gait == subject_gait and other_subject_face == subject_face and other_subject_ear == subject_ear:
                        continue
                    other_gait_templates = subjects_gait[other_subject_gait]['template']
                    other_face_templates = subjects_face[other_subject_face]['template']
                    other_ear_templates = subjects_ear[other_subject_ear]['template']
                    for j in range(len(other_gait_templates)):
                        score = self._compare_templates(query_gait_templates[i], query_face_templates[i], query_ear_templates[i],
                                                        other_gait_templates[j], other_face_templates[j], other_ear_templates[j])
                        if score is not None:
                            scores.append(score)
                            labels.append(0)  # confronto impostore

        # Invertiamo i punteggi per far corrispondere valori più alti a una maggiore similarità
        inverted_scores = [-s for s in scores]

        # Calcolo della curva ROC
        far, gar, thresholds = roc_curve(labels, inverted_scores, pos_label=1)
        
        # Conversione in percentuale
        FAR_pct = far * 100
        GAR_pct = gar * 100
        FRR_pct = (1 - gar) * 100          # FRR (%) = 100 - GAR (%)          

        # Disegno del grafico ROC con assi in scala logaritmica
        plt.figure()
        plt.plot(FAR_pct, GAR_pct, color='blue', lw=2, marker='o')
        plt.xlabel("FAR (%)")
        plt.ylabel("GAR (%)")
        plt.title(f"ROC Curve")
        plt.grid(True, which="both", ls="--")
        
        # Imposta la scala logaritmica
        plt.xscale("log")
        plt.xlim(1e-3, 1e2)
        
        plt.savefig(self.multimodal_config.results_path + f"/multimodal_score_fusion_ROC.png")
        plt.close()

        # Individua l'indice in cui la differenza tra FAR e FRR è minima (punto di EER)
        diff = np.abs(FAR_pct - FRR_pct)
        EER_index = np.argmin(diff)
        EER_pct = (FAR_pct[EER_index] + FRR_pct[EER_index]) / 2.0  # EER approssimato

        # Disegno del grafico DET
        plt.figure()
        plt.plot(FAR_pct, FRR_pct, color='blue', lw=2, marker='o', label='DET Curve')
        plt.xlabel("FAR (%)")
        plt.ylabel("FRR (%)")
        plt.title(f"DET Curve with EER")
        plt.grid(True, which="both", ls="--")
        
        # Traccia il punto EER
        plt.plot(EER_pct, EER_pct, marker='o', color='red', markersize=8, label=f'EER = {EER_pct:.2f}%')
        
        # Traccia linee guida orizzontali e verticali dal punto EER
        plt.axhline(y=EER_pct, color='red', linestyle='--')
        plt.axvline(x=EER_pct, color='red', linestyle='--')
        
        plt.legend(loc='best')
        
        # Salva il grafico
        plt.savefig(self.multimodal_config.results_path + f"/multimodal_score_fusion_DET.png")
        plt.close()

        return FAR_pct, GAR_pct, FRR_pct, EER_pct, thresholds

    def far_vs_frr(self, subjects_gait, subjects_face, subjects_ear):
        # Definisci un range di valori di threshold su cui effettuare il test
        # Ad esempio, varia da 0.1 a 1.0 (puoi modificare i limiti e il numero di passi in base alle tue esigenze)
        thresholds = np.linspace(0.1, 100.0, num=200)

        # Liste per memorizzare FAR e FRR per ogni valore di threshold
        far_values = []
        frr_values = []

        original_threshold = self.multimodal_config.score_fusion.threshold

        # Loop sul range di threshold
        for th in thresholds:
            # Aggiorna la threshold nel file di configurazione
            self.multimodal_config.score_fusion.threshold = th
            
            # Calcola FAR e FRR sui soggetti (il metodo scrive anche su file, seppur tu possa ignorare questo aspetto per il grafico)
            far, _, _ = self.calculate_far(subjects_gait, subjects_face, subjects_ear)
            frr, _, _ = self.calculate_frr(subjects_gait, subjects_face, subjects_ear)
            
            far_values.append(far)
            frr_values.append(frr)
            
            print(f"Threshold: {th:.4f} --> FAR: {far:.2f}% , FRR: {frr:.2f}%")

        self.multimodal_config.score_fusion.threshold = original_threshold

        # Creazione del grafico
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, far_values, marker='o', label="FAR (%)")
        plt.plot(thresholds, frr_values, marker='s', label="FRR (%)")
        plt.xlabel("Threshold")
        plt.ylabel("Error Rate (%)")
        plt.title("Variazione di FAR e FRR al variare della Threshold")
        plt.legend()
        plt.grid(True)
        
        # Salva il grafico
        plt.savefig(self.multimodal_config.results_path + f"/multimodal_score_fusion_far_vs_frr.png")
        plt.close()