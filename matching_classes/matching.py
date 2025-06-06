import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_similarity


class Matching:
    def __init__(self, config, biometric_trait):
        self._config = config
        self.biometric_trait = biometric_trait

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
        return cosine_similarity(template1.reshape(1, -1), template2.reshape(1, -1))[0][0]

    def _compare_templates(self, template1, template2):
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
        else:
            raise ValueError("Matching algorithm not supported.")
        
    def calculate_far(self, subjects):
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
        with open(self._config.results_path + f"/{self.biometric_trait}_results_FAR", "w") as file:
            # Per ogni soggetto (che sarà la query)
            for subject in subjects:
                templates = subjects[subject]['template']
                num_acquisizioni = len(templates)
                
                file.write(f"Matching score between {subject} vs ALL: \n")
                
                # Ogni acquisizione del soggetto viene usata come query
                for i in range(num_acquisizioni):
                    query_template = templates[i]
                    
                    # Confronta la query con tutte le acquisizioni dei soggetti differenti
                    for other_subject in subjects:
                        if other_subject == subject:
                            continue  # Salta lo stesso soggetto
                        
                        other_templates = subjects[other_subject]['template']

                        for j in range(len(other_templates)):
                            T_imp += 1
                            # Se il sistema considera le due acquisizioni come appartenenti allo stesso soggetto,
                            # allora l'impostore viene accettato erroneamente: conteggiamo una false acceptance.
                            matching_score = self._compare_templates(query_template, other_templates[j])

                            if self._config == 'cosine_similarity':
                                if matching_score > self._config.matching.threshold:
                                    FA += 1
                            else:
                                if matching_score < self._config.matching.threshold:
                                    FA += 1

                            # Print the results of the matching process in a text file
                            file.write(f"Acquisition {subjects[subject]['acquisition_name'][i]} vs {subjects[other_subject]['acquisition_name'][j]} \n")
                            file.write(f"Matching score: {matching_score} \n")

                        file.write("\n\n")
                file.write("\n\n\n")
                    
            FAR = (FA / T_imp) * 100 if T_imp > 0 else 0

        return FAR, FA, T_imp

    def calculate_frr(self, subjects):
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
        with open(self._config.results_path + f"/{self.biometric_trait}_results_FRR", "w") as file:
            # Per ogni soggetto
            for subject in subjects:
                templates = subjects[subject]['template']
                K = len(templates)  # Numero di acquisizioni per il soggetto
                
                file.write(f"Matching score between {subject} vs ALL: \n")

                # Effettuiamo confronti unici: per ogni acquisizione i, confronta solo con acquisizioni successive (j > i)
                for i in range(K):
                    for j in range(i + 1, K):
                        T_legit += 1
                        # Se il sistema NON riconosce le due acquisizioni come appartenenti allo stesso soggetto,
                        # allora abbiamo un false rejection.
                        matching_score = self._compare_templates(templates[i], templates[j])

                        if self._config == 'cosine_similarity':
                            if matching_score < self._config.matching.threshold:
                                FR += 1
                        else:
                            if matching_score > self._config.matching.threshold:
                                FR += 1

                        # Print the results of the matching process in a text file
                        file.write(f"Acquisition {subjects[subject]['acquisition_name'][i]} vs {subjects[subject]['acquisition_name'][j]} \n")
                        file.write(f"Matching score: {matching_score} \n")

                    file.write("\n\n")
                file.write("\n\n\n")

        FRR = (FR / T_legit) * 100 if T_legit > 0 else 0

        return FRR, FR, T_legit
    
    def calculate_accuracy(self, T_imp, T_legit, FA, FR):
        return (((T_imp + T_legit) - FA - FR) / (T_imp + T_legit)) * 100
    
    def calculate_roc_and_det(self, subjects):
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
        for subject in subjects:
            templates = subjects[subject]['template']
            num_acquisizioni = len(templates)
            for i in range(num_acquisizioni):
                for j in range(i + 1, num_acquisizioni):
                    score = self._compare_templates(templates[i], templates[j])
                    if score is not None:
                        scores.append(score)
                        labels.append(1)  # confronto genuino

        # Raccolta dei punteggi per confronti impostori (soggetti differenti)
        for subject in subjects:
            query_templates = subjects[subject]['template']
            for i in range(len(query_templates)):
                for other_subject in subjects:
                    if other_subject == subject:
                        continue
                    other_templates = subjects[other_subject]['template']
                    for j in range(len(other_templates)):
                        score = self._compare_templates(query_templates[i], other_templates[j])
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
        
        plt.savefig(self._config.results_path + f"/{self.biometric_trait}_ROC.png")
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
        plt.savefig(self._config.results_path + f"/{self.biometric_trait}_DET.png")
        plt.close()

        return FAR_pct, GAR_pct, FRR_pct, EER_pct, thresholds

    def far_vs_frr(self, subjects):
        # Definisci un range di valori di threshold su cui effettuare il test
        if self._config.matching_algorithm == 'chi-square':
            thresholds = np.linspace(0.0, 100.0, num=200)
        elif self._config.matching_algorithm == 'hamming':
            thresholds = np.linspace(0.0, 1.0, num=200)
        elif self._config.matching_algorithm == 'euclidean':
            thresholds = np.linspace(1.0, 60.0, num=200)
        elif self._config.matching_algorithm == 'cosine_similarity':
            thresholds = np.linspace(-1.0, 1.0, num=200)
        else:
            raise ValueError("Matching algorithm not supported.")

        # Liste per memorizzare FAR e FRR per ogni valore di threshold
        far_values = []
        frr_values = []
        accuracy_values = []

        original_threshold = self._config.matching.threshold

        # Loop sul range di threshold
        for th in thresholds:
            # Aggiorna la threshold nel file di configurazione
            self._config.matching.threshold = th
            
            # Calcola FAR e FRR sui soggetti (il metodo scrive anche su file, seppur tu possa ignorare questo aspetto per il grafico)
            far, fa, t_imp = self.calculate_far(subjects)
            frr, fr, t_legit = self.calculate_frr(subjects)
            accuracy = self.calculate_accuracy(t_imp, t_legit, fa, fr)
            eer = (far + frr) / 2.0  # EER approssimato alla soglia corrente
            
            far_values.append(far)
            frr_values.append(frr)
            accuracy_values.append(accuracy)
            
            print(f"Threshold: {th:.4f} --> FAR: {far:.4f}% - FRR: {frr:.4f}% - Accuracy: {accuracy:.4f} - EER: {eer:.4f}")

        self._config.matching.threshold = original_threshold

        # Trova e stampa la migliore accuracy
        best_accuracy = max(accuracy_values)
        best_index = accuracy_values.index(best_accuracy)
        best_threshold = thresholds[best_index]
        best_far = far_values[best_index]
        best_frr = frr_values[best_index]

        print("", f"Best Accuracy: {best_accuracy:.4f}% at threshold {best_threshold:.4f}", f"FAR: {best_far:.4f}% - FRR: {best_frr:.4f}%", sep='\n')

        # Calcolo EER (Equal Error Rate)
        far_array = np.array(far_values)
        frr_array = np.array(frr_values)
        thresholds_array = np.array(thresholds)

        # Trova l'indice dove FAR e FRR sono più vicini
        diff = np.abs(far_array - frr_array)
        eer_index = np.argmin(diff)
        eer_threshold = thresholds_array[eer_index]
        eer = (far_array[eer_index] + frr_array[eer_index]) / 2.0

        print("", f"Best EER: {eer:.4f}% at threshold {eer_threshold:.4f}", "", sep='\n')

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
        plt.savefig(self._config.results_path + f"/{self.biometric_trait}_far_vs_frr.png")
        plt.close()