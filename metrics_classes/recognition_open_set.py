import time
import random
import numpy as np
import matplotlib.pyplot as plt

from matching_classes.matching import Matching


class RecognitionOpenSet:
    """
    Identificazione open-set 1 : N

    • split soggetti 80 % genuini / 20 % impostori  (seed = 42)
    • FPIR / FNIR per soglia fissata
    • Curva FPIR-FNIR (EER)         – grafico + valori min |FPIR-FNIR|
    • Curva DIR-FPIR                – stile NIST FRVT
    • Tempo di ricerca medio e throughput (probe/s)
    """

    # ------------------------------------------------------------------ #
    #  Init                                                              #
    # ------------------------------------------------------------------ #
    def __init__(self, config, biometric_trait="gait"):
        self._config = config
        self.biometric_trait = biometric_trait
        self.matcher = Matching(config)

    # ------------------------------------------------------------------ #
    #  Split 80 % / 20 %                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_subjects(subjects, impostor_ratio=0.2, seed=42):
        ids = list(subjects.keys())
        random.Random(seed).shuffle(ids)

        n_imp = max(1, int(len(ids) * impostor_ratio))
        impostor_ids = set(ids[:n_imp])

        gallery, probe_genuine, probe_imp = {}, {}, {}

        for sid, data in subjects.items():
            tmpl = data["template"]
            if sid in impostor_ids:
                # tutte le acquisizioni diventano probe impostori
                probe_imp[sid] = {"template": tmpl}
            else:
                # enrolment: usa tutti meno l’ultima come gallery
                if len(tmpl) < 2:
                    continue  # troppo pochi sample
                gallery[sid] = {"template": tmpl[:-1]}
                probe_genuine[sid] = {"template": [tmpl[-1]]}

        return gallery, probe_genuine, probe_imp

    # ------------------------------------------------------------------ #
    #  Utils                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _flatten(subj_dict):
        tmpls, ids = [], []
        for sid, data in subj_dict.items():
            for t in data["template"]:
                tmpls.append(t)
                ids.append(sid)
        return np.asarray(tmpls, dtype=np.float32), np.asarray(ids)

    def _descending(self):
        return self._config.matching_algorithm.lower().endswith("similarity")

    def _score_matrix(self, probes, gallery):
        """Restituisce matrice P×G **e** tempo totale."""
        P, G = len(probes), len(gallery)
        S = np.empty((P, G), dtype=np.float32)

        tic = time.perf_counter()
        for i in range(P):
            for j in range(G):
                S[i, j] = self.matcher.compare_templates(probes[i], gallery[j])
        elapsed = time.perf_counter() - tic
        return S, elapsed

    # ------------------------------------------------------------------ #
    #  FPIR / FNIR per soglia                                            #
    # ------------------------------------------------------------------ #
    def fpir_fnir(
        self,
        subjects,
        threshold,
        rank_k=1,
        impostor_ratio=0.2,
        seed=42,
    ):
        """
        Calcola FPIR e FNIR a una soglia specifica.

        Ritorna
        -------
        fpir, fnir     (percentuali)
        search_ms      (tempo medio ms / probe)
        throughput     (probe al secondo)
        """
        gallery, probe_g, probe_i = self._split_subjects(
            subjects, impostor_ratio, seed
        )

        g_tmpl, g_ids = self._flatten(gallery)
        pg_tmpl, pg_ids = self._flatten(probe_g)
        pi_tmpl, pi_ids = self._flatten(probe_i)

        # unisci i due insiemi probe per calcolare tempo/throughput su tutti
        all_probes = np.concatenate([pg_tmpl, pi_tmpl])
        S, elapsed = self._score_matrix(all_probes, g_tmpl)
        order = np.argsort(S, axis=1)
        if self._descending():
            order = np.fliplr(order)

        # separa indici per genuini / impostori
        split_idx = len(pg_tmpl)
        order_g = order[:split_idx]
        order_i = order[split_idx:]
        scores_best = S[np.arange(S.shape[0]), order[:, 0]]

        # mask accepted
        if self._descending():
            accepted = scores_best >= threshold
        else:
            accepted = scores_best <= threshold

        # ---------- genuini ---------- #
        acc_g = accepted[:split_idx]
        order_g_best = order_g[:, 0]
        best_ids_g = g_ids[order_g_best]

        TP = np.sum(
            (acc_g)
            & (best_ids_g == pg_ids)
            & (np.zeros_like(acc_g, dtype=int) < rank_k)
        )
        FN = len(pg_ids) - TP

        # ---------- impostori ---------- #
        acc_i = accepted[split_idx:]
        FP = np.sum(acc_i)  # ogni accettazione è FP
        # TN non serve

        FPIR = FP / len(pi_ids) * 100.0 if len(pi_ids) else 0.0
        FNIR = FN / len(pg_ids) * 100.0 if len(pg_ids) else 0.0

        search_ms = (elapsed / len(all_probes)) * 1000.0
        throughput = 1000.0 / search_ms if search_ms > 0 else 0.0

        return FPIR, FNIR, search_ms, throughput

    # ------------------------------------------------------------------ #
    #  Curva FPIR-FNIR + EER                                             #
    # ------------------------------------------------------------------ #
    def fpir_fnir_curve(
        self,
        subjects,
        thresholds=np.linspace(0.0, 1.0, 101),
        rank_k=1,
        impostor_ratio=0.2,
        seed=42,
        plot=True,
    ):
        fpir_list, fnir_list = [], []

        # Definisci un range di valori di threshold su cui effettuare il test
        if self._config.matching_algorithm == 'chi-square':
            thresholds = np.linspace(0.0, 100.0, num=200)
        elif self._config.matching_algorithm == 'hamming':
            thresholds = np.linspace(0.0, 1.0, num=200)
        elif self._config.matching_algorithm == 'euclidean':
            # thresholds = np.linspace(0.0, 1.0, num=200)
            thresholds = np.linspace(1.0, 60.0, num=200)
        elif self._config.matching_algorithm == 'cosine_similarity':
            thresholds = np.linspace(-1.0, 1.0, num=200)
        elif self._config.matching_algorithm == 'cosine_distance':
            thresholds = np.linspace(0.0, 2.0, num=200)
        else:
            raise ValueError("Matching algorithm not supported.")

        for th in thresholds:
            fp, fn, _, _ = self.fpir_fnir(
                subjects, th, rank_k, impostor_ratio, seed
            )
            fpir_list.append(fp)
            fnir_list.append(fn)

        fpir = np.asarray(fpir_list)
        fnir = np.asarray(fnir_list)

        # EER point
        idx = np.argmin(np.abs(fpir - fnir))
        eer = (fpir[idx] + fnir[idx]) / 2.0
        eer_th = thresholds[idx]

        if plot:
            plt.figure()
            plt.plot(thresholds, fpir, label="FPIR")
            plt.plot(thresholds, fnir, label="FNIR")
            plt.scatter([eer_th], [eer], color="red", zorder=3, label=f"EER={eer:.2f}%")
            plt.xlabel("Threshold")
            plt.ylabel("Rate (%)")
            plt.title(f"FPIR / FNIR – open-set ({self.biometric_trait})")
            plt.legend()
            plt.grid(True)
            path = f"{self._config.results_path}/{self.biometric_trait}_FPIR_FNIR_curve.png"
            plt.savefig(path, dpi=300)
            plt.close()

        return fpir, fnir, eer, eer_th

    # ------------------------------------------------------------------ #
    #  Curva DET  (FNIR vs FPIR)                                         #
    # ------------------------------------------------------------------ #
    def det_curve(
        self,
        subjects,
        thresholds=np.linspace(0.0, 1.0, 101),
        rank_k=1,
        impostor_ratio=0.2,
        seed=42,
        plot=True,
    ):
        """
        Crea una DET curve per l'identificazione open‑set:
        asse x = FPIR, asse y = FNIR al variare della soglia.

        Restituisce
        -----------
        fpir_arr : np.ndarray  (%)
        fnir_arr : np.ndarray  (%)
        eer      : float       (%)
        eer_th   : float       (threshold a cui avviene l'EER)
        """
        fpir_vals, fnir_vals = [], []

        # Definisci un range di valori di threshold su cui effettuare il test
        if self._config.matching_algorithm == 'chi-square':
            thresholds = np.linspace(0.0, 100.0, num=200)
        elif self._config.matching_algorithm == 'hamming':
            thresholds = np.linspace(0.0, 1.0, num=200)
        elif self._config.matching_algorithm == 'euclidean':
            # thresholds = np.linspace(0.0, 1.0, num=200)
            thresholds = np.linspace(1.0, 60.0, num=200)
        elif self._config.matching_algorithm == 'cosine_similarity':
            thresholds = np.linspace(-1.0, 1.0, num=200)
        elif self._config.matching_algorithm == 'cosine_distance':
            thresholds = np.linspace(0.0, 2.0, num=200)
        else:
            raise ValueError("Matching algorithm not supported.")

        for th in thresholds:
            fp, fn, _, _ = self.fpir_fnir(
                subjects, th, rank_k, impostor_ratio, seed
            )
            fpir_vals.append(fp)
            fnir_vals.append(fn)

        fpir_arr = np.asarray(fpir_vals)
        fnir_arr = np.asarray(fnir_vals)

        # Equal Error Rate
        idx = np.argmin(np.abs(fpir_arr - fnir_arr))
        eer = (fpir_arr[idx] + fnir_arr[idx]) / 2.0
        eer_th = thresholds[idx]

        if plot:
            plt.figure()
            # DET curve
            plt.plot(fpir_arr, fnir_arr, marker="o", color="blue", label="DET Curve")
            # EER point
            plt.scatter([fpir_arr[idx]], [fnir_arr[idx]],
                        color="red", zorder=3, label=f"EER = {eer:.2f}%")
            # Dashed reference lines at the EER point
            plt.axhline(eer, color="red", linestyle="--", linewidth=1)
            plt.axvline(fpir_arr[idx], color="red", linestyle="--", linewidth=1)

            plt.xlabel("FPIR (%)")   # sinonimo di FPIR su asse x
            plt.ylabel("FNIR (%)")   # sinonimo di FNIR su asse y
            plt.title("DET Curve with EER")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(loc="upper right")
            fname = f"{self._config.results_path}/{self.biometric_trait}_DET_curve.png"
            plt.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close()

        return fpir_arr, fnir_arr, eer, eer_th

    # ------------------------------------------------------------------ #
    #  Curva DIR-FPIR                                                    #
    # ------------------------------------------------------------------ #
    def dir_fpir_curve(
        self,
        subjects,
        thresholds=np.linspace(0.0, 1.0, 101),
        rank_k=1,
        impostor_ratio=0.2,
        seed=42,
        plot=True,
    ):
        dir_list, fpir_list, fnir_list = [], [], []

        # Definisci un range di valori di threshold su cui effettuare il test
        if self._config.matching_algorithm == 'chi-square':
            thresholds = np.linspace(0.0, 100.0, num=200)
        elif self._config.matching_algorithm == 'hamming':
            thresholds = np.linspace(0.0, 1.0, num=200)
        elif self._config.matching_algorithm == 'euclidean':
            # thresholds = np.linspace(0.0, 1.0, num=200)
            thresholds = np.linspace(1.0, 60.0, num=200)
        elif self._config.matching_algorithm == 'cosine_similarity':
            thresholds = np.linspace(-1.0, 1.0, num=200)
        elif self._config.matching_algorithm == 'cosine_distance':
            thresholds = np.linspace(0.0, 2.0, num=200)
        else:
            raise ValueError("Matching algorithm not supported.")

        for th in thresholds:
            fp, fn, _, _ = self.fpir_fnir(
                subjects, th, rank_k, impostor_ratio, seed
            )
            tpir = 100.0 - fn  # DIR = TPIR = 1 - FNIR
            dir_list.append(tpir)
            fpir_list.append(fp)
            fnir_list.append(fn)

        dir_arr = np.asarray(dir_list)
        fpir_arr = np.asarray(fpir_list)
        fnir_arr = np.asarray(fnir_list)

        if plot:
            plt.figure()
            plt.semilogx(fpir_arr, dir_arr, marker="o")
            plt.xlabel("FPIR (%)")
            plt.ylabel("DIR (%)")
            plt.title(f"DIR-FPIR curve ({self.biometric_trait})")
            plt.grid(True, which="both", ls="--")
            path = f"{self._config.results_path}/{self.biometric_trait}_DIR_FPIR_curve.png"
            plt.savefig(path, dpi=300)
            plt.close()

        return fpir_arr, fnir_arr, dir_arr