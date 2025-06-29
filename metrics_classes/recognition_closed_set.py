import time
import numpy as np
import matplotlib.pyplot as plt

from matching_classes.matching import Matching


class RecognitionClosedSet:
    """
    Valutazione closed-set con leave-one-out a 5 fold.

    • Rank-1  (Top-1 accuracy)
    • Rank-5  (Top-5 accuracy)
    • mAP     (mean Average Precision, stile person-ReID)
    • CMC     (plottata solo sul 1° fold, per velocità)

    La struttura del dizionario `subjects` deve essere:

        subjects[subject_id] = {
            "acquisition_name": [f"{subject_id}_0", ..., f"{subject_id}_4"],
            "template":         [np.ndarray,        ..., np.ndarray]
        }
    """

    def __init__(self, config, biometric_trait="gait"):
        self._config = config
        self.biometric_trait = biometric_trait
        self.matcher = Matching(config)

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _flatten(subjects_dict):
        """Converte {id:{'template':[...]}} in (tmpl_arr, id_arr)."""
        tmpl, ids = [], []
        for sid, data in subjects_dict.items():
            for t in data["template"]:
                tmpl.append(t)
                ids.append(sid)
        return np.asarray(tmpl, dtype=np.float32), np.asarray(ids)

    def _score_matrix(self, probes, gallery):
        """Compute full P×G score matrix (loop Python, OK per dataset piccolo)."""
        P, G = len(probes), len(gallery)
        S = np.empty((P, G), dtype=np.float32)
        for i in range(P):
            for j in range(G):
                S[i, j] = self.matcher.compare_templates(probes[i], gallery[j])
        return S

    def _descending(self):
        """True se score alto ⇒ più simile (es. cosine_similarity)."""
        return self._config.matching_algorithm.lower().endswith("similarity")

    # ------------------------------------------------------------------ #
    #  Metriche per uno split (gallery, probe)                           #
    # ------------------------------------------------------------------ #
    def _metrics_for_split(self, gallery, probe, max_rank=100):
        g_tmpl, g_ids = self._flatten(gallery)
        p_tmpl, p_ids = self._flatten(probe)

        tic = time.perf_counter()
        S = self._score_matrix(p_tmpl, g_tmpl)
        elapsed = time.perf_counter() - tic          # secondi totali
        search_time = elapsed / len(p_tmpl)          # sec/probe
        throughput  = 1.0 / search_time              # probe al secondo

        order = np.argsort(S, axis=1)
        if self._descending():
            order = np.fliplr(order)

        rank1_hits = 0
        rank5_hits = 0
        cumulative = np.zeros(max_rank, dtype=int)
        APs = []

        for i in range(order.shape[0]):
            ranked_ids = g_ids[order[i]]
            relevant = ranked_ids == p_ids[i]

            # posizione della prima occorrenza corretta
            rank_pos = np.where(relevant)[0][0]

            # Rank-1 / Rank-5
            if rank_pos == 0:
                rank1_hits += 1
            if rank_pos < 5:
                rank5_hits += 1

            # CMC
            if rank_pos < max_rank:
                cumulative[rank_pos:] += 1

            # AP per mAP
            hits, prec_at_hits = 0, []
            for r, rel in enumerate(relevant[:max_rank], start=1):
                if rel:
                    hits += 1
                    prec_at_hits.append(hits / r)
            APs.append(np.mean(prec_at_hits) if prec_at_hits else 0.0)

        n_probe = len(p_tmpl)
        rank1 = rank1_hits / n_probe * 100.0
        rank5 = rank5_hits / n_probe * 100.0
        cmc = cumulative / n_probe * 100.0
        mAP = np.mean(APs) * 100.0

        return rank1, rank5, cmc, mAP, search_time, throughput

    # ------------------------------------------------------------------ #
    #  CMC plot                                                          #
    # ------------------------------------------------------------------ #
    def _plot_cmc(self, ranks, cmc, suffix="fold0"):
        plt.figure()
        plt.plot(ranks, cmc, marker="o")
        plt.xlabel("Rank k")
        plt.ylabel("Identification rate (%)")
        plt.title(f"CMC curve ({self.biometric_trait}) – {suffix}")
        plt.grid(True)
        path = f"{self._config.results_path}/{self.biometric_trait}_CMC_{suffix}.png"
        plt.savefig(path, dpi=300)
        plt.close()

    # ------------------------------------------------------------------ #
    #  Leave-one-out k-fold (k = 5)                                      #
    # ------------------------------------------------------------------ #
    def evaluate_kfold(self, subjects, max_rank=100, plot_first_fold=True):
        """
        Esegue i 5 fold LOO e ritorna le medie di Rank-1, Rank-5, mAP.

        Ritorno
        -------
        rank1_mean, rank5_mean, mAP_mean  (float, float, float)
        """
        k = 5  # numero di acquisizioni per soggetto
        rank1_all, rank5_all, mAP_all = [], [], []
        time_all, tput_all = [], []

        for fold in range(k):
            gallery, probe = {}, {}

            # Split per soggetto: 4 template in gallery, 1 (indice = fold) in probe
            for sid, data in subjects.items():
                templates = data["template"]
                probe_idx = fold  # 0..4
                probe[sid] = {"template": [templates[probe_idx]]}
                gallery[sid] = {"template": [t for i, t in enumerate(templates)
                                             if i != probe_idx]}

            # Metriche per questo fold
            rank1, rank5, cmc, mAP, t_search, tput = self._metrics_for_split(
                gallery, probe, max_rank=max_rank
            )

            rank1_all.append(rank1)
            rank5_all.append(rank5)
            mAP_all.append(mAP)
            time_all.append(t_search)
            tput_all.append(tput)

            # CMC plot solo per il primo fold
            if fold == 0 and plot_first_fold:
                ranks = np.arange(1, max_rank + 1)
                self._plot_cmc(ranks, cmc, suffix="fold0")

        rank1_mean = float(np.mean(rank1_all))
        rank5_mean = float(np.mean(rank5_all))
        mAP_mean = float(np.mean(mAP_all))
        search_ms_mean = float(np.mean(time_all) * 1000.0)   # in millisecondi
        tput_mean = float(np.mean(tput_all))            # probe/sec

        return rank1_mean, rank5_mean, mAP_mean, search_ms_mean, tput_mean