import cv2
import numpy as np
from scipy.stats import entropy


class GaborWavelet:
    def __init__(self, face_config):
        self.face_config = face_config

    def _compute_entropy(self, face_template):
        unique, counts = np.unique(face_template, return_counts=True)
        probabilities = counts / counts.sum()
        return entropy(probabilities, base=2)

    def _gabor_wavelet_2d(self, sigma, theta, lambd):
        real = cv2.getGaborKernel(
            self.face_config.features_extraction.gabor_wavelet.ksize, 
            sigma,
            theta, 
            lambd, 
            self.face_config.features_extraction.gabor_wavelet.gamma, 
            self.face_config.features_extraction.gabor_wavelet.psi, 
            ktype=cv2.CV_64F
        )
        imag = cv2.getGaborKernel(
            self.face_config.features_extraction.gabor_wavelet.ksize, 
            sigma,
            theta, 
            lambd, 
            self.face_config.features_extraction.gabor_wavelet.gamma, 
            self.face_config.features_extraction.gabor_wavelet.psi + np.pi / 2, 
            ktype=cv2.CV_64F
        )
        # return real + 1j * imag
        return real, imag

    def _build_gabor_bank(self):
        """
        Costruisce la banca di filtri Gabor. Per ciascuna combinazione di scala (lambd) e orientamento (theta),
        vengono creati due kernel: uno per la parte reale (psi=0) e uno per la parte immaginaria (psi=pi/2).
        
        Ritorna:
          - bank: lista di tuple (kernel_reale, kernel_immaginario)
        """
        bank = []

        orientations = np.linspace(0, np.pi, self.face_config.features_extraction.gabor_wavelet.num_orientations, endpoint=False)

        for lambd in self.face_config.features_extraction.gabor_wavelet.scales:
            sigma = 0.75 * lambd  # relazione empirica tra sigma e lunghezza d'onda
            for theta in orientations:
                kernel_real, kernel_imag = self._gabor_wavelet_2d(sigma, theta, lambd)
                bank.append((kernel_real, kernel_imag))
        return bank
    
    def extract_gabor_wavelet_features(self, image):
        """
        Estrae le feature del volto applicando la banca di filtri Gabor.
        
        Per ciascun filtro:
          1. Convolve l'immagine con le parti reale e immaginaria.
          2. Calcola la risposta complessa e ne estrae la fase.
          3. Quantizza la fase in 4 livelli:
                - phase < -pi/2       --> codice 0  (rappresenta -2, binario "00")
                - -pi/2 <= phase < 0   --> codice 1  (rappresenta -1, binario "01")
                - 0 <= phase < pi/2    --> codice 2  (rappresenta  1, binario "10")
                - phase >= pi/2        --> codice 3  (rappresenta  2, binario "11")
          4. Converte il codice in due insiemi di bit.
        
        Ritorna:
          - combined_template: array 1D di bit (0 e 1) che rappresenta il template binario.
        """

        # features_list = []
    
        image = image.astype(np.float32) / 255

        gabor_bank = self._build_gabor_bank()

        # Inizializza le variabili per accumulare le risposte
        response_real_total = np.zeros_like(image, dtype=np.float32)
        response_imag_total = np.zeros_like(image, dtype=np.float32)
        
        # Per ciascun filtro della banca:
        for idx, (kernel_real, kernel_imag) in enumerate(gabor_bank):
            # Convoluzione per ottenere le risposte reale e immaginaria
            response_real = cv2.filter2D(image, cv2.CV_64F, kernel_real)
            response_imag = cv2.filter2D(image, cv2.CV_64F, kernel_imag)

            response_real_total += response_real
            response_imag_total += response_imag

        # Costruisci la risposta complessa e ne estrae la fase
        complex_response = response_real_total + 1j * response_imag_total

        # Estrazione della fase (in radianti, nell'intervallo [-pi, pi])
        phase = np.angle(complex_response)

        # **Quantizzazione della fase in 4 livelli (Quadranti)**
        bins = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]  # Copertura uniforme dei quadranti
        quantized_phase = np.digitize(phase, bins)
            
        # **Mappatura dei livelli a Iris Code**
        face_code = np.zeros_like(phase, dtype=int)
        face_code[quantized_phase == 1] = 0  # [-pi, -pi/2)
        face_code[quantized_phase == 2] = 2  # [-pi/2, 0)
        face_code[quantized_phase == 3] = 3  # [0, pi/2)
        face_code[quantized_phase == 4] = 1  # [pi/2, pi)

        # Converti in template binario: 0 per valori negativi, 255 per quelli positivi
        face_code_binary = ((face_code > 0) * 255).astype(np.uint8)

        entropy = self._compute_entropy(face_code_binary)
        print(f"Entropy: {entropy}")

        return face_code_binary, face_code_binary