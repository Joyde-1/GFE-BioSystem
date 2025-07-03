import os
import pickle
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from features_extraction_classes.features_scaling import FeaturesScaling


class FisherFaceExtractor:
	"""
	Classe per l'estrazione delle Fisherfaces da immagini.
	
	È necessario addestrare il modello LDA con un set di immagini (in scala di grigi) 
	e le rispettive etichette (ad es. l'identità della persona).
	"""
	def __init__(self, config):
		self._config = config
		self.n_components = 0
		self._scaler = FeaturesScaling(self._config.features_extraction.fisherfaces.scaler_type)

		if self._config.features_extraction.fisherfaces.load_model:
			# Carica il modello LDA da un file
			self._load_lda_model()
		else:
			self._lda = None
		
	def _scale_images(self, images):
		if self._config.features_extraction.fisherfaces.scaler_type != 'None':
			self._scaler.fit_scaler(images.copy())

		images_scaled = []

		for image in images:
			image_scaled = self._scaler.scaling(image)
			images_scaled.append(image_scaled)
		
		return images_scaled
	
	def _prepare_images(self, subjects):
		# Concatena i template
		images = [image for subject in subjects.values() for image in subject['template']]
		
		print("Shape di una image:", images[0].shape)
		
		images = self._scale_images(images)

		# Convertiamo la lista in un array numpy di forma (n_samples, n_features)
		return np.array(images)

	def _prepare_subjects_ID(self, subjects):
		# Itera su subjects per creare la lista degli ID dei soggetti associati a ciascun acquisizione
		subjects_ID = []

		for subject in subjects.keys():
			num_acquisitions = len(subjects[subject]['acquisition_name'])

			for i in range(num_acquisitions):
				subjects_ID.append(subject)

			self.n_components += 1

		self.n_components -= 1

		print("n_components:", self.n_components)

		return np.array(subjects_ID)
	
	def _save_lda_model(self):
		"""
		Salva un modello addestrato in un file.
		"""

		# Crea la directory se non esiste
		os.makedirs(self._config.features_extraction.fisherfaces.checkpoints_dir, exist_ok=True)

		model_path = os.path.join(self._config.features_extraction.fisherfaces.checkpoints_dir, "lda.pkl")
		with open(model_path, 'wb') as f:
			pickle.dump(self._lda, f)  # Salva con Pickle

		print(f"LDA model saved as {model_path}.")

	def _load_lda_model(self):
		# Load LDA model using pickle
		self._lda = pickle.load(open(f"{self._config.features_extraction.fisherfaces.checkpoints_dir}/lda.pkl", 'rb'))

		print("LDA model loaded.")

	def _train_lda(self, subjects):
		"""
		Addestra il modello LDA (Fisherfaces) utilizzando un insieme di immagini e le relative etichette.
		
		Parametri:
		  - images: lista di immagini in scala di grigi (tutte della stessa dimensione)
		  - labels: lista di etichette (interi o stringhe) corrispondenti a ciascuna immagine
		"""
		# Salviamo la forma dell'immagine (supponiamo tutte della stessa dimensione)
		
		self.images = self._prepare_images(subjects)
		self.subjects_ID = self._prepare_subjects_ID(subjects)

		print("Face images shape after scaling:", self.images.shape)
		
		unique_labels = np.unique(self.subjects_ID)
		self.n_components = min(len(unique_labels) - 1, len(self.images) - len(unique_labels))
		print("n_components:", self.n_components)
		
		# Creiamo ed addestriamo il modello LDA.
		self._lda = LinearDiscriminantAnalysis(n_components=self.n_components)
		self._lda.fit(self.images, self.subjects_ID)

		self._save_lda_model()
	
	def extract_fisherface(self, image):
		"""
		Estrae il vettore delle caratteristiche (proiezione LDA) per l'immagine fornita.
		
		Parametri:
		  - image: immagine in scala di grigi contenente il volto (già rilevato e pre-processato)
		
		Ritorna:
		  - features: vettore (array numpy) delle caratteristiche estratte
		"""
		if self._lda is None:
			raise Exception("Il modello LDA non è stato addestrato. Esegui il metodo train_lda() prima. \n")
		
		# print("Type type(image): ", type(image))
		# print("Shape: ", image.shape)
		
		image = image.reshape(1, -1)

		# print("Type: ", type(image))
		# print("Shape: ", image.shape)

		# print("Type type(image[0]): ", type(image[0]))
		# print("Shape: ", mage[0].shape)

		# Appiattisci l'immagine e trasforma con il modello LDA
		# image = self._scaler.scaling(image)
		face_features = self._lda.transform(image)

		# print("Type: ", type(face_features))
		# print("Shape: ", face_features.shape)
		# print("Type: ", type(face_features[0]))
		# print("Shape: ", face_features[0].shape)

		return face_features[0]
	
	def extract_visual(self, fisherface, width, height):
		"""
		Crea una rappresentazione visiva della Fisherface.

		Parametri:
		- fisherface_index: indice della Fisherface da visualizzare.

		Ritorna:
		- fisherface_img: immagine (array numpy di tipo uint8) pronta per essere visualizzata.
		"""
		if self._lda is None:
			raise Exception("Il modello LDA non è stato addestrato. Esegui il metodo train_lda() prima.")
		
		# Ricostruisci l'immagine nello spazio originale moltiplicando per i vettori di proiezione
		fisherface_vector = np.dot(fisherface, self._lda.scalings_.T)  

		# Rimappa la Fisherface alla dimensione dell'immagine originale
		fisherface_image = fisherface_vector.reshape((width, height))

		# Normalizza per visualizzazione con cv2
		fisherface_image = cv2.normalize(fisherface_image, None, 0, 255, cv2.NORM_MINMAX)
		fisherface_image = np.uint8(fisherface_image)

		# Mostra l'immagine della Fisherface
		if self._config.show_images.features_extracted_image:
			cv2.imshow(f"Fisherface image", fisherface_image)
			cv2.moveWindow("Fisherface image", 0, 0)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return fisherface_image

	def extract_fisherfaces(self, subjects, width, height):
		self._train_lda(subjects)
		fisherfaces = [self.extract_fisherface(np.array(image)) for image in self.images]
		print("Fisher faces: ", type(fisherfaces))
		print("Fisher faces: ", len(fisherfaces))
		print("Fisher faces: ", type(fisherfaces[0]))
		print("Fisher faces: ", fisherfaces[0].shape)
		visual_fisherfaces = [self.extract_visual(fisherface, width, height) for fisherface in fisherfaces]
		return fisherfaces, visual_fisherfaces