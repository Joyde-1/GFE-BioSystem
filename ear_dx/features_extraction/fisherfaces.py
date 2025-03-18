import os
import pickle
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

try:
	from face.features_extraction.features_scaling import FeaturesScaling
except:
	from features_extraction.features_scaling import FeaturesScaling

class FisherFaceExtractor:
	"""
	Classe per l'estrazione delle Fisherfaces da immagini di volti.
	
	È necessario addestrare il modello LDA con un set di immagini (in scala di grigi) 
	e le rispettive etichette (ad es. l'identità della persona).
	"""
	def __init__(self, face_config):
		self._face_config = face_config
		self.n_components = 0
		self._face_scaler = FeaturesScaling(self._face_config.features_extraction.fisherfaces.scaler_type)
		self._lda = None
		
	def _scale_face_images(self, face_images):
		if self._face_config.features_extraction.fisherfaces.scaler_type != 'None':
			self._face_scaler.fit_scaler(face_images.copy())

		face_images_scaled = []

		for face_image in face_images:
			face_image_scaled = self._face_scaler.scaling(face_image)
			face_images_scaled.append(face_image_scaled)
		
		return face_images_scaled
	
	def _prepare_face_images(self, subjects):
		# Concatena i template
		face_images = [face_image for subject in subjects.values() for face_image in subject['template']]
		
		print("Shape di una face image:", face_images[0].shape)

		# if self._face_config.features_extraction.fisherfaces.scaler_type != None:
		# 	face_images = self._scale_face_images(face_images)
		# else:
		# 	if len(face_images[0].shape) == 1:  # Se è un array 1D (face template)
		# 		face_images = np.array([face_image.reshape(1, -1) for face_image in face_images])		# Rendi 2D con shape (1, 640)
		# 	else:  # Se è un array 2D (iris templates)
		# 		face_images = np.array([face_image.flatten().reshape(1, -1) for face_image in face_images])	# Appiattisci in (1, 8192)
		
		face_images = self._scale_face_images(face_images)

		# Convertiamo la lista in un array numpy di forma (n_samples, n_features)
		# return np.vstack(face_images)
		return np.array(face_images)

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
		os.makedirs(self._face_config.features_extraction.fisherfaces.checkpoints_dir, exist_ok=True)

		model_path = os.path.join(self._face_config.features_extraction.fisherfaces.checkpoints_dir, "lda.pkl")
		with open(model_path, 'wb') as f:
			pickle.dump(self._lda, f)  # Salva con Pickle

		print(f"LDA model saved as {model_path}.")

	def _load_lda_model(self):
		# Load LDA model using pickle
		self._lda = pickle.load(open(f"{self._face_config.features_extraction.fisherfaces.checkpoints_dir}/lda.pkl", 'rb'))

		print("LDA model loaded.")

	def _train_lda(self, subjects):
		"""
		Addestra il modello LDA (Fisherfaces) utilizzando un insieme di immagini e le relative etichette.
		
		Parametri:
		  - images: lista di immagini in scala di grigi (tutte della stessa dimensione)
		  - labels: lista di etichette (interi o stringhe) corrispondenti a ciascuna immagine
		"""
		# Salviamo la forma dell'immagine (supponiamo tutte della stessa dimensione)
		
		self.face_images = self._prepare_face_images(subjects)
		self.subjects_ID = self._prepare_subjects_ID(subjects)

		print("Face images shape after scaling:", self.face_images.shape)
		
		unique_labels = np.unique(self.subjects_ID)
		self.n_components = min(len(unique_labels) - 1, len(self.face_images) - len(unique_labels))
		print("n_components:", self.n_components)
		
		# Creiamo ed addestriamo il modello LDA.
		# Uso il solver 'eigen' per poter accedere ai coefficienti (coef_) successivamente.
		# self._lda = LinearDiscriminantAnalysis(n_components=self.n_components, solver='eigen', shrinkage='auto')
		self._lda = LinearDiscriminantAnalysis(n_components=self.n_components)
		self._lda.fit(self.face_images, self.subjects_ID)

		self._save_lda_model()
	
	def _extract_fisherface(self, face_image):
		"""
		Estrae il vettore delle caratteristiche (proiezione LDA) per l'immagine fornita.
		
		Parametri:
		  - face_image: immagine in scala di grigi contenente il volto (già rilevato e pre-processato)
		
		Ritorna:
		  - features: vettore (array numpy) delle caratteristiche estratte
		"""
		if self._lda is None:
			raise Exception("Il modello LDA non è stato addestrato. Esegui il metodo train_lda() prima. \n")
		
		# print("CIola")
		
		# print("Type type(face_image): ", type(face_image))
		# print("Shape: ", face_image.shape)
		
		face_image = face_image.reshape(1, -1)

		# print("Type: ", type(face_image))
		# print("Shape: ", face_image.shape)

		# print("Type type(face_image[0]): ", type(face_image[0]))
		# print("Shape: ", face_image[0].shape)


		# Appiattisci l'immagine e trasforma con il modello LDA
		# face_image = self._face_scaler.scaling(face_image)
		face_features = self._lda.transform(face_image)

		# print("Type: ", type(face_features))
		# print("Shape: ", face_features.shape)
		# print("Type: ", type(face_features[0]))
		# print("Shape: ", face_features[0].shape)

		return face_features[0]
	
	def _extract_visual(self, fisherface):
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
		fisherface_image = fisherface_vector.reshape((self._face_config.post_processing.image_size, self._face_config.post_processing.image_size))

		# Normalizza per visualizzazione con cv2
		fisherface_image = cv2.normalize(fisherface_image, None, 0, 255, cv2.NORM_MINMAX)
		fisherface_image = np.uint8(fisherface_image)



		# # Prendi la fisherface corretta dagli autovettori LDA
		# fisherface_vector = self._lda.scalings_[:, fisherface_index]

		# # Verifica la dimensione dell'immagine e ridimensiona
		# image_size = self._face_config.post_processing.image_size
		# fisherface_vector = fisherface_vector.reshape((image_size, image_size))

		# # Normalizza per cv2.imshow()
		# fisherface_image = cv2.normalize(fisherface_vector, None, 0, 255, cv2.NORM_MINMAX)
		# fisherface_image = np.uint8(fisherface_image)

		# Mostra l'immagine della Fisherface
		if self._face_config.show_images.features_extracted_face_image:
			cv2.imshow(f"Fisherface image", fisherface_image)
			cv2.moveWindow("Fisherface image", 0, 0)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return fisherface_image
	
	# def _extract_visual(self, fisherface):
	# 	"""
	# 	Crea una rappresentazione visiva della Fisherface.
		
	# 	In questo esempio, si utilizza il vettore dei coefficienti della prima discriminante (coef_[0])
	# 	come approssimazione della fisherface. Il vettore viene rimodellato alla dimensione originale
	# 	dell'immagine e normalizzato in modo da poter essere visualizzato con cv2.imshow().
		
	# 	Parametri:
	# 	  - image: immagine in scala di grigi contenente il volto (già rilevato e pre-processato)
		
	# 	Ritorna:
	# 	  - fisherface_img: immagine (array numpy di tipo uint8) pronta per essere visualizzata
	# 	"""
	# 	if self._lda is None:
	# 		raise Exception("Il modello LDA non è stato addestrato. Esegui il metodo train_lda() prima.")
		
	# 	# Verifica la dimensione e ridimensiona l'immagine
	# 	image_size = self._face_config.post_processing.image_size

	# 	if fisherface.size != image_size * image_size:
	# 		fisherface = np.resize(fisherface, (image_size, image_size))

	# 	# Normalizzazione per cv2.imshow()
	# 	fisherface_image = cv2.normalize(fisherface, None, 0, 255, cv2.NORM_MINMAX)
	# 	fisherface_image = np.uint8(fisherface_image)

	# 	# Mostra l'immagine della Fisherface
	# 	if self._face_config.show_images.features_extracted_face_image:
	# 		cv2.imshow("Fisherface image", fisherface_image)
	# 		cv2.moveWindow("Fisherface image", 0, 0)
	# 		cv2.waitKey(0)
	# 		cv2.destroyAllWindows()

	# 	return fisherface_image

	def extract_fisherfaces(self, subjects):
		self._train_lda(subjects)
		fisherfaces = [self._extract_fisherface(np.array(face_image)) for face_image in self.face_images]
		print("Pompa faces: ", type(fisherfaces))
		print("Pompa faces: ", len(fisherfaces))
		print("Pompa faces: ", type(fisherfaces[0]))
		print("Pompa faces: ", fisherfaces[0].shape)
		visual_fisherfaces = [self._extract_visual(fisherface) for fisherface in fisherfaces]
		# visual_fisherfaces = [self._lda.scalings_[:, i].reshape((self._face_config.post_processing.image_size, self._face_config.post_processing.image_size)) for i in range(len(fisherfaces))]
		return fisherfaces, visual_fisherfaces