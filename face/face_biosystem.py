import sys
import os

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_classes.load_data import LoadData
from pre_processing.pre_processing import PreProcessing
from yolo_detection.yolo_detection import Yolo
from post_processing.post_processing import PostProcessing
from features_extraction_classes.lbp import LBP
from features_extraction_classes.gabor_wavelet import GaborWavelet
from features_extraction_classes.fisherfaces import FisherFaceExtractor
from matching_class.matching import Matching
from utils import load_config, browse_path, path_extractor, save_image, load_checkpoint, save_checkpoint



if __name__ == '__main__':
    face_config = load_config('config/face_config.yaml')

    if face_config.browse_path:
        face_config.data_dir = browse_path('Select the database folder')
        face_config.save_path = browse_path('Select the folder where images and plots will be saved')

        if face_config.detector == 'CNN':
            face_config.cnn.checkpoints_dir = browse_path('Select the folder that contains model checkpoint')
        if face_config.detector == 'yolo':
            face_config.yolo.checkpoints_dir = browse_path('Select the folder that contains model checkpoint')

    images_data = LoadData(face_config, 'face')
    images, image_names, image_paths = images_data.load_images()

    pre_processing = PreProcessing(face_config)

    yolo = Yolo(face_config, 'face')

    post_processing = PostProcessing(face_config)

    lbp = LBP(face_config)
    gabor_wavelet = GaborWavelet(face_config)

    if face_config.use_checkpoint:
        checkpoint = load_checkpoint('checkpoint_face.json')
        start_index = checkpoint['current_index'] if checkpoint else 0

    subjects = {}

    max_width = 0
    max_height = 0

    for current_index, (image, image_name, image_path) in enumerate(zip(images, image_names, image_paths)):
        # Pre-Processing and Segmentation phases
        # -----------------------------------------------------------
        
        # Extract the subject number from the image name
        subject = image_name.split('_')[0]

        print("Image name:", image_name)

        if subject not in subjects:
            subjects[subject] = {
                'acquisition_name': [], 
                'template': []
            }

        pre_processed_face_image = pre_processing.pre_processing_image(image.copy())

        if face_config.save_image.pre_processed:
            save_image(face_config, 'face', pre_processed_face_image, image_name, 'pre_processed_face_image')

        if face_config.detector == 'yolo':
            pre_processed_face_image_path = path_extractor(face_config, 'face', image_name, 'pre_processed_face_image')
            detected_image, bounding_box = yolo.predict_bounding_box(pre_processed_face_image_path)
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        if face_config.save_image.detected_face:
            save_image(face_config, 'face', detected_image, image_name, 'detected_face_bounding_box')

        post_processed_face_image, shape = post_processing.post_processing_image(pre_processed_face_image.copy(), bounding_box)

        if max_width < shape[0]:
            max_width = shape[0]
        
        if max_height < shape[1]:
            max_height = shape[1]

        if face_config.save_image.post_processed:
            save_image(face_config, 'face', post_processed_face_image, image_name, 'post_processed_face_image')

        if face_config.features_extractor == 'lbp':
            face_template, face_template_vis = lbp.extract_lbp_features(post_processed_face_image)
        elif face_config.features_extractor == 'gabor_wavelet':
            face_template, face_template_vis = gabor_wavelet.extract_gabor_wavelet_features(post_processed_face_image)

            print(type(face_template))
        elif face_config.features_extractor == 'fisherface':
            subjects[subject]['acquisition_name'].append(image_name)
            subjects[subject]['template'].append(post_processed_face_image)       
            
            # Salva il checkpoint dopo ogni combinazione
            if face_config.use_checkpoint:
                save_checkpoint('checkpoint_face.json', current_index + 1)

            continue
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        if face_config.save_image.features_extracted:
            save_image(face_config, 'face', face_template_vis, image_name, f"{face_config.features_extractor}_face_image")

        subjects[subject]['acquisition_name'].append(image_name)
        subjects[subject]['template'].append(face_template)       
        
        # Salva il checkpoint dopo ogni combinazione
        if face_config.use_checkpoint:
            save_checkpoint('checkpoint_face.json', current_index + 1)

    print("Max width: ", max_width)
    print("Max height: ", max_height)

    if face_config.features_extractor == 'fisherface':
        fisherface_extractor = FisherFaceExtractor(face_config)

        fisherfaces, visual_fisherfaces = fisherface_extractor.extract_fisherfaces(subjects, face_config.post_processing.image_size, face_config.post_processing.image_size)

        # Itera su subjects e assegna i templates (fisherfaces)
        fisherfaces_index = 0  # Indice per tracciare la posizione nei fisherfaces

        for subject in subjects.keys():
            subjects[subject]['template'] = []

            num_acquisitions = len(subjects[subject]['acquisition_name'])

            # Aggiungi i fisherfaces al soggetto a blocchi di num_acquisitions
            subjects[subject]['template'].extend(
                fisherfaces[fisherfaces_index:fisherfaces_index + num_acquisitions]
            )

            if face_config.save_image.features_extracted:
                for i in range(num_acquisitions):
                    save_image(face_config, 'face', visual_fisherfaces[i + fisherfaces_index], f"{subject}_{i}", f"{face_config.features_extractor}_face_image")

            # Aggiorna l'indice per il prossimo soggetto
            fisherfaces_index += num_acquisitions
    
    matching_face = Matching(face_config, 'face')

    far, fa, t_imp = matching_face.calculate_far(subjects)
    frr, fr, t_legit = matching_face.calculate_frr(subjects)
    accuracy = matching_face.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Matching face metrics:")
    print(f"FAR: {far:.4f} %")
    print(f"FRR: {frr:.4f} %")
    print(f"accuracy: {accuracy:.4f} %")

    matching_face.calculate_roc_and_det(subjects)
    matching_face.far_vs_frr(subjects)