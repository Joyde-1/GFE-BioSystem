import sys
import numpy as np

from multimodal.utils import load_config, browse_path, path_extractor, save_image
from pre_processing.prepare_data import PrepareData
from pre_processing.pre_processing import PreProcessing
from detection.viola_jones import ViolaJones
from detection.yolo import Yolo
from detection.cnn import CNN
from post_processing.post_processing import PostProcessing
from features_extraction.lbp import LBP
from features_extraction.gabor_wavelet import GaborWavelet
from features_extraction.features_scaling import FeaturesScaling
from features_extraction.fisherfaces import FisherFaceExtractor
from face.matching.face_matching import MatchingFace


def load_checkpoint(file_path='checkpoint_face.json'):
    import os
    import json

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint
    else:
        return None

def save_checkpoint(current_index, file_path='checkpoint_face.json'):
    import json

    checkpoint = {
        'current_index': current_index
    }
    with open(file_path, 'w') as f:
        json.dump(checkpoint, f)  


if __name__ == '__main__':
    face_config = load_config('face/config/face_config.yaml')

    if face_config.browse_path:
        face_config.data_dir = browse_path('Select the database folder')
        face_config.save_path = browse_path('Select the folder where images and plots will be saved')

        if face_config.algorithm_type == 'CNN':
            face_config.cnn.checkpoints_dir = browse_path('Select the folder that contains model checkpoint')
        if face_config.algorithm_type == 'yolo':
            face_config.yolo.checkpoints_dir = browse_path('Select the folder that contains model checkpoint')

        if face_config.save_groundtruths:
            face_config.groundtruths_path = browse_path('Select the folder where groundtruths will be saved')

    images_data = PrepareData(face_config)
    images, image_names, image_paths = images_data.load_face_images()

    pre_processing = PreProcessing(face_config)

    yolo = Yolo(face_config)
    cnn = CNN(face_config)
    viola_jones = ViolaJones(face_config)

    post_processing = PostProcessing(face_config)

    lbp = LBP(face_config)
    gabor_wavelet = GaborWavelet(face_config)


    checkpoint = load_checkpoint()
    start_index = checkpoint['current_index'] if checkpoint else 0

    subjects = {}

    for current_index, (image, image_name, image_path) in enumerate(zip(images, image_names, image_paths)):
        # if image_name != "3_1" and image_name != "3_2" and image_name != "3_3" and image_name != "3_4" and image_name != "3_5":
        # if image_name != "3_3":
        #     continue

        # Pre-Processing and Segmentation phases with the possibility 
        # to choose the algorithm between viola-jones, yolo or CNN
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

        if face_config.save_image.pre_processed_face_image:
            save_image(face_config, pre_processed_face_image, image_name, "pre_processed_face_image")

        if face_config.face_detection.detector == 'viola-jones':
            detected_image, bounding_box = viola_jones.detect_face(pre_processed_face_image.copy())
        elif face_config.face_detection.detector == 'yolo':
            pre_processed_face_image_path = path_extractor(face_config, image_name, "pre_processed_face_image")
            detected_image, bounding_box = yolo.predict_face_bounding_box(pre_processed_face_image_path)
        elif face_config.face_detection.detector == 'CNN':
            detected_image, bounding_box = cnn.predict_face_bounding_box(pre_processed_face_image.copy())
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        if face_config.save_image.detected_face_bounding_box:
            save_image(face_config, detected_image, image_name, "detected_face_bounding_box")

        post_processed_face_image = post_processing.post_processing_image(pre_processed_face_image.copy(), bounding_box)

        if face_config.save_image.post_processed_face_image:
            save_image(face_config, post_processed_face_image, image_name, "post_processed_face_image")

        if face_config.features_extractor == 'lbp':
            face_template, face_template_vis = lbp.extract_lbp_features(post_processed_face_image)
        elif face_config.features_extractor == 'gabor_wavelet':
            face_template, face_template_vis = gabor_wavelet.extract_gabor_wavelet_features(post_processed_face_image)

            print(type(face_template))
        elif face_config.features_extractor == 'fisherface':
            subjects[subject]['acquisition_name'].append(image_name)
            subjects[subject]['template'].append(post_processed_face_image)       
            
            # Salva il checkpoint dopo ogni combinazione
            save_checkpoint(current_index + 1)

            continue
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        if face_config.save_image.features_extracted_face_image:
            save_image(face_config, face_template_vis, image_name, f"{face_config.features_extractor}_face_image")

        subjects[subject]['acquisition_name'].append(image_name)
        subjects[subject]['template'].append(face_template)       
        
        # Salva il checkpoint dopo ogni combinazione
        save_checkpoint(current_index + 1)

    if face_config.features_extractor == 'fisherface':
        fisherface_extractor = FisherFaceExtractor(face_config)

        fisherfaces, visual_fisherfaces = fisherface_extractor.extract_fisherfaces(subjects)

        # Itera su subjects e assegna i templates (fisherfaces)
        fisherfaces_index = 0  # Indice per tracciare la posizione nei fisherfaces

        for subject in subjects.keys():
            subjects[subject]['template'] = []

            num_acquisitions = len(subjects[subject]['acquisition_name'])

            # Aggiungi i fisherfaces al soggetto a blocchi di num_acquisitions
            subjects[subject]['template'].extend(
                fisherfaces[fisherfaces_index:fisherfaces_index + num_acquisitions]
            )

            if face_config.save_image.features_extracted_face_image:
                for i in range(num_acquisitions):
                    save_image(face_config, visual_fisherfaces[i + fisherfaces_index], f"{subject}_{i}", f"{face_config.features_extractor}_face_image")

            # Aggiorna l'indice per il prossimo soggetto
            fisherfaces_index += num_acquisitions
    
    matching_face = MatchingFace(face_config)

    far, fa, t_imp = matching_face.calculate_far(subjects)
    frr, fr, t_legit = matching_face.calculate_frr(subjects)
    accuracy = matching_face.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Matching face metrics:")
    print(f"FAR: {far:.4f} %")
    print(f"FRR: {frr:.4f} %")
    print(f"accuracy: {accuracy:.4f} %")

    matching_face.calculate_roc_and_det(subjects)
    matching_face.far_vs_frr(subjects)