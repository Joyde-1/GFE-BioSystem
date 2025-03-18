import sys
import numpy as np

from data_classes.load_data import LoadData
from ear_utils import load_config, browse_path, path_extractor, save_image
from pre_processing.pre_processing import PreProcessing
from detection.viola_jones import ViolaJones
from detection.yolo import Yolo
from post_processing.post_processing import PostProcessing
from features_extraction.lbp import LBP
from features_extraction.gabor_wavelet import GaborWavelet
from features_extraction.features_scaling import FeaturesScaling
from features_extraction.fisherfaces import FisherFaceExtractor
from matching.ear_matching import EarMatching


def load_checkpoint(file_path='checkpoint_ear.json'):
    import os
    import json

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint
    else:
        return None

def save_checkpoint(current_index, file_path='checkpoint_ear.json'):
    import json

    checkpoint = {
        'current_index': current_index
    }
    with open(file_path, 'w') as f:
        json.dump(checkpoint, f)  


if __name__ == '__main__':
    ear_config = load_config('ear_dx/config/ear_config.yaml')

    if ear_config.browse_path:
        ear_config.data_dir = browse_path('Select the database folder')
        ear_config.save_path = browse_path('Select the folder where images and plots will be saved')

        if ear_config.algorithm_type == 'yolo':
            ear_config.yolo.checkpoints_dir = browse_path('Select the folder that contains model checkpoint')

        if ear_config.save_groundtruths:
            ear_config.groundtruths_path = browse_path('Select the folder where groundtruths will be saved')

    images_data = LoadData(ear_config, 'ear_dx')
    images, image_names, image_paths = images_data.load_images()

    pre_processing = PreProcessing(ear_config)

    yolo = Yolo(ear_config)
    viola_jones = ViolaJones(ear_config)

    post_processing = PostProcessing(ear_config)

    lbp = LBP(ear_config)
    gabor_wavelet = GaborWavelet(ear_config)


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

        pre_processed_ear_image = pre_processing.pre_processing_image(image.copy())

        if ear_config.save_image.pre_processed:
            save_image(ear_config, 'ear_dx', pre_processed_ear_image, image_name, "pre_processed_ear_dx_image")

        if ear_config.ear_detection.detector == 'viola-jones':
            detected_image, bounding_box = viola_jones.detect_ear(pre_processed_ear_image.copy())
        elif ear_config.ear_detection.detector == 'yolo':
            pre_processed_ear_image_path = path_extractor(ear_config, 'ear_dx', image_name, "pre_processed_ear_dx_image")
            detected_image, bounding_box = yolo.predict_ear_bounding_box(pre_processed_ear_image_path)
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        if ear_config.save_image.detected:
            save_image(ear_config, 'ear_dx', detected_image, image_name, "detected_ear_dx_bounding_box")

        post_processed_ear_image = post_processing.post_processing_image(pre_processed_ear_image.copy(), bounding_box)

        if ear_config.save_image.post_processed:
            save_image(ear_config, 'ear_dx', post_processed_ear_image, image_name, "post_processed_ear_dx_image")

        if ear_config.features_extractor == 'lbp':
            ear_template, ear_template_vis = lbp.extract_lbp_features(post_processed_ear_image)
        elif ear_config.features_extractor == 'gabor_wavelet':
            ear_template, ear_template_vis = gabor_wavelet.extract_gabor_wavelet_features(post_processed_ear_image)

            print(type(ear_template))
        elif ear_config.features_extractor == 'fisherface':
            subjects[subject]['acquisition_name'].append(image_name)
            subjects[subject]['template'].append(post_processed_ear_image)       
            
            # Salva il checkpoint dopo ogni combinazione
            save_checkpoint(current_index + 1)

            continue
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        if ear_config.save_image.features_extracted:
            save_image(ear_config, 'ear_dx', ear_template_vis, image_name, f"{ear_config.features_extractor}_ear_dx_image")

        subjects[subject]['acquisition_name'].append(image_name)
        subjects[subject]['template'].append(ear_template)       
        
        # Salva il checkpoint dopo ogni combinazione
        save_checkpoint(current_index + 1)

    if ear_config.features_extractor == 'fisherface':
        fisherface_extractor = FisherFaceExtractor(ear_config)

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

            if ear_config.save_image.features_extracted:
                for i in range(num_acquisitions):
                    save_image(ear_config, 'ear_dx', visual_fisherfaces[i + fisherfaces_index], f"{subject}_{i}", f"{ear_config.features_extractor}_ear_dx_image")

            # Aggiorna l'indice per il prossimo soggetto
            fisherfaces_index += num_acquisitions
    
    ear_matching = EarMatching(ear_config)

    far, fa, t_imp = ear_matching.calculate_far(subjects)
    frr, fr, t_legit = ear_matching.calculate_frr(subjects)
    accuracy = ear_matching.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Matching ear dx metrics:")
    print(f"FAR: {far:.4f} %")
    print(f"FRR: {frr:.4f} %")
    print(f"accuracy: {accuracy:.4f} %")

    ear_matching.calculate_roc_and_det(subjects)
    ear_matching.far_vs_frr(subjects)