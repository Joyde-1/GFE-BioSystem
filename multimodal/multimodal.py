import sys
import cv2
import numpy as np


# IMPORT GAIT CLASSES
# TODO: add gait classes

# IMPORT FACE CLASSES
from face.pre_processing.pre_processing import PreProcessing
from face.detection.viola_jones import ViolaJones
from face.detection.yolo import Yolo
from face.detection.cnn import CNN
from face.post_processing.post_processing import PostProcessing
from face.features_extraction.lbp import LBP
from face.features_extraction.gabor_wavelet import GaborWavelet
#from face.features_extraction.features_scaling import FeaturesScaling
from face.features_extraction.fisherfaces import FisherFaceExtractor
from face.matching.face_matching import MatchingFace
from face.face_utils import path_extractor

# IMPORT EAR CLASSES
from ear.ear_utils import path_extractor

# MULTIMODAL CLASSES
from preprocessing_classes.prepare_data import PrepareData
from features_fusion.features_scaling import FeaturesScalingMultimodal
from features_fusion.pca import FeaturesFusionPCA
from matching.matching import MatchingMultimodal
from matching_fusion.matching_fusion import MatchingMultimodalFusion
from multimodal.utils import load_config, browse_path, save_image


if __name__ == '__main__':
    # LOAD CONFIG FILES
    multimodal_config = load_config('multimodal/config/multimodal_config.yaml')
    gait_config = load_config('face/config/gait_config.yaml')
    face_config = load_config('face/config/face_config.yaml')
    ear_config = load_config('face/config/ear_config.yaml')

    # BROWSE PATHS
    if multimodal_config.browse_path:
        multimodal_config.data_dir = browse_path('Select the database folder')
        multimodal_config.save_path = browse_path('Select the folder where images and plots will be saved')
        if face_config.algorithm_type == 'CNN':
            face_config.cnn.checkpoints_dir = browse_path('Select the folder that contains CNN model checkpoint')

    # INSTANTIATE MULTIMODAL CLASS OBJECTS
    images_data = PrepareData(multimodal_config)
    matching_first_multimodal = MatchingMultimodal(multimodal_config)
    matching_second_multimodal = MatchingMultimodal(multimodal_config)
    matching_third_multimodal = MatchingMultimodalFusion(multimodal_config)

    # INSTANTIATE GAIT CLASS OBJECTS
    # TODO: instantiate gait class objects
    matching_gait = MatchingGait(gait_config)

    # INSTANTIATE FACE CLASS OBJECTS
    pre_processing = PreProcessing(face_config)
    yolo = Yolo(face_config)
    cnn = CNN(face_config)
    viola_jones = ViolaJones(face_config)
    post_processing = PostProcessing(face_config)
    lbp = LBP(face_config)
    gabor_wavelet = GaborWavelet(face_config)
    matching_face = MatchingFace(face_config)

    # INSTANTIATE EAR CLASS OBJECTS
    # TODO: instantiate ear class objects
    matching_dx_ear = MatchingEar(ear_config)
    matching_sx_ear = MatchingEar(ear_config)
    
    # CHECK GAIT, FACE AND EARS ASSOCIATION
    # if not gait_frames_names == face_image_names == ear_image_names_dx == ear_image_names_sx:
    #     print("Gait, face and ears don't match!")

    # INITIALIZE SUBJECTS
    subjects = {
        'gait': {},
        'face': {},
        'ear_dx': {},
        'ear_sx': {},
        'fused': {},
        'fused_reduced': {}
    }



    #----------------------
    # 1 - LOAD IMAGES PHASE
    #----------------------

    gait_images, gait_image_names, gait_image_paths = images_data.load_gait_images()
    face_images, face_image_names, face_image_paths = images_data.load_face_images()
    ear_images, ear_image_names, ear_image_paths = images_data.load_ears_images()



    for current_index, (acquisition_name, gait_image, gait_image_path, face_image, face_image_path, ear_dx_image, ear_dx_image_path, ear_sx_image, ear_sx_image_path) in enumerate(zip(face_image_names, gait_images, gait_image_paths, face_images, face_image_paths, ear_images['dx'], ear_image_paths['dx'], ear_images['sx'], ear_image_paths['sx'])):

        print("Image name: ", acquisition_name)

        # Extract the subject number from the image name
        subject = acquisition_name.split('_')[0]

        if subject not in subjects['face']:
            subjects['gait'][subject] = {
                'acquisition_name': [], 
                'template': []
            }
            subjects['face'][subject] = {
                'acquisition_name': [], 
                'template': []
            }
            subjects['ear_dx'][subject] = {
                'acquisition_name': [], 
                'template': [],
                'mask': []
            }
            subjects['ear_sx'][subject] = {
                'acquisition_name': [], 
                'template': [],
                'mask': []
            }
            subjects['fused'][subject] = {
                'acquisition_name': [], 
                'template': []
            }
            subjects['fused_reduced'][subject] = {
                'acquisition_name': [], 
                'template': []
            }

        subjects['fused'][subject]['acquisition_name'].append(acquisition_name)
        subjects['fused_reduced'][subject]['acquisition_name'].append(acquisition_name)
        


        #-------------------------
        # 2 - PRE-PROCESSING PHASE
        #-------------------------

        # ----
        # GAIT
        # ----

        # TODO: add gait pre-processing

        # ----
        # FACE
        # ----

        pre_processed_face_image = pre_processing.pre_processing_image(face_original_image.copy())
        
        # ------
        # EAR DX
        # ------

        # TODO: add ear dx pre-processing

        # ------
        # EAR SX
        # ------

        # TODO: add ear sx pre-processing


        # SAVE PRE-PROCESSED IMAGES
        if multimodal_config.save_images.pre_processed:
            if gait_config.save_image.pre_processed:
                save_image(gait_config, pre_processed_gait_image, acquisition_name, file_suffix="pre_processed_gait")
            if face_config.save_image.pre_processed:
                save_image(face_config, pre_processed_face_image, acquisition_name, file_suffix="pre_processed_face")
            if ear_config.save_image.pre_processed:
                save_image(ear_config, pre_processed_ear_dx_image, acquisition_name, side='dx', file_suffix="pre_processed_ear_dx")
                save_image(ear_config, pre_processed_ear_sx_image, acquisition_name, side='sx', file_suffix="pre_processed_ear_sx")
        


        #--------------------
        # 3 - DETECTION PHASE
        #--------------------

        # ----
        # GAIT
        # ----

        # TODO: add gait detection

        # ----
        # FACE
        # ----

        # Face Detection phase with the possibility 
        # to choose the algorithm between viola-jones, yolo or CNN
        if face_config.algorithm_type == 'viola-jones':
            detected_face_image, bounding_box = viola_jones.detect_face(pre_processed_face_image.copy())
        elif face_config.algorithm_type == 'yolo':
            pre_processed_face_image_path = path_extractor(face_config, acquisition_name, "pre_processed_face")
            detected_face_image, bounding_box = yolo.predict_face_bounding_box(pre_processed_face_image_path)
        elif face_config.algorithm_type == 'CNN':
            detected_face_image, bounding_box = cnn.predict_face_bounding_box(pre_processed_face_image.copy())
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        # ------
        # EAR DX
        # ------

        # TODO: add ear dx detection

        # ------
        # EAR SX
        # ------

        # TODO: add ear sx detection


        # SAVE DETECTED IMAGES
        if multimodal_config.save_images.detected:
            if gait_config.save_image.detected:
                save_image(gait_config, detected_gait_image, acquisition_name, file_suffix="detected_gait")
            if face_config.save_image.detected:
                save_image(face_config, detected_face_image, acquisition_name, file_suffix="detected_face")
            if ear_config.save_image.detected:
                save_image(ear_config, detected_ear_dx_image, acquisition_name, side='dx', file_suffix="detected_ear_dx")
                save_image(ear_config, detected_ear_sx_image, acquisition_name, side='sx', file_suffix="detected_ear_sx")

        

        #--------------------------
        # 4 - POST-PROCESSING PHASE
        #--------------------------

        # ----
        # GAIT
        # ----

        # TODO: add gait post-processing

        # ----
        # FACE
        # ----

        post_processed_face_image = post_processing.post_processing_image(pre_processed_face_image.copy(), bounding_box)

        # ------
        # EAR DX
        # ------

        # TODO: add ear dx post-processing

        # ------
        # EAR SX
        # ------

        # TODO: add ear sx post-processing


        # SAVE POST-PROCESSED IMAGES
        if multimodal_config.save_images.post_processed:
            if gait_config.save_image.post_processed:
                save_image(gait_config, post_processed_gait_image, acquisition_name, file_suffix="post_processed_gait")
            if face_config.save_image.post_processed:
                save_image(face_config, post_processed_face_image, acquisition_name, file_suffix="post_processed_face")
            if ear_config.save_image.post_processed:
                save_image(ear_config, post_processed_ear_dx_image, acquisition_name, side='dx', file_suffix="post_processed_ear_dx")
                save_image(ear_config, post_processed_ear_sx_image, acquisition_name, side='sx', file_suffix="post_processed_ear_sx")


        
        #------------------------------
        # 5 - FEATURES EXTRACTION PHASE
        #------------------------------

        # ----
        # GAIT
        # ----

        # TODO: add gait features extraction

        # ----
        # FACE
        # ----

        if face_config.features_extractor == 'lbp':
            face_template, face_template_vis = lbp.extract_lbp_features(post_processed_face_image)
            subjects['face'][subject]['acquisition_name'].append(acquisition_name)
            subjects['face'][subject]['template'].append(face_template) 
        elif face_config.features_extractor == 'gabor_wavelet':
            face_template, face_template_vis = gabor_wavelet.extract_gabor_wavelet_features(post_processed_face_image)
            subjects['face'][subject]['acquisition_name'].append(acquisition_name)
            subjects['face'][subject]['template'].append(face_template) 
        elif face_config.features_extractor == 'fisherface':
            subjects['face'][subject]['acquisition_name'].append(acquisition_name)
            subjects['face'][subject]['template'].append(post_processed_face_image)
        else:
            raise ValueError("Unknown algorithm type! \n")

        # subjects['face'][subject]['acquisition_name'].append(acquisition_name)
        # subjects['face'][subject]['template'].append(face_template)

        # ------
        # EAR DX
        # ------

        # TODO: add ear dx features extraction

        # ------
        # EAR SX
        # ------

        # TODO: add ear sx features extraction


        # SAVE FEATURES EXTRACTED IMAGES
        if multimodal_config.save_images.features_extracted:
            if gait_config.save_image.features_extracted:
                save_image(gait_config, features_extracted_gait_image, acquisition_name, file_suffix="features_extracted_gait")
            if face_config.save_image.features_extracted and face_config.features_extractor != 'fisherface':
                save_image(face_config, features_extracted_face_image, acquisition_name, file_suffix="features_extracted_face")
            if ear_config.save_image.features_extracted:
                save_image(ear_config, features_extracted_ear_dx_image, acquisition_name, side='dx', file_suffix="features_extracted_ear_dx")
                save_image(ear_config, features_extracted_ear_sx_image, acquisition_name, side='sx', file_suffix="features_extracted_ear_sx")

        

    if face_config.features_extractor == 'fisherface':
        fisherface_extractor = FisherFaceExtractor(face_config)

        fisherfaces, visual_fisherfaces = fisherface_extractor.extract_fisherfaces(subjects['face'])

        # Itera su subjects e assegna i templates (fisherfaces)
        fisherfaces_index = 0  # Indice per tracciare la posizione nei fisherfaces

        for subject in subjects['face'].keys():
            subjects['face'][subject]['template'] = []

            num_acquisitions = len(subjects['face'][subject]['acquisition_name'])

            # Aggiungi i fisherfaces al soggetto a blocchi di num_acquisitions
            subjects['face'][subject]['template'].extend(
                fisherfaces[fisherfaces_index:fisherfaces_index + num_acquisitions]
            )

            if face_config.save_image.features_extracted_face_image:
                for i in range(num_acquisitions):
                    save_image(visual_fisherfaces[i + fisherfaces_index], face_config, f"{subject}_{i}", f"{face_config.features_extractor}_face_image")

            # Aggiorna l'indice per il prossimo soggetto
            fisherfaces_index += num_acquisitions



    #-----------------------------------------------------
    # 6 - FIRST MULTIMODAL SYSTEM - FEATURES SCALING PHASE
    #-----------------------------------------------------

    # Concatena i template
    gait_templates = [template for subject in subjects['gait'].values() for template in subject['template']]
    face_templates = [template for subject in subjects['face'].values() for template in subject['template']]
    ear_dx_templates = [template for subject in subjects['ear_dx'].values() for template in subject['template']]
    ear_sx_templates = [template for subject in subjects['ear_sx'].values() for template in subject['template']]

    print("Shape di un template di gait_templates: ", gait_templates[0].shape)
    print("Shape di un template di face_templates: ", face_templates[0].shape)
    print("Shape di un template di ear_dx_templates: ", ear_dx_templates[0].shape)
    print("Shape di un template di ear_sx_templates: ", ear_sx_templates[0].shape)


    # gait_scaler = FeaturesScalingMultimodal()
    face_scaler = FeaturesScalingMultimodal()
    ear_dx_scaler = FeaturesScalingMultimodal()
    ear_sx_scaler = FeaturesScalingMultimodal()
    
    gait_scaler.fit_scaler(gait_templates)
    face_scaler.fit_scaler(face_templates)
    ear_dx_scaler.fit_scaler(ear_dx_templates)
    ear_sx_scaler.fit_scaler(ear_sx_templates)



    #----------------------------------------------------
    # 7 - FIRST MULTIMODAL SYSTEM - FEATURES FUSION PHASE
    #----------------------------------------------------

    features_fusion = FeaturesFusionPCA(multimodal_config)

    combined_templates = []

    for gait_template, face_template, ear_dx_template, ear_sx_template in zip(gait_templates, face_templates, ear_dx_templates, ear_sx_templates):
        gait_features = gait_scaler.scaling(gait_template)
        face_features = face_scaler.scaling(face_template)
        ear_dx_features = ear_dx_scaler.scaling(ear_dx_template)
        ear_sx_features = ear_sx_scaler.scaling(ear_sx_template)

        combined_features = features_fusion.weighted_concatenation(gait_features, face_features, ear_dx_features, ear_sx_features)

        combined_templates.append(combined_features)
    
    # Convertiamo la lista in un array numpy di forma (n_samples, n_features)
    combined_templates = np.vstack(combined_templates)

    print("Shape finale concatenated_templates:", combined_templates.shape)

    fused_templates = features_fusion.features_fusion_pca(combined_templates)

    # Output
    print("Forma dei template fusi:", fused_templates.shape)

    # Itera su subjects['fused'] e assegna i template fusi
    fused_template_index = 0  # Indice per tracciare la posizione nel fused_templates

    for subject in subjects['fused'].keys():

        num_acquisitions = len(subjects['fused'][subject]['acquisition_name'])

        # Aggiungi i template fusi al soggetto a blocchi di 5
        subjects['fused'][subject]['template'].extend(
            fused_templates[fused_template_index:fused_template_index + num_acquisitions]
        )

        # Aggiorna l'indice per il prossimo soggetto
        fused_template_index += num_acquisitions



    #------------------------------------------------------
    # 8 - SECOND MULTIMODAL SYSTEM - FEATURES SCALING PHASE
    #------------------------------------------------------

    gait_features_fusion = FeaturesFusionPCA(multimodal_config)
    face_features_fusion = FeaturesFusionPCA(multimodal_config)
    ear_dx_features_fusion = FeaturesFusionPCA(multimodal_config)
    ear_sx_features_fusion = FeaturesFusionPCA(multimodal_config)

    reduced_gait_templates_list = []
    reduced_face_templates_list = []
    reduced_ear_dx_templates_list = []
    reduced_ear_sx_templates_list = []

    for gait_template, face_template, ear_dx_template, ear_sx_template in zip(gait_templates, face_templates, ear_dx_templates, ear_sx_templates):
        gait_features = gait_scaler.scaling(gait_template)
        face_features = face_scaler.scaling(face_template)
        ear_dx_features = ear_dx_scaler.scaling(ear_dx_template)
        ear_sx_features = ear_sx_scaler.scaling(ear_sx_template)

        reduced_gait_templates_list.append(gait_features)
        reduced_face_templates_list.append(face_features)
        reduced_ear_dx_templates_list.append(ear_dx_features)
        reduced_ear_sx_templates_list.append(ear_sx_features)

    reduced_gait_templates = np.vstack(reduced_gait_templates_list)
    reduced_face_templates = np.vstack(reduced_face_templates_list)
    reduced_ear_dx_templates = np.vstack(reduced_ear_dx_templates_list)
    reduced_ear_sx_templates = np.vstack(reduced_ear_sx_templates_list)

    print("Shape finale reduced_gait_templates:", reduced_gait_templates.shape)
    print("Shape finale reduced_face_templates:", reduced_face_templates.shape)
    print("Shape finale reduced_ear_dx_templates:", reduced_ear_dx_templates.shape)
    print("Shape finale reduced_ear_sx_templates:", reduced_ear_sx_templates.shape)



    #-----------------------------------------------------
    # 9 - SECOND MULTIMODAL SYSTEM - FEATURES FUSION PHASE
    #-----------------------------------------------------

    reduced_gait_templates = gait_features_fusion.features_fusion_pca(reduced_gait_templates)
    reduced_face_templates = face_features_fusion.features_fusion_pca(reduced_face_templates)
    reduced_ear_dx_templates = ear_dx_features_fusion.features_fusion_pca(reduced_ear_dx_templates)
    reduced_ear_sx_templates = ear_sx_features_fusion.features_fusion_pca(reduced_ear_sx_templates)

    # Output
    print("Forma dei template reduced_gait_templates fusi:", reduced_gait_templates.shape)
    print("Forma dei template reduced_face_templates fusi:", reduced_face_templates.shape)
    print("Forma dei template reduced_ear_dx_templates fusi:", reduced_ear_dx_templates.shape)
    print("Forma dei template reduced_ear_sx_templates fusi:", reduced_ear_sx_templates.shape)

    features_fusion_red = FeaturesFusionPCA(multimodal_config)

    combined_templates = []

    # for gait_template, face_template, ear_dx_template, ear_sx_template in zip(gait_templates, face_templates, ear_dx_templates, ear_sx_templates):
    for gait_features, face_features, ear_dx_features, ear_sx_features in zip(reduced_gait_templates.tolist(), reduced_face_templates.tolist(), reduced_ear_dx_templates.tolist(), reduced_ear_sx_templates.tolist()):
        # gait_features = gait_scaler.scaling(face_template)
        # face_features = face_scaler.scaling(face_template)
        # ear_dx_features = ear_dx_scaler.scaling(ear_dx_template)
        # ear_sx_features = ear_sx_scaler.scaling(ear_sx_template)

        gait_features = np.array(gait_features, dtype=np.float32)
        face_features = np.array(face_features, dtype=np.float32)
        ear_dx_features = np.array(ear_dx_features, dtype=np.float32)
        ear_sx_features = np.array(ear_sx_features, dtype=np.float32)

        # **Forza ogni array a essere bidimensionale (1, N)**
        gait_features = gait_features.reshape(1, -1)
        face_features = face_features.reshape(1, -1)
        ear_dx_features = ear_dx_features.reshape(1, -1)
        ear_sx_features = ear_sx_features.reshape(1, -1)

        combined_features = features_fusion_red.weighted_concatenation(gait_features, face_features, ear_dx_features, ear_sx_features)

        combined_templates.append(combined_features)
    
    # Convertiamo la lista in un array numpy di forma (n_samples, n_features)
    combined_templates = np.vstack(combined_templates)

    print("Shape finale concatenated_templates:", combined_templates.shape)

    fused_templates = features_fusion_red.features_fusion_pca(combined_templates)

    # Output
    print("Forma dei template fusi:", fused_templates.shape)

    # Itera su subjects['fused'] e assegna i template fusi
    fused_template_index = 0  # Indice per tracciare la posizione nel fused_templates

    for subject in subjects['fused_reduced'].keys():

        num_acquisitions = len(subjects['fused_reduced'][subject]['acquisition_name'])

        # Aggiungi i template fusi al soggetto a blocchi di 5
        subjects['fused_reduced'][subject]['template'].extend(
            fused_templates[fused_template_index:fused_template_index + num_acquisitions]
        )

        # Aggiorna l'indice per il prossimo soggetto
        fused_template_index += num_acquisitions



    #--------------------
    # 10 - MATCHING PHASE
    #--------------------

    # ----
    # GAIT
    # ----

    far, fa, t_imp = matching_gait.calculate_far(subjects['gait'])
    frr, fr, t_legit = matching_gait.calculate_frr(subjects['gait'])
    accuracy = matching_gait.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Gait matching metrics:")
    print(f"FAR: {far:.4f} %")
    print(f"FRR: {frr:.4f} %")
    print(f"accuracy: {accuracy:.4f} %")

    matching_gait.calculate_roc_and_det(subjects['gait'])
    matching_gait.far_vs_frr(subjects['gait'])

    print("")

    # ----
    # FACE
    # ----

    far, fa, t_imp = matching_face.calculate_far(subjects['face'])
    frr, fr, t_legit = matching_face.calculate_frr(subjects['face'])
    accuracy = matching_face.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Face matching metrics:")
    print(f"FAR: {far:.4f} %")
    print(f"FRR: {frr:.4f} %")
    print(f"accuracy: {accuracy:.4f} %")

    matching_face.calculate_roc_and_det(subjects['face'])
    matching_face.far_vs_frr(subjects['face'])

    print("")
    
    # ------
    # EAR DX
    # ------

    far_dx, fa_dx, t_imp_dx = matching_dx_ear.calculate_far(subjects['ear_dx'], 'dx')
    frr_dx, fr_dx, t_legit_dx = matching_dx_ear.calculate_frr(subjects['ear_dx'], 'dx')
    accuracy_dx = matching_dx_ear.calculate_accuracy(t_imp_dx, t_legit_dx, fa_dx, fr_dx)

    print("", "Ear dx matching metrics:")
    print(f"FAR dx: {far_dx:.4f} %")
    print(f"FRR dx: {frr_dx:.4f} %")
    print(f"accuracy dx: {accuracy_dx:.4f} %")

    matching_dx_ear.calculate_roc_and_det(subjects['ear_dx'], 'dx')
    matching_dx_ear.far_vs_frr(subjects['ear_dx'], 'dx')

    print("")

    # ------
    # EAR SX
    # ------

    far_sx, fa_sx, t_imp_sx = matching_sx_ear.calculate_far(subjects['ear_sx'], 'sx')
    frr_sx, fr_sx, t_legit_sx = matching_sx_ear.calculate_frr(subjects['ear_sx'], 'sx')
    accuracy_sx = matching_sx_ear.calculate_accuracy(t_imp_sx, t_legit_sx, fa_sx, fr_sx)

    print("", "Ear sx matching metrics:")
    print(f"FAR sx: {far_sx:.4f} %")
    print(f"FRR sx: {frr_sx:.4f} %")
    print(f"accuracy sx: {accuracy_sx:.4f} %")

    matching_sx_ear.calculate_roc_and_det(subjects['ear_sx'], 'sx')
    matching_sx_ear.far_vs_frr(subjects['ear_sx'], 'sx')

    print("")

    #------------------------
    # FIRST MULTIMODAL SYSTEM
    #------------------------

    far, fa, t_imp = matching_first_multimodal.calculate_far(subjects['fused'])
    frr, fr, t_legit = matching_first_multimodal.calculate_frr(subjects['fused'])
    accuracy = matching_first_multimodal.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "First multimodal features fusion matching metrics:")
    print(f"FAR: {far:.4f} %")
    print(f"FRR: {frr:.4f} %")
    print(f"accuracy: {accuracy:.4f} %")

    matching_first_multimodal.calculate_roc_and_det(subjects['fused'])
    matching_first_multimodal.far_vs_frr(subjects['fused'])

    #-------------------------
    # SECOND MULTIMODAL SYSTEM
    #-------------------------

    far, fa, t_imp = matching_second_multimodal.calculate_far(subjects['fused_reduced'])
    frr, fr, t_legit = matching_second_multimodal.calculate_frr(subjects['fused_reduced'])
    accuracy = matching_second_multimodal.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Second multimodal features fusion matching metrics:")
    print(f"FAR: {far:.4f} %")
    print(f"FRR: {frr:.4f} %")
    print(f"accuracy: {accuracy:.4f} %")

    matching_second_multimodal.calculate_roc_and_det(subjects['fused_reduced'])
    matching_second_multimodal.far_vs_frr(subjects['fused_reduced'])

    #------------------------
    # THIRD MULTIMODAL SYSTEM
    #------------------------

    far, fa, t_imp = matching_third_multimodal.calculate_far(subjects['gait'], subjects['face'], subjects['ear_dx'], subjects['ear_sx'])
    frr, fr, t_legit = matching_third_multimodal.calculate_frr(subjects['gait'], subjects['face'], subjects['ear_dx'], subjects['ear_sx'])
    accuracy = matching_third_multimodal.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Third multimodal score fusion matching metrics:")
    print(f"FAR: {far:.4f} %")
    print(f"FRR: {frr:.4f} %")
    print(f"accuracy: {accuracy:.4f} %")

    matching_third_multimodal.calculate_roc_and_det(subjects['gait'], subjects['face'], subjects['ear_dx'], subjects['ear_sx'])
    matching_third_multimodal.far_vs_frr(subjects['gait'], subjects['face'], subjects['ear_dx'], subjects['ear_sx'])