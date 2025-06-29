import sys
import os
import numpy as np

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_classes.load_data import LoadData
from gait.pre_processing.pre_processing import GaitPreProcessing
from face.pre_processing.pre_processing import FacePreProcessing
from ear.pre_processing.pre_processing import EarPreProcessing
# from gait.pre_processing.gait_keypoints_detection import GaitKeypointsDetection
from yolo_pose_detection.yolo_pose_detection import YoloPose
from yolo_detection.yolo_detection import Yolo
from face.post_processing.post_processing import FacePostProcessing
from ear.post_processing.post_processing import EarPostProcessing
from gait.features_extraction.gait_embedding_extraction import GaitEmbeddingExtraction
from features_extraction_classes.fisherfaces import FisherFaceExtractor
from features_extraction_classes.features_scaling import FeaturesScaling
from metrics_classes.verification import Verification
from metrics_classes.recognition_closed_set import RecognitionClosedSet
from metrics_classes.recognition_open_set import RecognitionOpenSet
from matching_classes.matching_score_fusion import MatchingScoreFusion
from utils import load_config, browse_path, path_extractor, save_image, load_checkpoint, save_checkpoint

# MULTIMODAL CLASSES
from features_fusion.pca import FeaturesFusionPCA


if __name__ == '__main__':
    # LOAD CONFIG FILES
    multimodal_config = load_config('config/multimodal_config.yaml')
    gait_config = load_config('config/gait_config.yaml')
    face_config = load_config('config/face_config.yaml')
    ear_dx_config = load_config('config/ear_dx_config.yaml')
    ear_sx_config = load_config('config/ear_sx_config.yaml')

    # BROWSE PATHS
    if multimodal_config.browse_path:
        multimodal_config.data_dir = browse_path('Select the database folder')
        multimodal_config.save_path = browse_path('Select the folder where images and plots will be saved')
        if gait_config.browse_path:
            gait_config.data_dir = browse_path('Select the database folder')
            gait_config.save_dir = browse_path('Select the folder where images and plots will be saved')
        if face_config.browse_path:
            if face_config.detector == 'yolo':
                face_config.yolo.checkpoints_dir = browse_path('Select the folder that contains face detector yolo model checkpoint')
        if ear_dx_config.browse_path:
            if ear_dx_config.detector == 'yolo':
                ear_dx_config.yolo.checkpoints_dir = browse_path('Select the folder that contains ear dx detector yolo model checkpoint')
        if ear_sx_config.browse_path:
            if ear_sx_config.detector == 'yolo':
                ear_sx_config.yolo.checkpoints_dir = browse_path('Select the folder that contains ear sx detector yolo model checkpoint')

    # INSTATIATE LOAD IMAGES AND FRAMES CLASS OBJECTS
    load_data = LoadData()

    # INSTANTIATE GAIT CLASS OBJECTS
    gait_pre_processing = GaitPreProcessing(gait_config)
    # gait_keypoints_detector = GaitKeypointsDetection(gait_config)
    gait_yolo_pose = YoloPose(gait_config, 'gait')
    gait_embedding_extractor = GaitEmbeddingExtraction(gait_config)

    # INSTANTIATE FACE CLASS OBJECTS
    face_pre_processing = FacePreProcessing(face_config)
    face_yolo = Yolo(face_config, 'face')
    face_post_processing = FacePostProcessing(face_config)
    face_fisherface_extractor = FisherFaceExtractor(face_config)

    # INSTANTIATE EAR DX CLASS OBJECTS
    ear_dx_pre_processing = EarPreProcessing(ear_dx_config)
    ear_dx_yolo = Yolo(ear_dx_config, 'ear_dx')
    ear_dx_post_processing = EarPostProcessing(ear_dx_config)
    ear_dx_fisherface_extractor = FisherFaceExtractor(ear_dx_config)

    ear_sx_pre_processing = EarPreProcessing(ear_sx_config)
    ear_sx_yolo = Yolo(ear_sx_config, 'ear_sx')
    ear_sx_post_processing = EarPostProcessing(ear_sx_config)
    ear_sx_fisherface_extractor = FisherFaceExtractor(ear_sx_config)

    # INSTANTIATE MULTIMODAL CLASS OBJECTS
    gait_scaler = FeaturesScaling(multimodal_config.features_fusion.scaler_type)
    face_scaler = FeaturesScaling(multimodal_config.features_fusion.scaler_type)
    ear_dx_scaler = FeaturesScaling(multimodal_config.features_fusion.scaler_type)
    ear_sx_scaler = FeaturesScaling(multimodal_config.features_fusion.scaler_type)
    features_fusion = FeaturesFusionPCA(multimodal_config)
    gait_features_fusion = FeaturesFusionPCA(multimodal_config)
    face_features_fusion = FeaturesFusionPCA(multimodal_config)
    ear_dx_features_fusion = FeaturesFusionPCA(multimodal_config)
    ear_sx_features_fusion = FeaturesFusionPCA(multimodal_config)
    features_fusion_reduced = FeaturesFusionPCA(multimodal_config)
    multimodal_score_fusion_matching = MatchingScoreFusion(multimodal_config)

    if multimodal_config.use_checkpoint:
        checkpoint = load_checkpoint('checkpoint_multimodal.json')
        start_index = checkpoint['current_index'] if checkpoint else 0

    # INITIALIZE SUBJECTS
    subjects = {
        'gait': {},
        'face': {},
        'ear_dx': {},
        'ear_sx': {},
        'fused': {},
        'fused_reduced': {}
    }

    #-------------------------------
    # 1 - LOAD IMAGES & FRAMES PHASE
    #-------------------------------
    # LOAD GAIT FRAMES
    # gait_frame_sequences, gait_frame_sequences_names, gait_frame_sequences_paths = load_data.load_frames(gait_config, 'gait')
    gait_frame_sequences, gait_frame_sequences_names, gait_frame_sequences_paths, all_subject_ids, all_sequence_names, all_frame_names = load_data.load_frames(gait_config, 'gait')
    # LOAD FACE IMAGES
    face_images, face_image_names, face_image_paths = load_data.load_images(face_config, 'face')
    # LOAD EAR DX IMAGES
    ear_dx_images, ear_dx_image_names, ear_dx_image_paths = load_data.load_images(ear_dx_config, 'ear_dx')
    # LOAD EAR SX IMAGES
    ear_sx_images, ear_sx_image_names, ear_sx_image_paths = load_data.load_images(ear_sx_config, 'ear_sx')

    # # CHECK GAIT, FACE AND EARS ASSOCIATION
    # if not gait_frame_sequences_names == face_image_names == ear_dx_image_names == ear_sx_image_names:
    #     print("Gait, face and ears don't match!")
    #     sys.exit(1)

    for current_index, (acquisition_name, gait_frame_sequence, gait_frame_sequence_path, face_image, face_image_path, ear_dx_image, ear_dx_image_path, ear_sx_image, ear_sx_image_path) in enumerate(zip(face_image_names, gait_frame_sequences, gait_frame_sequences_paths, face_images, face_image_paths, ear_dx_images, ear_dx_image_paths, ear_sx_images, ear_sx_image_paths)):
        # Extract the subject number from the acquisition name
        subject = acquisition_name.split('_')[0]

        print("Acquisition name:", acquisition_name)

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
                'template': []
            }
            subjects['ear_sx'][subject] = {
                'acquisition_name': [],
                'template': []
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

        #-------------------------------
        # 2 - PRE-PROCESSING PHASE
        #-------------------------------
        # GAIT PRE-PROCESSING
        gait_pre_processed_frame_sequence = []
        gait_pre_processing_params_sequence = []

        for i, gait_frame in enumerate(gait_frame_sequence):
            gait_pre_processed_frame, gait_pre_processing_params = gait_pre_processing.pre_processing_frame(gait_frame.copy())

            if multimodal_config.save_images.pre_processed:
                if gait_config.save_image.pre_processed:
                    save_image(gait_config, 'gait', gait_pre_processed_frame, acquisition_name, 'pre_processed_frame_sequence', i + 1)

            gait_pre_processed_frame_sequence.append(gait_pre_processed_frame)
            gait_pre_processing_params_sequence.append(gait_pre_processing_params)

        # FACE PRE-PROCESSING
        face_pre_processed_image = face_pre_processing.pre_processing_image(face_image.copy())
        if multimodal_config.save_images.pre_processed:
            if face_config.save_image.pre_processed:
                save_image(face_config, 'face', face_pre_processed_image, acquisition_name, 'pre_processed_face_image')

        # EAR DX PRE-PROCESSING
        ear_dx_pre_processed_image = ear_dx_pre_processing.pre_processing_image(ear_dx_image.copy())
        if multimodal_config.save_images.pre_processed:
            if ear_dx_config.save_image.pre_processed:
                save_image(ear_dx_config, 'ear_dx', ear_dx_pre_processed_image, acquisition_name, 'pre_processed_ear_dx_image')

        # EAR SX PRE-PROCESSING
        ear_sx_pre_processed_image = ear_sx_pre_processing.pre_processing_image(ear_sx_image.copy())
        if multimodal_config.save_images.pre_processed:
            if ear_sx_config.save_image.pre_processed:
                save_image(ear_sx_config, 'ear_sx', ear_sx_pre_processed_image, acquisition_name, 'pre_processed_ear_sx_image')

        #--------------------
        # 3 - DETECTION PHASE
        #--------------------
        # GAIT KEYPOINTS DETECTION
        gait_keypoints_sequence = []

        # Gait Keypoints Detection phase
        # -----------------------------------------------------------
        for i, gait_pre_processed_frame in enumerate(gait_pre_processed_frame_sequence):
            # gait_keypoints, gait_frame_with_detected_keypoints = gait_keypoints_detector.detect_keypoints(gait_pre_processed_frame)

            pre_processed_frame_path = path_extractor(gait_config, 'gait', acquisition_name, 'pre_processed_frame_sequence', i + 1)
            gait_keypoints, gait_frame_with_detected_keypoints = gait_yolo_pose.predict_keypoints(pre_processed_frame_path)

            if multimodal_config.save_images.detected:
                if gait_config.save_image.detected_keypoints:
                    save_image(gait_config, 'gait', gait_frame_with_detected_keypoints, acquisition_name, 'detected_frame_keypoints_sequence', i + 1)

            gait_keypoints_sequence.append(gait_keypoints)

        # FACE DETECTION
        if face_config.detector == 'yolo':
            face_pre_processed_image_path = path_extractor(face_config, 'face', acquisition_name, 'pre_processed_face_image')
            face_detected_image, face_bounding_box = face_yolo.predict_bounding_box(face_pre_processed_image_path)
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        if multimodal_config.save_images.detected:
            if face_config.save_image.detected_face:
                save_image(face_config, 'face', face_detected_image, acquisition_name, 'detected_face_bounding_box')

        # EAR DX DETECTION
        if ear_dx_config.detector == 'yolo':
            ear_dx_pre_processed_image_path = path_extractor(ear_dx_config, 'ear_dx', acquisition_name, 'pre_processed_ear_dx_image')
            ear_dx_detected_image, ear_dx_bounding_box = ear_dx_yolo.predict_bounding_box(ear_dx_pre_processed_image_path)
        elif ear_dx_config.detector == 'None':
            ear_dx_bounding_box = [0, 0, 1, 1]
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        if multimodal_config.save_images.detected:
            if ear_dx_config.save_image.detected and not ear_dx_config.detector == 'None':
                save_image(ear_dx_config, 'ear_dx', ear_dx_detected_image, acquisition_name, 'detected_ear_dx_bounding_box')

        # EAR SX DETECTION
        if ear_sx_config.detector == 'yolo':
            ear_sx_pre_processed_image_path = path_extractor(ear_sx_config, 'ear_sx', acquisition_name, 'pre_processed_ear_sx_image')
            ear_sx_detected_image, ear_sx_bounding_box = ear_sx_yolo.predict_bounding_box(ear_sx_pre_processed_image_path)
        else:
            raise ValueError("Unknown algorithm type! \n")
        
        if multimodal_config.save_images.detected:
            if ear_sx_config.save_image.detected:
                save_image(ear_sx_config, 'ear_sx', ear_sx_detected_image, acquisition_name, 'detected_ear_sx_bounding_box')

        #--------------------------
        # 4 - POST-PROCESSING PHASE
        #--------------------------
        # GAIT KEYPOINTS POST-PROCESSING
        gait_post_processed_keypoints_sequence = []

        for i, (gait_keypoints, gait_pre_processed_frame, gait_pre_processing_params) in enumerate(zip(gait_keypoints_sequence, gait_pre_processed_frame_sequence, gait_pre_processing_params_sequence)):
            post_processed_keypoints = gait_pre_processing.pre_processing_keypoints(gait_keypoints, gait_pre_processing_params)
            gait_post_processed_frame_with_keypoints = gait_pre_processing.visualize_keypoints(post_processed_keypoints, gait_pre_processed_frame)

            if multimodal_config.save_images.post_processed:
                if gait_config.save_image.post_processed:
                    save_image(gait_config, 'gait', gait_post_processed_frame_with_keypoints, acquisition_name, 'post_processed_frame_with_keypoints', i + 1)
            
            gait_post_processed_keypoints_sequence.append(post_processed_keypoints)

        gait_keypoints_sequence = gait_pre_processing.pre_processing_keypoints_sequence(gait_post_processed_keypoints_sequence)

        # FACE POST-PROCESSING
        face_post_processed_image, face_shape = face_post_processing.post_processing_image(face_pre_processed_image.copy(), face_bounding_box)

        if multimodal_config.save_images.post_processed:
            if face_config.save_image.post_processed:
                save_image(face_config, 'face', face_post_processed_image, acquisition_name, 'post_processed_face_image')

        # EAR DX POST-PROCESSING
        ear_dx_post_processed_image, ear_dx_shape = ear_dx_post_processing.post_processing_image(ear_dx_pre_processed_image.copy(), ear_dx_bounding_box)
        if multimodal_config.save_images.post_processed:
            if ear_dx_config.save_image.post_processed:
                save_image(ear_dx_config, 'ear_dx', ear_dx_post_processed_image, acquisition_name, 'post_processed_ear_dx_image')

        # EAR SX POST-PROCESSING
        ear_sx_post_processed_image, ear_sx_shape = ear_sx_post_processing.post_processing_image(ear_sx_pre_processed_image.copy(), ear_sx_bounding_box)
        if multimodal_config.save_images.post_processed:
            if ear_sx_config.save_image.post_processed:
                save_image(ear_sx_config, 'ear_sx', ear_sx_post_processed_image, acquisition_name, 'post_processed_ear_sx_image')

        #------------------------------
        # 5 - FEATURES EXTRACTION PHASE
        #------------------------------
        # GAIT EMBEDDING EXTRACTION
        gait_embedding = gait_embedding_extractor.extract_embedding(gait_keypoints_sequence)

        subjects['gait'][subject]['acquisition_name'].append(acquisition_name)
        subjects['gait'][subject]['template'].append(gait_embedding)

        # FACE FEATURES EXTRACTION
        if face_config.features_extraction.fisherfaces.load_model:
            face_template = face_fisherface_extractor.extract_fisherface(np.array(face_post_processed_image))
            face_template_vis = face_fisherface_extractor.extract_visual(face_template, face_config.post_processing.image_size, face_config.post_processing.image_size)

            subjects['face'][subject]['acquisition_name'].append(acquisition_name)
            subjects['face'][subject]['template'].append(face_template)

            if multimodal_config.save_images.features_extracted:
                if face_config.save_image.features_extracted:
                    save_image(face_config, 'face', face_template_vis, acquisition_name, 'features_extracted_face_image')
        else:
            subjects['face'][subject]['acquisition_name'].append(acquisition_name)
            subjects['face'][subject]['template'].append(face_post_processed_image)

        # EAR DX FEATURES EXTRACTION
        if ear_dx_config.features_extraction.fisherfaces.load_model:
            ear_dx_template = ear_dx_fisherface_extractor.extract_fisherface(np.array(ear_dx_post_processed_image))
            ear_dx_template_vis = ear_dx_fisherface_extractor.extract_visual(ear_dx_template, ear_dx_config.post_processing.image_size, ear_dx_config.post_processing.image_size)

            subjects['ear_dx'][subject]['acquisition_name'].append(acquisition_name)
            subjects['ear_dx'][subject]['template'].append(ear_dx_template)

            if multimodal_config.save_images.features_extracted:
                if ear_dx_config.save_image.features_extracted:
                    save_image(ear_dx_config, 'ear_dx', ear_dx_template_vis, acquisition_name, 'features_extracted_ear_dx_image')
        else:
            subjects['ear_dx'][subject]['acquisition_name'].append(acquisition_name)
            subjects['ear_dx'][subject]['template'].append(ear_dx_post_processed_image)

        # EAR SX FEATURES EXTRACTION
        if ear_sx_config.features_extraction.fisherfaces.load_model:
            ear_sx_template = ear_sx_fisherface_extractor.extract_fisherface(np.array(ear_sx_post_processed_image))
            ear_sx_template_vis = ear_sx_fisherface_extractor.extract_visual(ear_sx_template, ear_sx_config.post_processing.image_size, ear_sx_config.post_processing.image_size)

            subjects['ear_sx'][subject]['acquisition_name'].append(acquisition_name)
            subjects['ear_sx'][subject]['template'].append(ear_sx_template)

            if multimodal_config.save_images.features_extracted:
                if ear_sx_config.save_image.features_extracted:
                    save_image(ear_sx_config, 'ear_sx', ear_sx_template_vis, acquisition_name, 'features_extracted_ear_sx_image')
        else:
            subjects['ear_sx'][subject]['acquisition_name'].append(acquisition_name)
            subjects['ear_sx'][subject]['template'].append(ear_sx_post_processed_image)

        # Salva il checkpoint dopo ogni combinazione
        if multimodal_config.use_checkpoint:
            save_checkpoint('checkpoint_multimodal.json', current_index + 1)

    if not face_config.features_extraction.fisherfaces.load_model:
        # FACE FEATURES EXTRACTION
        face_fisherfaces, face_visual_fisherfaces = face_fisherface_extractor.extract_fisherfaces(subjects['face'], face_config.post_processing.image_size, face_config.post_processing.image_size)

        # Itera su subjects e assegna i templates (fisherfaces)
        face_fisherfaces_index = 0  # Indice per tracciare la posizione nei fisherfaces

        for subject in subjects['face'].keys():
            subjects['face'][subject]['template'] = []

            num_acquisitions = len(subjects['face'][subject]['acquisition_name'])

            # Aggiungi i fisherfaces al soggetto a blocchi di num_acquisitions
            subjects['face'][subject]['template'].extend(
                face_fisherfaces[face_fisherfaces_index:face_fisherfaces_index + num_acquisitions]
            )

            if multimodal_config.save_images.features_extracted:
                if face_config.save_image.features_extracted:
                    for i in range(num_acquisitions):
                        save_image(face_config, 'face', face_visual_fisherfaces[i + face_fisherfaces_index], f"{subject}_{i}", f"{face_config.features_extractor}_face_image")

            # Aggiorna l'indice per il prossimo soggetto
            face_fisherfaces_index += num_acquisitions

    if not ear_dx_config.features_extraction.fisherfaces.load_model:
        # EAR DX FEATURES EXTRACTION
        ear_dx_fisherfaces, ear_dx_visual_fisherfaces = ear_dx_fisherface_extractor.extract_fisherfaces(subjects['ear_dx'], ear_dx_config.post_processing.image_size, ear_dx_config.post_processing.image_size)
        
        # Itera su subjects e assegna i templates (fisherfaces)
        
        ear_dx_fisherfaces_index = 0  # Indice per tracciare la posizione nei fisherfaces
        for subject in subjects['ear_dx'].keys():
            subjects['ear_dx'][subject]['template'] = []

            num_acquisitions = len(subjects['ear_dx'][subject]['acquisition_name'])

            # Aggiungi i fisherfaces al soggetto a blocchi di num_acquisitions
            subjects['ear_dx'][subject]['template'].extend(
                ear_dx_fisherfaces[ear_dx_fisherfaces_index:ear_dx_fisherfaces_index + num_acquisitions]
            )

            if multimodal_config.save_images.features_extracted:
                if ear_dx_config.save_image.features_extracted:
                    for i in range(num_acquisitions):
                        save_image(ear_dx_config, 'ear_dx', ear_dx_visual_fisherfaces[i + ear_dx_fisherfaces_index], f"{subject}_{i}", f"{ear_dx_config.features_extractor}_ear_dx_image")

            # Aggiorna l'indice per il prossimo soggetto
            ear_dx_fisherfaces_index += num_acquisitions

    if not ear_sx_config.features_extraction.fisherfaces.load_model:
        # EAR SX FEATURES EXTRACTION
        ear_sx_fisherfaces, ear_sx_visual_fisherfaces = ear_sx_fisherface_extractor.extract_fisherfaces(subjects['ear_sx'], ear_sx_config.post_processing.image_size, ear_sx_config.post_processing.image_size)
        
        # Itera su subjects e assegna i templates (fisherfaces)
        ear_sx_fisherfaces_index = 0  # Indice per tracciare la posizione nei fisherfaces
        
        for subject in subjects['ear_sx'].keys():
            subjects['ear_sx'][subject]['template'] = []

            num_acquisitions = len(subjects['ear_sx'][subject]['acquisition_name'])

            # Aggiungi i fisherfaces al soggetto a blocchi di num_acquisitions
            subjects['ear_sx'][subject]['template'].extend(
                ear_sx_fisherfaces[ear_sx_fisherfaces_index:ear_sx_fisherfaces_index + num_acquisitions]
            )

            if multimodal_config.save_images.features_extracted:
                if ear_sx_config.save_image.features_extracted:
                    for i in range(num_acquisitions):
                        save_image(ear_sx_config, 'ear_sx', ear_sx_visual_fisherfaces[i + ear_sx_fisherfaces_index], f"{subject}_{i}", f"{ear_sx_config.features_extractor}_ear_sx_image")

            # Aggiorna l'indice per il prossimo soggetto
            ear_sx_fisherfaces_index += num_acquisitions



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
    
    gait_scaler.fit_scaler(gait_templates, multimodal=True)
    face_scaler.fit_scaler(face_templates, multimodal=True)
    ear_dx_scaler.fit_scaler(ear_dx_templates, multimodal=True)
    ear_sx_scaler.fit_scaler(ear_sx_templates, multimodal=True)



    #----------------------------------------------------
    # 7 - FIRST MULTIMODAL SYSTEM - FEATURES FUSION PHASE
    #----------------------------------------------------

    combined_templates = []

    for gait_template, face_template, ear_dx_template, ear_sx_template in zip(gait_templates, face_templates, ear_dx_templates, ear_sx_templates):
        gait_features = gait_scaler.scaling(gait_template, multimodal=True)
        face_features = face_scaler.scaling(face_template, multimodal=True)
        ear_dx_features = ear_dx_scaler.scaling(ear_dx_template, multimodal=True)
        ear_sx_features = ear_sx_scaler.scaling(ear_sx_template, multimodal=True)

        if multimodal_config.features_fusion.ear_dx:
            combined_features = features_fusion.weighted_concatenation(gait_features, face_features, ear_dx_features)
        else:
            combined_features = features_fusion.weighted_concatenation(gait_features, face_features, ear_sx_features)

        combined_templates.append(combined_features)
    
    # Convertiamo la lista in un array numpy di forma (n_samples, n_features)
    combined_templates = np.vstack(combined_templates)

    print("Shape finale concatenated_templates:", combined_templates.shape)

    fused_templates = features_fusion.features_fusion_pca(combined_templates, "multimodal_system_features_early_fusion_pca_cumulative_variance")

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

    reduced_gait_templates_list = []
    reduced_face_templates_list = []
    reduced_ear_dx_templates_list = []
    reduced_ear_sx_templates_list = []

    for gait_template, face_template, ear_dx_template, ear_sx_template in zip(gait_templates, face_templates, ear_dx_templates, ear_sx_templates):
        gait_features = gait_scaler.scaling(gait_template, multimodal=True)
        face_features = face_scaler.scaling(face_template, multimodal=True)
        ear_dx_features = ear_dx_scaler.scaling(ear_dx_template, multimodal=True)
        ear_sx_features = ear_sx_scaler.scaling(ear_sx_template, multimodal=True)

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

    reduced_gait_templates = gait_features_fusion.features_fusion_pca(reduced_gait_templates, "gait_pca_cumulative_variance")
    reduced_face_templates = face_features_fusion.features_fusion_pca(reduced_face_templates, "face_pca_cumulative_variance")
    reduced_ear_dx_templates = ear_dx_features_fusion.features_fusion_pca(reduced_ear_dx_templates, "ear_dx_pca_cumulative_variance")
    reduced_ear_sx_templates = ear_sx_features_fusion.features_fusion_pca(reduced_ear_sx_templates, "ear_sx_pca_cumulative_variance")

    # Output
    print("Forma dei template reduced_gait_templates fusi:", reduced_gait_templates.shape)
    print("Forma dei template reduced_face_templates fusi:", reduced_face_templates.shape)
    print("Forma dei template reduced_ear_dx_templates fusi:", reduced_ear_dx_templates.shape)
    print("Forma dei template reduced_ear_sx_templates fusi:", reduced_ear_sx_templates.shape)

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

        if multimodal_config.features_fusion.ear_dx:
            combined_features = features_fusion_reduced.weighted_concatenation(gait_features, face_features, ear_dx_features)
        else:
            combined_features = features_fusion_reduced.weighted_concatenation(gait_features, face_features, ear_sx_features)

        combined_templates.append(combined_features)
    
    # Convertiamo la lista in un array numpy di forma (n_samples, n_features)
    combined_templates = np.vstack(combined_templates)

    print("Shape finale concatenated_templates:", combined_templates.shape)

    fused_templates = features_fusion_reduced.features_fusion_pca(combined_templates, "multimodal_system_features_late_fusion_pca_cumulative_variance")

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

    # # ----
    # # GAIT
    # # ----

    # # Verification phase
    # gait_verification = Verification(gait_config, 'gait')

    # far, fa, t_imp, ms_far, thr_far = gait_verification.calculate_far(subjects['gait'])
    # frr, fr, t_legit, ms_frr, thr_frr = gait_verification.calculate_frr(subjects['gait'])
    # accuracy = gait_verification.calculate_accuracy(t_imp, t_legit, fa, fr)

    # print("", "Gait verification task metrics:", sep='\n')
    # print(f"FAR: {far:.4f} %")
    # print(f"Tempo di ricerca (FAR): {ms_far:.4f} ms/probe")
    # print(f"Throughput (FAR): {thr_far:.4f} probe/sec")
    # print(f"FRR: {frr:.4f} %")
    # print(f"Tempo di ricerca (FRR): {ms_frr:.4f} ms/probe")
    # print(f"Throughput (FRR): {thr_frr:.4f} probe/sec")
    # print(f"Accuracy: {accuracy:.4f} %")

    # gait_verification.calculate_roc_and_det(subjects['gait'])
    # gait_verification.far_vs_frr(subjects['gait'])

    # # Recognition closed-set phase
    # gait_recognition_closed_set = RecognitionClosedSet(gait_config, 'gait')

    # rank1, rank5, mAP, t_ms, tps = gait_recognition_closed_set.evaluate_kfold(subjects['gait'], max_rank=20)

    # print("", "Gait recognition (closed-set) task metrics:", sep='\n')
    # print(f"Rank-1 medio: {rank1:.4f}%")
    # print(f"Rank-5 medio: {rank5:.4f}%")
    # print(f"mAP medio: {mAP:.4f}%")
    # print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    # print(f"Throughput: {tps:.4f} probe/sec")

    # # Recognition open-set phase
    # gait_recognition_open_set = RecognitionOpenSet(gait_config, 'gait')

    # fp, fn, t_ms, tps = gait_recognition_open_set.fpir_fnir(subjects['gait'], threshold=0.35)
    # fpir, fnir, eer, eer_th = gait_recognition_open_set.fpir_fnir_curve(subjects['gait'])
    # fpir_arr, fnir_arr, eer, eer_th = gait_recognition_open_set.det_curve(subjects['gait'])
    # fpir_arr, fnir_arr, dir_arr = gait_recognition_open_set.dir_fpir_curve(subjects['gait'])

    # print("", "Gait recognition (open-set) task metrics:", sep='\n')
    # print(f"FPIR: {fp:.4f} %")
    # print(f"FNIR: {fn:.4f} %")
    # print(f"EER: {eer:.4f} %")
    # print(f"threshold: {eer_th:.4f} %")
    # print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    # print(f"Throughput: {tps:.4f} probe/sec")

    # print("")

    # # ----
    # # FACE
    # # ----

    # # Verification phase
    # face_verification = Verification(face_config, 'face')

    # far, fa, t_imp, ms_far, thr_far = face_verification.calculate_far(subjects['face'])
    # frr, fr, t_legit, ms_frr, thr_frr = face_verification.calculate_frr(subjects['face'])
    # accuracy = face_verification.calculate_accuracy(t_imp, t_legit, fa, fr)

    # print("", "Face verification task metrics:", sep='\n')
    # print(f"FAR: {far:.4f} %")
    # print(f"Tempo di ricerca (FAR): {ms_far:.4f} ms/probe")
    # print(f"Throughput (FAR): {thr_far:.4f} probe/sec")
    # print(f"FRR: {frr:.4f} %")
    # print(f"Tempo di ricerca (FRR): {ms_frr:.4f} ms/probe")
    # print(f"Throughput (FRR): {thr_frr:.4f} probe/sec")
    # print(f"Accuracy: {accuracy:.4f} %")

    # face_verification.calculate_roc_and_det(subjects['face'])
    # face_verification.far_vs_frr(subjects['face'])

    # # Recognition closed-set phase
    # face_recognition_closed_set = RecognitionClosedSet(face_config, 'face')

    # rank1, rank5, mAP, t_ms, tps = face_recognition_closed_set.evaluate_kfold(subjects['face'], max_rank=20)

    # print("", "Face recognition (closed-set) task metrics:", sep='\n')
    # print(f"Rank-1 medio: {rank1:.4f}%")
    # print(f"Rank-5 medio: {rank5:.4f}%")
    # print(f"mAP medio: {mAP:.4f}%")
    # print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    # print(f"Throughput: {tps:.4f} probe/sec")

    # # Recognition open-set phase
    # face_recognition_open_set = RecognitionOpenSet(face_config['face'], 'face')

    # fp, fn, t_ms, tps = face_recognition_open_set.fpir_fnir(subjects['face'], threshold=0.35)
    # fpir, fnir, eer, eer_th = face_recognition_open_set.fpir_fnir_curve(subjects['face'])
    # fpir_arr, fnir_arr, eer, eer_th = face_recognition_open_set.det_curve(subjects['face'])
    # fpir_arr, fnir_arr, dir_arr = face_recognition_open_set.dir_fpir_curve(subjects['face'])

    # print("", "Face recognition (open-set) task metrics:", sep='\n')
    # print(f"FPIR: {fp:.4f} %")
    # print(f"FNIR: {fn:.4f} %")
    # print(f"EER: {eer:.4f} %")
    # print(f"threshold: {eer_th:.4f} %")
    # print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    # print(f"Throughput: {tps:.4f} probe/sec")

    # print("")
    
    # # ------
    # # EAR DX
    # # ------

    # # Verification phase
    # ear_dx_verification = Verification(ear_dx_config, 'ear_dx')

    # far, fa, t_imp, ms_far, thr_far = ear_dx_verification.calculate_far(subjects['ear_dx'])
    # frr, fr, t_legit, ms_frr, thr_frr = ear_dx_verification.calculate_frr(subjects['ear_dx'])
    # accuracy = ear_dx_verification.calculate_accuracy(t_imp, t_legit, fa, fr)

    # print("", "Ear dx verification task metrics:", sep='\n')
    # print(f"FAR: {far:.4f} %")
    # print(f"Tempo di ricerca (FAR): {ms_far:.4f} ms/probe")
    # print(f"Throughput (FAR): {thr_far:.4f} probe/sec")
    # print(f"FRR: {frr:.4f} %")
    # print(f"Tempo di ricerca (FRR): {ms_frr:.4f} ms/probe")
    # print(f"Throughput (FRR): {thr_frr:.4f} probe/sec")
    # print(f"Accuracy: {accuracy:.4f} %")

    # ear_dx_verification.calculate_roc_and_det(subjects['ear_dx'])
    # ear_dx_verification.far_vs_frr(subjects['ear_dx'])

    # # Recognition closed-set phase
    # ear_dx_recognition_closed_set = RecognitionClosedSet(ear_dx_config, 'ear_dx')

    # rank1, rank5, mAP, t_ms, tps = ear_dx_recognition_closed_set.evaluate_kfold(subjects['ear_dx'], max_rank=20)

    # print("", "Ear dx recognition (closed-set) task metrics:", sep='\n')
    # print(f"Rank-1 medio: {rank1:.4f}%")
    # print(f"Rank-5 medio: {rank5:.4f}%")
    # print(f"mAP medio: {mAP:.4f}%")
    # print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    # print(f"Throughput: {tps:.4f} probe/sec")

    # # Recognition open-set phase
    # ear_dx_recognition_open_set = RecognitionOpenSet(ear_dx_config, 'ear_dx')

    # fp, fn, t_ms, tps = ear_dx_recognition_open_set.fpir_fnir(subjects['ear_dx'], threshold=0.35)
    # fpir, fnir, eer, eer_th = ear_dx_recognition_open_set.fpir_fnir_curve(subjects['ear_dx'])
    # fpir_arr, fnir_arr, eer, eer_th = ear_dx_recognition_open_set.det_curve(subjects['ear_dx'])
    # fpir_arr, fnir_arr, dir_arr = ear_dx_recognition_open_set.dir_fpir_curve(subjects['ear_dx'])

    # print("", "Ear dx recognition (open-set) task metrics:", sep='\n')
    # print(f"FPIR: {fp:.4f} %")
    # print(f"FNIR: {fn:.4f} %")
    # print(f"EER: {eer:.4f} %")
    # print(f"threshold: {eer_th:.4f} %")
    # print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    # print(f"Throughput: {tps:.4f} probe/sec")

    # print("")

    # # ------
    # # EAR SX
    # # ------

    # # Verification phase
    # ear_sx_verification = Verification(ear_sx_config, 'ear_sx')

    # far, fa, t_imp, ms_far, thr_far = ear_sx_verification.calculate_far(subjects['ear_sx'])
    # frr, fr, t_legit, ms_frr, thr_frr = ear_sx_verification.calculate_frr(subjects['ear_sx'])
    # accuracy = ear_sx_verification.calculate_accuracy(t_imp, t_legit, fa, fr)

    # print("", "Ear sx verification task metrics:", sep='\n')
    # print(f"FAR: {far:.4f} %")
    # print(f"Tempo di ricerca (FAR): {ms_far:.4f} ms/probe")
    # print(f"Throughput (FAR): {thr_far:.4f} probe/sec")
    # print(f"FRR: {frr:.4f} %")
    # print(f"Tempo di ricerca (FRR): {ms_frr:.4f} ms/probe")
    # print(f"Throughput (FRR): {thr_frr:.4f} probe/sec")
    # print(f"Accuracy: {accuracy:.4f} %")

    # ear_sx_verification.calculate_roc_and_det(subjects['ear_sx'])
    # ear_sx_verification.far_vs_frr(subjects['ear_sx'])

    # # Recognition closed-set phase
    # ear_sx_recognition_closed_set = RecognitionClosedSet(ear_sx_config, 'ear_sx')

    # rank1, rank5, mAP, t_ms, tps = ear_sx_recognition_closed_set.evaluate_kfold(subjects['ear_sx'], max_rank=20)

    # print("", "Ear sx recognition (closed-set) task metrics:", sep='\n')
    # print(f"Rank-1 medio: {rank1:.4f}%")
    # print(f"Rank-5 medio: {rank5:.4f}%")
    # print(f"mAP medio: {mAP:.4f}%")
    # print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    # print(f"Throughput: {tps:.4f} probe/sec")

    # # Recognition open-set phase
    # ear_sx_recognition_open_set = RecognitionOpenSet(ear_sx_config, 'ear_sx')

    # fp, fn, t_ms, tps = ear_sx_recognition_open_set.fpir_fnir(subjects['ear_sx'], threshold=0.35)
    # fpir, fnir, eer, eer_th = ear_sx_recognition_open_set.fpir_fnir_curve(subjects['ear_sx'])
    # fpir_arr, fnir_arr, eer, eer_th = ear_sx_recognition_open_set.det_curve(subjects['ear_sx'])
    # fpir_arr, fnir_arr, dir_arr = ear_sx_recognition_open_set.dir_fpir_curve(subjects['ear_sx'])

    # print("", "Ear sx recognition (open-set) task metrics:", sep='\n')
    # print(f"FPIR: {fp:.4f} %")
    # print(f"FNIR: {fn:.4f} %")
    # print(f"EER: {eer:.4f} %")
    # print(f"threshold: {eer_th:.4f} %")
    # print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    # print(f"Throughput: {tps:.4f} probe/sec")

    # print("")

    #------------------------
    # FIRST MULTIMODAL SYSTEM
    #------------------------

    # Verification phase
    multimodal_early_fusion_verification = Verification(multimodal_config, 'multimodal_early_fusion')

    far, fa, t_imp, ms_far, thr_far = multimodal_early_fusion_verification.calculate_far(subjects['fused'])
    frr, fr, t_legit, ms_frr, thr_frr = multimodal_early_fusion_verification.calculate_frr(subjects['fused'])
    accuracy = multimodal_early_fusion_verification.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Multimodal early fusion verification task metrics:", sep='\n')
    print(f"FAR: {far:.4f} %")
    print(f"Tempo di ricerca (FAR): {ms_far:.4f} ms/probe")
    print(f"Throughput (FAR): {thr_far:.4f} probe/sec")
    print(f"FRR: {frr:.4f} %")
    print(f"Tempo di ricerca (FRR): {ms_frr:.4f} ms/probe")
    print(f"Throughput (FRR): {thr_frr:.4f} probe/sec")
    print(f"Accuracy: {accuracy:.4f} %")

    multimodal_early_fusion_verification.calculate_roc_and_det(subjects['fused'])
    multimodal_early_fusion_verification.far_vs_frr(subjects['fused'])

    # Recognition closed-set phase
    multimodal_early_fusion_recognition_closed_set = RecognitionClosedSet(multimodal_config, 'multimodal_early_fusion')

    rank1, rank5, mAP, t_ms, tps = multimodal_early_fusion_recognition_closed_set.evaluate_kfold(subjects['fused'], max_rank=20)

    print("", "Multimodal early fusion recognition (closed-set) task metrics:", sep='\n')
    print(f"Rank-1 medio: {rank1:.4f}%")
    print(f"Rank-5 medio: {rank5:.4f}%")
    print(f"mAP medio: {mAP:.4f}%")
    print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    print(f"Throughput: {tps:.4f} probe/sec")

    # Recognition open-set phase
    multimodal_early_fusion_recognition_open_set = RecognitionOpenSet(multimodal_config, 'multimodal_early_fusion')

    fp, fn, t_ms, tps = multimodal_early_fusion_recognition_open_set.fpir_fnir(subjects['fused'], threshold=0.35)
    fpir, fnir, eer, eer_th = multimodal_early_fusion_recognition_open_set.fpir_fnir_curve(subjects['fused'])
    fpir_arr, fnir_arr, eer, eer_th = multimodal_early_fusion_recognition_open_set.det_curve(subjects['fused'])
    fpir_arr, fnir_arr, dir_arr = multimodal_early_fusion_recognition_open_set.dir_fpir_curve(subjects['fused'])

    print("", "Multimodal early fusion recognition (open-set) task metrics:", sep='\n')
    print(f"FPIR: {fp:.4f} %")
    print(f"FNIR: {fn:.4f} %")
    print(f"EER: {eer:.4f} %")
    print(f"threshold: {eer_th:.4f} %")
    print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    print(f"Throughput: {tps:.4f} probe/sec")

    print("")

    #-------------------------
    # SECOND MULTIMODAL SYSTEM
    #-------------------------

    # Verification phase
    multimodal_late_fusion_verification = Verification(multimodal_config, 'multimodal_late_fusion')

    far, fa, t_imp, ms_far, thr_far = multimodal_late_fusion_verification.calculate_far(subjects['fused_reduced'])
    frr, fr, t_legit, ms_frr, thr_frr = multimodal_late_fusion_verification.calculate_frr(subjects['fused_reduced'])
    accuracy = multimodal_late_fusion_verification.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Multimodal late fusion verification task metrics:", sep='\n')
    print(f"FAR: {far:.4f} %")
    print(f"Tempo di ricerca (FAR): {ms_far:.4f} ms/probe")
    print(f"Throughput (FAR): {thr_far:.4f} probe/sec")
    print(f"FRR: {frr:.4f} %")
    print(f"Tempo di ricerca (FRR): {ms_frr:.4f} ms/probe")
    print(f"Throughput (FRR): {thr_frr:.4f} probe/sec")
    print(f"Accuracy: {accuracy:.4f} %")

    multimodal_late_fusion_verification.calculate_roc_and_det(subjects['fused_reduced'])
    multimodal_late_fusion_verification.far_vs_frr(subjects['fused_reduced'])

    # Recognition closed-set phase
    multimodal_late_fusion_recognition_closed_set = RecognitionClosedSet(multimodal_config, 'multimodal_late_fusion')

    rank1, rank5, mAP, t_ms, tps = multimodal_late_fusion_recognition_closed_set.evaluate_kfold(subjects['fused_reduced'], max_rank=20)

    print("", "Multimodal late fusion recognition (closed-set) task metrics:", sep='\n')
    print(f"Rank-1 medio: {rank1:.4f}%")
    print(f"Rank-5 medio: {rank5:.4f}%")
    print(f"mAP medio: {mAP:.4f}%")
    print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    print(f"Throughput: {tps:.4f} probe/sec")

    # Recognition open-set phase
    multimodal_late_fusion_recognition_open_set = RecognitionOpenSet(multimodal_config, 'multimodal_late_fusion')

    fp, fn, t_ms, tps = multimodal_late_fusion_recognition_open_set.fpir_fnir(subjects['fused_reduced'], threshold=0.35)
    fpir, fnir, eer, eer_th = multimodal_late_fusion_recognition_open_set.fpir_fnir_curve(subjects['fused_reduced'])
    fpir_arr, fnir_arr, eer, eer_th = multimodal_late_fusion_recognition_open_set.det_curve(subjects['fused_reduced'])
    fpir_arr, fnir_arr, dir_arr = multimodal_late_fusion_recognition_open_set.dir_fpir_curve(subjects['fused_reduced'])

    print("", "Multimodal late fusion recognition (open-set) task metrics:", sep='\n')
    print(f"FPIR: {fp:.4f} %")
    print(f"FNIR: {fn:.4f} %")
    print(f"EER: {eer:.4f} %")
    print(f"threshold: {eer_th:.4f} %")
    print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    print(f"Throughput: {tps:.4f} probe/sec")

    print("")

    #------------------------
    # THIRD MULTIMODAL SYSTEM
    #------------------------
    
    if multimodal_config.features_fusion.ear_dx:
        ear_side = 'dx'
    else:
        ear_side = 'sx'

    far, fa, t_imp = multimodal_score_fusion_matching.calculate_far(subjects['gait'], subjects['face'], subjects[f'ear_{ear_side}'])
    frr, fr, t_legit = multimodal_score_fusion_matching.calculate_frr(subjects['gait'], subjects['face'], subjects[f'ear_{ear_side}'])
    accuracy = multimodal_score_fusion_matching.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Third multimodal score fusion matching metrics:")
    print(f"FAR: {far:.4f} %")
    print(f"FRR: {frr:.4f} %")
    print(f"accuracy: {accuracy:.4f} %")

    multimodal_score_fusion_matching.calculate_roc_and_det(subjects['gait'], subjects['face'], subjects[f'ear_{ear_side}'])
    multimodal_score_fusion_matching.far_vs_frr(subjects['gait'], subjects['face'], subjects[f'ear_{ear_side}'])