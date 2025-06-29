import numpy as np
import sys
import os

# Add the parent directory to sys.path to allow imports from data_classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_classes.load_data import LoadData
from pre_processing.pre_processing import FacePreProcessing
from yolo_detection.yolo_detection import Yolo
from post_processing.post_processing import FacePostProcessing
from features_extraction_classes.lbp import LBP
from features_extraction_classes.gabor_wavelet import GaborWavelet
from features_extraction_classes.fisherfaces import FisherFaceExtractor
from metrics_classes.verification import Verification
from metrics_classes.recognition_closed_set import RecognitionClosedSet
from metrics_classes.recognition_open_set import RecognitionOpenSet
from utils import load_config, browse_path, path_extractor, save_image, load_checkpoint, save_checkpoint


if __name__ == '__main__':
    face_config = load_config('config/face_config.yaml')

    if face_config.browse_path:
        face_config.data_dir = browse_path('Select the database folder')
        face_config.save_path = browse_path('Select the folder where images and plots will be saved')

        if face_config.detector == 'CNN':
            face_config.cnn.checkpoints_dir = browse_path('Select the folder that contains CNN model checkpoint')
        if face_config.detector == 'yolo':
            face_config.yolo.checkpoints_dir = browse_path('Select the folder that contains yolo model checkpoint')

    images_data = LoadData()
    images, image_names, image_paths = images_data.load_images(face_config, 'face')

    pre_processing = FacePreProcessing(face_config)

    yolo = Yolo(face_config, 'face')

    post_processing = FacePostProcessing(face_config)

    lbp = LBP(face_config)
    gabor_wavelet = GaborWavelet(face_config)

    if face_config.features_extraction.fisherfaces.load_model:
        fisherface_extractor = FisherFaceExtractor(face_config)

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
        elif face_config.features_extractor == 'fisherface' and face_config.features_extraction.fisherfaces.load_model:
            face_template = fisherface_extractor.extract_fisherface(np.array(post_processed_face_image))
            face_template_vis = fisherface_extractor.extract_visual(face_template, face_config.post_processing.image_size, face_config.post_processing.image_size)
        elif face_config.features_extractor == 'fisherface' and not face_config.features_extraction.fisherfaces.load_model:
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

    if face_config.features_extractor == 'fisherface' and not face_config.features_extraction.fisherfaces.load_model:
        print("Fisherface nooooo:")
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
    
    # Verification phase
    face_verification = Verification(face_config, 'face')

    far, fa, t_imp, ms_far, thr_far = face_verification.calculate_far(subjects)
    frr, fr, t_legit, ms_frr, thr_frr = face_verification.calculate_frr(subjects)
    accuracy = face_verification.calculate_accuracy(t_imp, t_legit, fa, fr)

    print("", "Face verification task metrics:", sep='\n')
    print(f"FAR: {far:.4f} %")
    print(f"Tempo di ricerca (FAR): {ms_far:.4f} ms/probe")
    print(f"Throughput (FAR): {thr_far:.4f} probe/sec")
    print(f"FRR: {frr:.4f} %")
    print(f"Tempo di ricerca (FRR): {ms_frr:.4f} ms/probe")
    print(f"Throughput (FRR): {thr_frr:.4f} probe/sec")
    print(f"Accuracy: {accuracy:.4f} %")

    face_verification.calculate_roc_and_det(subjects)
    face_verification.far_vs_frr(subjects)

    # Recognition closed-set phase
    face_recognition_closed_set = RecognitionClosedSet(face_config, 'face')

    rank1, rank5, mAP, t_ms, tps = face_recognition_closed_set.evaluate_kfold(subjects, max_rank=20)

    print("", "Face recognition (closed-set) task metrics:", sep='\n')
    print(f"Rank-1 medio: {rank1:.4f}%")
    print(f"Rank-5 medio: {rank5:.4f}%")
    print(f"mAP medio: {mAP:.4f}%")
    print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    print(f"Throughput: {tps:.4f} probe/sec")

    # Recognition open-set phase
    face_recognition_open_set = RecognitionOpenSet(face_config, 'face')

    fp, fn, t_ms, tps = face_recognition_open_set.fpir_fnir(subjects, threshold=0.35)
    fpir, fnir, eer, eer_th = face_recognition_open_set.fpir_fnir_curve(subjects)
    fpir_arr, fnir_arr, eer, eer_th = face_recognition_open_set.det_curve(subjects)
    fpir_arr, fnir_arr, dir_arr = face_recognition_open_set.dir_fpir_curve(subjects)

    print("", "Face recognition (open-set) task metrics:", sep='\n')
    print(f"FPIR: {fp:.4f} %")
    print(f"FNIR: {fn:.4f} %")
    print(f"EER: {eer:.4f} %")
    print(f"threshold: {eer_th:.4f} %")
    print(f"Tempo di ricerca: {t_ms:.4f} ms/probe")
    print(f"Throughput: {tps:.4f} probe/sec")
